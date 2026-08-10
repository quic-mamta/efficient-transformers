"""Export Kimi K2.5 weight-free ONNX, compile QPCs, and compare QAIC tokens with HF."""

import argparse
import copy
import gc
import json
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch
import onnx
from onnx import TensorProto
from safetensors import safe_open
from transformers import AutoProcessor, AutoTokenizer

from export_kimi_k25_dynamo import (
    DEFAULT_IMAGE_URL,
    compile_components,
    configure_qaic_tool_path,
    load_generation_image,
    validate_dynamo_torch_version,
)
from export_kimi_k25_weight_free import assert_config_only_meta, resolve_model_path, write_reduced_checkpoint
from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.load_kimi_utils import (
    DEFAULT_MODEL_PATH,
    LOADED_EXPERT_IDS,
    NUM_EXPERTS_PER_TOKEN,
    NUM_TEXT_LAYERS,
    NUM_VISION_LAYERS,
    ensure_torch_fx_import_compatibility,
    load_kimi_k25_class,
    load_layer_subset_model,
    parse_expert_ids,
    prepare_config,
    set_deterministic,
)

DEFAULT_EXPORT_DIR = Path.home() / ".cache" / "qeff" / "kimi_k25_weight_free_compare_export"
DEFAULT_COMPILE_DIR = Path.home() / ".cache" / "qeff" / "kimi_k25_weight_free_compare_compile"
DEFAULT_QEFF_HOME = Path("/tmp/qeff-kimi-k25-weight-free-compare")
DEFAULT_PROMPT = "Describe this image."


def parse_device_ids(value: str) -> list[int]:
    return [int(device_id) for device_id in value.strip().strip("[]").split(",") if device_id.strip()]


def parse_torch_dtype(value: str):
    if value == "auto":
        return None
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[value]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Kimi K2.5 weight-free ONNX, compile QPCs, and compare QAIC tokens with HF tokens.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Local Kimi-K2.5 snapshot path. Uses HF cache/download fallback when omitted or missing.",
    )
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--compile-dir", type=Path, default=DEFAULT_COMPILE_DIR)
    parser.add_argument("--qeff-home", type=Path, default=DEFAULT_QEFF_HOME)
    parser.add_argument("--hf-hub-cache", type=Path, default=Path("/home/huggingface_hub"))
    parser.add_argument("--vision-onnx-path", type=Path, default=None)
    parser.add_argument("--lang-onnx-path", type=Path, default=None)
    parser.add_argument("--vision-qpc-path", type=Path, default=None)
    parser.add_argument("--lang-qpc-path", type=Path, default=None)
    parser.add_argument("--skip-export", action="store_true", help="Use existing ONNX paths instead of exporting.")
    parser.add_argument("--skip-compile", action="store_true", help="Use existing QPC paths instead of compiling.")
    parser.add_argument(
        "--reduced-smoke",
        action="store_true",
        help="Compare against a reduced HF model slice and export a matching reduced weight-free checkpoint.",
    )
    parser.add_argument("--num-vision-layers", type=int, default=NUM_VISION_LAYERS)
    parser.add_argument("--num-text-layers", type=int, default=NUM_TEXT_LAYERS)
    parser.add_argument("--all-experts", action="store_true", help="Use all routed experts in reduced-smoke mode.")
    parser.add_argument("--expert-ids", type=str, default=",".join(str(expert_id) for expert_id in LOADED_EXPERT_IDS))
    parser.add_argument("--num-experts-per-token", type=int, default=NUM_EXPERTS_PER_TOKEN)
    parser.add_argument(
        "--keep-reduced-source",
        type=Path,
        default=None,
        help="Keep the reduced checkpoint source here instead of using a temporary directory.",
    )
    parser.add_argument("--prefill-seq-len", type=int, default=2)
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--device-ids", type=parse_device_ids, default=[0], help="QAIC device IDs, e.g. [0] or [0,1].")
    parser.add_argument(
        "--auto-device",
        action="store_true",
        help="Use qaic-util to select the first device with free NSPs.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--image-url", type=str, default=DEFAULT_IMAGE_URL)
    parser.add_argument("--image-path", type=Path, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--generation-len", type=int, default=10)
    parser.add_argument("--hf-torch-dtype", choices=("auto", "float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--mxfp6-matmul", action="store_true")
    parser.add_argument("--mxint8-kv-cache", action="store_true")
    parser.add_argument("--mos", type=int, default=1)
    parser.add_argument("--aic-enable-depth-first", action="store_true")
    parser.add_argument("--use-onnx-subfunctions", action="store_true")
    parser.add_argument(
        "--allow-qaic-token-drift",
        action="store_true",
        help="Print mismatches without failing. By default, HF and QAIC tokens must match exactly.",
    )
    args = parser.parse_args()

    if (args.image_height is None) != (args.image_width is None):
        parser.error("--image-height and --image-width must be provided together.")
    if args.skip_export and (args.vision_onnx_path is None or args.lang_onnx_path is None):
        parser.error("--skip-export requires --vision-onnx-path and --lang-onnx-path.")
    if args.skip_compile and (args.vision_qpc_path is None or args.lang_qpc_path is None):
        parser.error("--skip-compile requires --vision-qpc-path and --lang-qpc-path.")
    if args.generation_len < 2:
        parser.error("--generation-len must be >= 2 for QEff dual-QPC generation perf accounting.")
    return args


def configure_environment(args):
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ["HF_HUB_CACHE"] = str(args.hf_hub_cache.expanduser().resolve())
    os.environ["QEFF_HOME"] = str(args.qeff_home.expanduser().resolve())


def has_qaic_runtime_access() -> bool:
    try:
        _ = QAICInferenceSession
        import qaicrt

        _ = qaicrt.Context()
        return True
    except Exception:
        return False


def find_free_qaic_device_id() -> int | None:
    qaic_util_path = Path("/opt/qti-aic/tools/qaic-util")
    if not qaic_util_path.exists():
        return None

    try:
        result = subprocess.run(
            [str(qaic_util_path), "-q"],
            check=False,
            text=True,
            capture_output=True,
            timeout=30,
        )
    except Exception:
        return None

    current_qid = None
    for line in result.stdout.splitlines():
        qid_match = re.match(r"^QID\s+(\d+)", line.strip())
        if qid_match:
            current_qid = int(qid_match.group(1))
            continue

        free_match = re.search(r"Nsp Free:\s*(\d+)", line)
        if free_match and current_qid is not None and int(free_match.group(1)) > 0:
            return current_qid

    return None


def resolve_device_ids(args) -> list[int]:
    if not args.auto_device:
        return args.device_ids

    free_device_id = find_free_qaic_device_id()
    if free_device_id is None:
        raise RuntimeError("Could not find a QAIC device with free NSPs via qaic-util.")
    return [free_device_id]


def resolve_local_model_path(model_path: Path | None) -> Path:
    if model_path is not None:
        candidate = model_path.expanduser().resolve()
        if candidate.exists():
            return candidate
    return resolve_model_path(None)


@contextmanager
def weight_free_model_source(args):
    source_model_path = resolve_local_model_path(args.model_path)
    if not args.reduced_smoke:
        yield source_model_path, source_model_path, prepare_config(source_model_path)
        return

    source_config = prepare_config(source_model_path)
    if args.all_experts:
        num_experts = source_config.text_config.n_routed_experts
        args.expert_ids = ",".join(str(expert_id) for expert_id in range(num_experts))
        args.num_experts_per_token = source_config.text_config.num_experts_per_tok
        print(
            "Using all routed experts for reduced checkpoint: "
            f"experts={num_experts}, num_experts_per_token={args.num_experts_per_token}"
        )

    if args.keep_reduced_source is not None:
        reduced_source = args.keep_reduced_source.expanduser().resolve()
        if reduced_source.exists():
            shutil.rmtree(reduced_source)
        reduced_config = write_reduced_checkpoint(source_model_path, reduced_source, args)
        yield reduced_source, source_model_path, reduced_config
        return

    with tempfile.TemporaryDirectory(prefix="qeff_kimi_k25_wf_compare_source_") as source_tmp:
        reduced_source = Path(source_tmp)
        reduced_config = write_reduced_checkpoint(source_model_path, reduced_source, args)
        yield reduced_source, source_model_path, reduced_config


def build_weight_free_qeff_model(model_path: Path, config):
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        str(model_path),
        config=config,
        kv_offload=True,
        trust_remote_code=True,
        use_weight_free_export=True,
    )
    assert qeff_model.__class__.__name__ == "_QEffAutoModelForImageTextToTextDualQPC", qeff_model.__class__.__name__
    assert_config_only_meta(qeff_model)
    qeff_model.model.eval()
    qeff_model.vision_model.model.eval()
    qeff_model.lang_model.model.eval()
    return qeff_model


def _resolve_extdata_file(base_dir: Path, metadata: dict, file_index: int | str) -> Path:
    file_entry = metadata["files"][file_index] if isinstance(file_index, int) else {"path": file_index}
    file_path = Path(file_entry["path"])
    if file_path.is_absolute():
        return file_path

    candidates = [
        base_dir / file_path,
        Path(metadata["model_id"]).expanduser().resolve().parent / file_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _safe_external_tensor_name(name: str) -> str:
    return name.replace("/", "_").replace(".", "_") + ".bin"


def _add_external_uint8_initializer(
    model,
    tensor_name: str,
    tensor: torch.Tensor,
    external_dir: Path,
    model_dir: Path,
):
    external_dir.mkdir(parents=True, exist_ok=True)
    external_path = external_dir / _safe_external_tensor_name(tensor_name)
    tensor = tensor.contiguous().cpu()
    with external_path.open("wb") as handle:
        handle.write(tensor.numpy().tobytes(order="C"))

    initializer = TensorProto()
    initializer.name = tensor_name
    initializer.data_type = TensorProto.UINT8
    initializer.dims.extend(tensor.shape)
    initializer.data_location = TensorProto.EXTERNAL
    location = external_path.relative_to(model_dir)
    initializer.external_data.add(key="location", value=str(location))
    initializer.external_data.add(key="offset", value="0")
    initializer.external_data.add(key="length", value=str(external_path.stat().st_size))
    model.graph.initializer.append(initializer)


def patch_kimi_uint4_packed_extdata_for_compile(onnx_path: Path) -> Path:
    """Move Kimi packed uint4 qweight/qzeros from QTI extdata to ONNX external initializers."""
    onnx_path = Path(onnx_path).expanduser().resolve()
    model = onnx.load(str(onnx_path), load_external_data=False)
    extdata_prop = next((prop for prop in model.metadata_props if prop.key == "com.qti.aisw.extdata"), None)
    if extdata_prop is None:
        return onnx_path

    metadata = json.loads(extdata_prop.value)
    packed_entries = [
        entry
        for entry in metadata.get("inputs", [])
        if ".mlp.all_" in entry["name"] and entry["name"].endswith(("_qweight", "_qzeros"))
    ]
    if not packed_entries:
        return onnx_path

    existing_initializers = {initializer.name for initializer in model.graph.initializer}
    packed_names = {entry["name"] for entry in packed_entries}
    patched_path = onnx_path.with_name(f"{onnx_path.stem}_compile.onnx")
    external_dir = patched_path.with_suffix("").with_name(f"{patched_path.stem}_external_data")
    for entry in packed_entries:
        if entry["name"] in existing_initializers:
            continue
        location = entry["location"]
        tensor_path = _resolve_extdata_file(onnx_path.parent, metadata, location["file"])
        with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(location["key"]).contiguous()
        _add_external_uint8_initializer(model, entry["name"], tensor, external_dir, patched_path.parent)

    kept_inputs = [graph_input for graph_input in model.graph.input if graph_input.name not in packed_names]
    del model.graph.input[:]
    model.graph.input.extend(kept_inputs)
    metadata["inputs"] = [entry for entry in metadata["inputs"] if entry["name"] not in packed_names]
    extdata_prop.value = json.dumps(metadata, separators=(",", ":"), sort_keys=True)

    onnx.save(model, str(patched_path))
    print(f"Patched Kimi packed uint4 tensors as ONNX external data for compile: {patched_path}")
    return patched_path


def export_weight_free_components(qeff_model, args) -> dict[str, Path]:
    if args.skip_export:
        qeff_model.vision_model.onnx_path = args.vision_onnx_path.expanduser().resolve()
        qeff_model.lang_model.onnx_path = patch_kimi_uint4_packed_extdata_for_compile(
            args.lang_onnx_path.expanduser().resolve()
        )
        return {"vision": qeff_model.vision_model.onnx_path, "lang": qeff_model.lang_model.onnx_path}

    qeff_model.export(
        export_dir=str(args.export_dir.expanduser().resolve()),
        skip_vision=False,
        skip_lang=False,
        use_onnx_subfunctions=args.use_onnx_subfunctions,
        use_weight_free_export=True,
    )
    qeff_model.lang_model.onnx_path = patch_kimi_uint4_packed_extdata_for_compile(qeff_model.lang_model.onnx_path)
    exported_paths = {"vision": qeff_model.vision_model.onnx_path, "lang": qeff_model.lang_model.onnx_path}
    if qeff_model.vision_model.weight_spec_path is None:
        raise RuntimeError("Vision weight-free export did not produce weight_spec.json.")
    if qeff_model.lang_model.weight_spec_path is None:
        raise RuntimeError("Language weight-free export did not produce weight_spec.json.")
    return exported_paths


def load_hf_reference_model(args, source_model_path: Path):
    dtype = parse_torch_dtype(args.hf_torch_dtype)
    set_deterministic(args.seed)
    ensure_torch_fx_import_compatibility()
    config = prepare_config(source_model_path)

    if args.reduced_smoke:
        kimi_cls = load_kimi_k25_class(source_model_path)
        model, tokenizer, processor = load_layer_subset_model(
            model_path=source_model_path,
            kimi_cls=kimi_cls,
            config=config,
            num_vision_layers=args.num_vision_layers,
            num_text_layers=args.num_text_layers,
            loaded_expert_ids=parse_expert_ids(args.expert_ids),
            num_experts_per_tok=args.num_experts_per_token,
            dtype=dtype,
        )
        model.vision_tower.patch_embed.pos_emb.interpolation_mode = "bilinear"
        return model.eval().to("cpu"), tokenizer, processor

    kimi_cls = load_kimi_k25_class(source_model_path)
    model_kwargs = {
        "config": config,
        "trust_remote_code": True,
        "attn_implementation": "eager",
        "output_loading_info": True,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    model, loading_info = kimi_cls.from_pretrained(str(source_model_path), **model_kwargs)
    unexpected_keys = loading_info.get("unexpected_keys", [])
    missing_keys = loading_info.get("missing_keys", [])
    mismatched_keys = loading_info.get("mismatched_keys", [])
    if unexpected_keys or missing_keys or mismatched_keys:
        raise RuntimeError(
            "Failed to load the HF checkpoint cleanly. "
            f"missing={missing_keys}, unexpected={unexpected_keys}, mismatched={mismatched_keys}"
        )
    model.vision_tower.patch_embed.pos_emb.interpolation_mode = "bilinear"
    tokenizer = AutoTokenizer.from_pretrained(str(source_model_path), trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(str(source_model_path), trust_remote_code=True)
    return model.eval().to("cpu"), tokenizer, processor


def clone_inputs(inputs):
    return {name: value.clone() if torch.is_tensor(value) else copy.deepcopy(value) for name, value in inputs.items()}


def build_generation_inputs(processor, image, args, dtype):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": image},
                {"type": "text", "text": args.prompt},
            ],
        },
    ]
    inputs = processor(messages=messages, add_generation_prompt=True, tokenize=False, return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    return inputs


@torch.no_grad()
def greedy_generate_hf(model, inputs, max_new_tokens: int) -> torch.Tensor:
    generated_ids = inputs["input_ids"].to(torch.long)
    attention_mask = inputs["attention_mask"].to(torch.long)
    pixel_values = inputs["pixel_values"]
    grid_thws = inputs["grid_thws"]
    new_tokens = []

    eos_token_id = getattr(model.config, "eos_token_id", None)
    if eos_token_id is None and hasattr(model.config, "text_config"):
        eos_token_id = getattr(model.config.text_config, "eos_token_id", None)

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=generated_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        new_tokens.append(next_token)

        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device),
            ],
            dim=1,
        )

        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break

    return torch.cat(new_tokens, dim=1)


def decode_tokens(tokenizer, token_ids) -> str:
    decoded = tokenizer.batch_decode(torch.as_tensor(token_ids), skip_special_tokens=True)
    return decoded[0] if decoded else ""


def compare_tokens(tokenizer, hf_tokens: torch.Tensor, qaic_tokens: torch.Tensor, args):
    hf_tokens = hf_tokens.cpu()
    qaic_tokens = qaic_tokens.cpu()
    qaic_tokens_to_compare = qaic_tokens[:, : hf_tokens.shape[1]]

    print("HF tokens:", hf_tokens.tolist())
    print("HF text:", repr(decode_tokens(tokenizer, hf_tokens)))
    print("QAIC tokens:", qaic_tokens_to_compare.tolist())
    print("QAIC text:", repr(decode_tokens(tokenizer, qaic_tokens_to_compare)))

    if torch.equal(hf_tokens, qaic_tokens_to_compare):
        print("HF and QAIC generated tokens match exactly.")
        return

    message = f"HF and QAIC tokens differ: hf={hf_tokens.tolist()}, qaic={qaic_tokens_to_compare.tolist()}"
    if args.allow_qaic_token_drift:
        print(f"WARNING: {message}")
        if hf_tokens.shape != qaic_tokens_to_compare.shape:
            raise AssertionError("HF and QAIC generated token shapes do not match.")
        return
    raise AssertionError(message)


def main():
    args = parse_args()
    configure_environment(args)
    validate_dynamo_torch_version()
    configure_qaic_tool_path()
    if not has_qaic_runtime_access():
        raise RuntimeError("QAIC runtime access is required for QAIC token comparison.")

    args.component = "both"
    args.device_ids = resolve_device_ids(args)
    args.num_devices = len(args.device_ids)
    qaic_config = {"mla_absorption": {"cache_compressed": True, "absorption": False, "online": False}}
    set_deterministic(args.seed)

    with weight_free_model_source(args) as (export_model_path, hf_model_path, export_config):
        image = load_generation_image(args)
        qeff_model = build_weight_free_qeff_model(export_model_path, export_config)
        exported_paths = export_weight_free_components(qeff_model, args)
        print(f"Weight-free ONNX paths: {exported_paths}")
        print(f"Vision weight spec: {qeff_model.vision_model.weight_spec_path}")
        print(f"Language weight spec: {qeff_model.lang_model.weight_spec_path}")

        qpc_paths = compile_components(qeff_model, exported_paths, image, qaic_config, args)
        print(f"QPC paths: {qpc_paths}")

        hf_model, tokenizer, processor = load_hf_reference_model(args, hf_model_path)
        inputs = build_generation_inputs(processor, image, args, hf_model.config.torch_dtype)
        hf_tokens = greedy_generate_hf(hf_model, clone_inputs(inputs), args.generation_len)
        del hf_model
        gc.collect()

        qaic_output = qeff_model.generate(
            inputs=clone_inputs(inputs),
            device_ids=args.device_ids,
            generation_len=args.generation_len,
            image_height=image.height,
            image_width=image.width,
        )
        qaic_tokens = torch.as_tensor(qaic_output.generated_ids[:, : args.generation_len], dtype=hf_tokens.dtype)
        compare_tokens(tokenizer, hf_tokens, qaic_tokens, args)


if __name__ == "__main__":
    main()
