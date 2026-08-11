#!/home/mamtsing/13_08/env/bin/python
"""Validate Kimi K2.5 dual-QPC weight-free export and optional compile.

This script intentionally builds the Kimi module from config only. It never
loads a model checkpoint into the PyTorch module and never creates a reduced
layer/expert subset. Checkpoint tensors are referenced through QTI extdata
metadata and remain outside the ONNX protobuf.
"""

import argparse
import json
import os
from pathlib import Path

import onnx
import torch
from onnx import TensorProto

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.utils import constants
from QEfficient.utils.load_kimi_utils import DEFAULT_MODEL_PATH, KIMI_K25_MODEL_NAME, prepare_config, resolve_model_path

DEFAULT_EXPORT_DIR = Path.home() / ".cache" / "qeff" / "kimi_k25_weight_free_export"
DEFAULT_COMPILE_DIR = Path.home() / ".cache" / "qeff" / "kimi_k25_weight_free_compile"
DEFAULT_QEFF_HOME = Path("/tmp/qeff-kimi-k25-weight-free")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Kimi K2.5 with weight-free ONNX and compile dual QPCs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--compile-dir", type=Path, default=DEFAULT_COMPILE_DIR)
    parser.add_argument("--qeff-home", type=Path, default=DEFAULT_QEFF_HOME)
    parser.add_argument("--hf-hub-cache", type=Path, default=Path("/home/huggingface_hub"))
    parser.add_argument("--prefill-seq-len", type=int, default=32)
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--image-height", type=int, default=constants.KIMI_EXAMPLE_IMAGE_NUM_PATCHES_HEIGHT * constants.KIMI_PATCH_SIZE)
    parser.add_argument("--image-width", type=int, default=constants.KIMI_EXAMPLE_IMAGE_NUM_PATCHES_WIDTH * constants.KIMI_PATCH_SIZE)
    parser.add_argument("--expected-experts", type=int, default=384)
    parser.add_argument("--use-onnx-subfunctions", action="store_true")
    parser.add_argument("--mxfp6-matmul", action="store_true")
    parser.add_argument("--mxint8-kv-cache", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--vision-onnx-path", type=Path)
    parser.add_argument("--lang-onnx-path", type=Path)
    return parser.parse_args()


def configure_environment(args) -> None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ["HF_HUB_CACHE"] = str(args.hf_hub_cache.expanduser())
    os.environ["QEFF_HOME"] = str(args.qeff_home.expanduser())


def resolve_local_model_path(model_path: Path) -> Path:
    candidate = model_path.expanduser()
    if candidate.exists():
        return candidate.resolve()
    return resolve_model_path(KIMI_K25_MODEL_NAME).resolve()


def assert_full_kimi_config(config, expected_experts: int) -> None:
    if getattr(config, "model_type", None) != "kimi_k25":
        raise AssertionError(f"Expected Kimi K2.5 config, got model_type={getattr(config, 'model_type', None)!r}")
    text_config = getattr(config, "text_config", None)
    routed_experts = getattr(text_config, "n_routed_experts", None)
    if routed_experts != expected_experts:
        raise AssertionError(f"Expected all {expected_experts} routed experts, got n_routed_experts={routed_experts}")
    if getattr(text_config, "num_hidden_layers", 0) <= 2:
        raise AssertionError(f"Config looks reduced: num_hidden_layers={getattr(text_config, 'num_hidden_layers', None)}")


def assert_config_only_meta(qeff_model) -> None:
    allowed_generated = ("sin_cached", "cos_cached", "inv_freq")
    wrappers = [qeff_model.vision_model.model, qeff_model.lang_model.model]
    non_meta_params = []
    for wrapper in wrappers:
        non_meta_params.extend(
            name
            for name, param in wrapper.named_parameters()
            if not param.is_meta and not name.endswith(allowed_generated)
        )
    if non_meta_params:
        preview = ", ".join(non_meta_params[:10])
        raise AssertionError(f"Model parameters are not config-only/meta tensors: {preview}")


def build_qeff_model(model_path: Path, config):
    qaic_config = {"mla_absorption": {"cache_compressed": True, "absorption": False, "online": False}}
    qeff_model = QEFFAutoModelForImageTextToText.from_config(
        config,
        pretrained_model_name_or_path=str(model_path),
        kv_offload=True,
        continuous_batching=False,
        qaic_config=qaic_config,
        use_weight_free_export=True,
        trust_remote_code=True,
    )
    if qeff_model.__class__.__name__ != "_QEffAutoModelForImageTextToTextDualQPC":
        raise AssertionError(f"Weight-free Kimi must use dual QPC wrapper, got {qeff_model.__class__.__name__}")
    assert_config_only_meta(qeff_model)
    return qeff_model, qaic_config


def validate_weight_free_onnx(onnx_path: Path, component_name: str) -> None:
    if not onnx_path.is_file():
        raise AssertionError(f"Missing {component_name} ONNX: {onnx_path}")
    onnx_data_files = sorted(onnx_path.parent.glob("*.onnx.data"))
    if onnx_data_files:
        raise AssertionError(f"{component_name} export produced ONNX external-data files: {onnx_data_files}")

    model = onnx.load(str(onnx_path), load_external_data=False)
    external_initializers = [
        initializer.name
        for initializer in model.graph.initializer
        if initializer.data_location == TensorProto.EXTERNAL or initializer.external_data
    ]
    if external_initializers:
        raise AssertionError(f"{component_name} ONNX has external initializers instead of QTI extdata: {external_initializers[:10]}")

    extdata_prop = next((prop for prop in model.metadata_props if prop.key == "com.qti.aisw.extdata"), None)
    if extdata_prop is None:
        raise AssertionError(f"{component_name} ONNX is missing com.qti.aisw.extdata metadata")

    metadata = json.loads(extdata_prop.value)
    spec_inputs = {entry["name"] for entry in metadata.get("inputs", [])}
    if not spec_inputs:
        raise AssertionError(f"{component_name} weight spec has no externalized inputs")

    graph_inputs = {value.name: value for value in model.graph.input}
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    embedded_weights = sorted(spec_inputs & initializer_names)
    if embedded_weights:
        raise AssertionError(f"{component_name} extdata weights are embedded as ONNX initializers: {embedded_weights[:10]}")

    missing_inputs = sorted(spec_inputs - set(graph_inputs))
    if missing_inputs:
        raise AssertionError(f"{component_name} extdata weights are missing graph inputs: {missing_inputs[:10]}")

    kimi_int4_inputs = sorted(name for name in spec_inputs if ".mlp.all_" in name and name.endswith(("_qweight", "_qzeros")))
    for name in kimi_int4_inputs:
        elem_type = graph_inputs[name].type.tensor_type.elem_type
        if elem_type != TensorProto.INT32:
            raise AssertionError(f"{component_name} Kimi int4 extdata input {name} dtype={elem_type}, expected INT32")

    print(
        f"Validated {component_name}: extdata_weights={len(spec_inputs)}, "
        f"kimi_int4_inputs={len(kimi_int4_inputs)}, onnx_data_files=0"
    )


def export_weight_free(qeff_model, args) -> dict[str, Path]:
    if args.skip_export:
        if args.vision_onnx_path is None or args.lang_onnx_path is None:
            raise ValueError("--skip-export requires --vision-onnx-path and --lang-onnx-path")
        qeff_model.vision_model.onnx_path = str(args.vision_onnx_path.expanduser().resolve())
        qeff_model.lang_model.onnx_path = str(args.lang_onnx_path.expanduser().resolve())
    else:
        qeff_model.export(
            export_dir=str(args.export_dir.expanduser().resolve()),
            use_weight_free_export=True,
            use_onnx_subfunctions=args.use_onnx_subfunctions,
            prefill_seq_len=args.prefill_seq_len,
            offload_pt_weights=False,
        )

    paths = {
        "vision": Path(qeff_model.vision_model.onnx_path),
        "language": Path(qeff_model.lang_model.onnx_path),
    }
    for component_name, onnx_path in paths.items():
        validate_weight_free_onnx(onnx_path, component_name)
    return paths


def compile_dual_qpc(qeff_model, args, qaic_config):
    qpc_paths = qeff_model.compile(
        compile_dir=str(args.compile_dir.expanduser().resolve()),
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        batch_size=args.batch_size,
        num_devices=args.num_devices,
        num_cores=args.num_cores,
        mxfp6_matmul=args.mxfp6_matmul,
        mxint8_kv_cache=args.mxint8_kv_cache,
        image_height=args.image_height,
        image_width=args.image_width,
        use_onnx_subfunctions=args.use_onnx_subfunctions,
        use_weight_free_export=True,
        qaic_config=qaic_config,
    )
    if not isinstance(qpc_paths, dict) or "vision_qpc_path" not in qpc_paths:
        raise AssertionError(f"Expected dual-QPC compile output dict, got {qpc_paths!r}")
    if not any(key.startswith("lang_") for key in qpc_paths):
        raise AssertionError(f"Expected language QPC path in compile output, got {qpc_paths!r}")
    print(f"Compiled dual QPCs: {qpc_paths}")
    return qpc_paths


def main() -> None:
    args = parse_args()
    configure_environment(args)
    model_path = resolve_local_model_path(args.model_path)
    config = prepare_config(model_path)
    assert_full_kimi_config(config, args.expected_experts)
    qeff_model, qaic_config = build_qeff_model(model_path, config)
    onnx_paths = export_weight_free(qeff_model, args)
    print(f"Weight-free ONNX paths: {onnx_paths}")
    print(f"Vision weight spec: {qeff_model.vision_model.weight_spec_path}")
    print(f"Language weight spec: {qeff_model.lang_model.weight_spec_path}")
    if args.skip_compile:
        print("Skipping compile because --skip-compile was set.")
        return
    compile_dual_qpc(qeff_model, args, qaic_config)


if __name__ == "__main__":
    main()
