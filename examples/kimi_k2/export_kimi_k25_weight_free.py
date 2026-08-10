"""Dual-QPC weight-free export smoke for Kimi K2.5.

This script constructs Kimi K2.5 from config only, asserts the model parameters
remain on the meta device, and exports dual-QPC ONNX graphs with external weight
specs. Use ``--reduced-smoke`` for a fast local check; omit it to exercise the
full checkpoint without loading model weights into RAM at construction time.
"""

import argparse
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

DEFAULT_MODEL_PATH = Path(
    "/home/huggingface_hub/models--moonshotai--Kimi-K2.5/"
    "snapshots/4d01dfe0332d63057c186e0b262165819efb6611"
)
DEFAULT_EXPORT_DIR = Path.home() / ".cache" / "qeff" / "kimi_k25_weight_free_export"
DEFAULT_QEFF_HOME = Path("/tmp/qeff-kimi-k25-weight-free")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Kimi K2.5 dual-QPC ONNX graphs using config-only weight-free export.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local Kimi-K2.5 snapshot path. Uses the local HF cache/default snapshot when omitted.",
    )
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--qeff-home", type=Path, default=DEFAULT_QEFF_HOME)
    parser.add_argument("--hf-hub-cache", type=Path, default=None)
    parser.add_argument("--component", choices=("both", "vision", "lang"), default="both")
    parser.add_argument("--use-onnx-subfunctions", action="store_true")
    parser.add_argument("--skip-export", action="store_true", help="Only validate config-only dual-QPC loading.")
    parser.add_argument(
        "--reduced-smoke",
        action="store_true",
        help="Materialize a tiny Kimi checkpoint slice before config-only export for a faster smoke test.",
    )
    parser.add_argument("--num-vision-layers", type=int, default=1)
    parser.add_argument("--num-text-layers", type=int, default=3)
    parser.add_argument("--expert-ids", type=str, default="0,1,2,3")
    parser.add_argument("--num-experts-per-token", type=int, default=2)
    parser.add_argument(
        "--keep-reduced-source",
        type=Path,
        default=None,
        help="Directory to keep the reduced checkpoint source instead of using a temporary directory.",
    )
    return parser.parse_args()


def configure_environment(args):
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    if args.hf_hub_cache is not None:
        os.environ["HF_HUB_CACHE"] = str(args.hf_hub_cache.expanduser().resolve())
    if args.qeff_home is not None:
        os.environ["QEFF_HOME"] = str(args.qeff_home.expanduser().resolve())


def parse_expert_ids(value: str) -> tuple[int, ...]:
    expert_ids = tuple(int(expert_id) for expert_id in value.split(",") if expert_id.strip())
    if not expert_ids:
        raise ValueError("--expert-ids must contain at least one expert id")
    return expert_ids


def resolve_model_path(model_path: Path | None) -> Path:
    if model_path is not None:
        return model_path.expanduser().resolve()
    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH

    from QEfficient.utils.load_kimi_utils import resolve_model_path as resolve_hf_model_path

    return resolve_hf_model_path()


def write_reduced_checkpoint(source_model_path: Path, destination: Path, args):
    from QEfficient.utils.load_kimi_utils import (
        allowed_prefixes,
        build_layer_subset_config,
        materialize_subset_checkpoint,
        prepare_config,
    )

    config = prepare_config(source_model_path)
    stripped_config, loaded_expert_ids = build_layer_subset_config(
        config,
        args.num_vision_layers,
        args.num_text_layers,
        parse_expert_ids(args.expert_ids),
        args.num_experts_per_token,
    )
    index = json.loads((source_model_path / "model.safetensors.index.json").read_text())
    destination.mkdir(parents=True, exist_ok=True)
    filtered_weight_map, subset_shards = materialize_subset_checkpoint(
        source_model_path,
        destination,
        index["weight_map"],
        allowed_prefixes(args.num_vision_layers, args.num_text_layers),
        loaded_expert_ids,
    )
    for py_file in source_model_path.glob("*.py"):
        shutil.copy2(py_file, destination / py_file.name)
    (destination / "config.json").write_text(stripped_config.to_json_string(use_diff=False))
    (destination / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum((destination / shard).stat().st_size for shard in subset_shards)},
                "weight_map": filtered_weight_map,
            }
        )
    )
    return stripped_config


@contextmanager
def model_source(args):
    source_model_path = resolve_model_path(args.model_path)
    if not args.reduced_smoke:
        from QEfficient.utils.load_kimi_utils import prepare_config

        yield source_model_path, prepare_config(source_model_path)
        return

    if args.keep_reduced_source is not None:
        reduced_source = args.keep_reduced_source.expanduser().resolve()
        if reduced_source.exists():
            shutil.rmtree(reduced_source)
        config = write_reduced_checkpoint(source_model_path, reduced_source, args)
        yield reduced_source, config
        return

    with tempfile.TemporaryDirectory(prefix="qeff_kimi_k25_wf_source_") as source_tmp:
        reduced_source = Path(source_tmp)
        config = write_reduced_checkpoint(source_model_path, reduced_source, args)
        yield reduced_source, config


def assert_config_only_meta(qeff_model):
    allowed_computed_params = {
        "language_model.model.sin_cached",
        "language_model.model.cos_cached",
    }
    real_params = [
        name
        for name, param in qeff_model.model.named_parameters()
        if not param.is_meta and name not in allowed_computed_params
    ]
    if real_params:
        preview = ", ".join(real_params[:8])
        raise RuntimeError(f"Expected config-only meta checkpoint parameters, but found materialized parameters: {preview}")


def export_weight_free(args, model_path: Path, config):
    from QEfficient import QEFFAutoModelForImageTextToText

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        str(model_path),
        config=config,
        kv_offload=True,
        trust_remote_code=True,
        use_weight_free_export=True,
    )
    assert qeff_model.__class__.__name__ == "_QEffAutoModelForImageTextToTextDualQPC", qeff_model.__class__.__name__
    assert_config_only_meta(qeff_model)
    print(f"loaded={qeff_model.__class__.__name__} config_only_meta=True model_path={model_path}")

    if args.skip_export:
        return qeff_model

    skip_vision = args.component == "lang"
    skip_lang = args.component == "vision"
    onnx_paths = qeff_model.export(
        export_dir=str(args.export_dir.expanduser().resolve()),
        skip_vision=skip_vision,
        skip_lang=skip_lang,
        use_onnx_subfunctions=args.use_onnx_subfunctions,
        use_weight_free_export=True,
    )
    print(f"onnx_paths={onnx_paths}")

    if not skip_vision:
        print(f"vision_weight_spec={qeff_model.vision_model.weight_spec_path}")
        assert qeff_model.vision_model.weight_spec_path is not None
        assert Path(qeff_model.vision_model.weight_spec_path).exists()
    if not skip_lang:
        print(f"lang_weight_spec={qeff_model.lang_model.weight_spec_path}")
        assert qeff_model.lang_model.weight_spec_path is not None
        assert Path(qeff_model.lang_model.weight_spec_path).exists()
    return qeff_model


def main():
    args = parse_args()
    configure_environment(args)
    with model_source(args) as (model_path, config):
        export_weight_free(args, model_path, config)


if __name__ == "__main__":
    main()
