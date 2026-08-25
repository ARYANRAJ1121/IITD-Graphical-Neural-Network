from pathlib import Path

from src.config import load_config
from src.data.stage3_bundle import example_index, load_processed_bundle, make_synthetic_bundle
from src.training.validate_stage3 import run_forward, write_validation_report


def main() -> None:
    config_path = Path("configs/base.yaml")
    config = load_config(config_path)
    processed_dir = Path(config["data"]["processed_dir"])
    ratios = tuple(config["training"]["patient_split"])
    seed = int(config["training"]["seed"])

    bundle = load_processed_bundle(processed_dir, seed, ratios)
    processed_loaded = bundle is not None
    if bundle is None:
        bundle = make_synthetic_bundle(seed)

    model, outputs, loss = run_forward(bundle)
    splits = example_index(bundle.patient_ids, bundle.pair_mask.numpy(), bundle.split_ids)
    split_counts = {name: int(idx.size) for name, idx in splits.items()}

    report_path = Path("experiments/stage3_implementation_validation.md")
    write_validation_report(
        report_path,
        bundle=bundle,
        model=model,
        outputs=outputs,
        loss=loss,
        split_counts=split_counts,
        processed_loaded=processed_loaded,
    )
    print(f"Stage 3 forward-pass OK. loss={loss.item():.6f} source={bundle.source}")
    print(f"Validation report: {report_path}")
    print("Full training was not started.")


if __name__ == "__main__":
    main()
