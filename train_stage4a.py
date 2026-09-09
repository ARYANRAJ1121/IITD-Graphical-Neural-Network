from pathlib import Path

from src.config import load_config
from src.data.stage3_bundle import load_processed_bundle
from src.training.train_stage4a import train_stage4a


def main() -> None:
    config = load_config(Path("configs/base.yaml"))
    bundle = load_processed_bundle(
        Path(config["data"]["processed_dir"]),
        int(config["training"]["seed"]),
        tuple(config["training"]["patient_split"]),
    )
    if bundle is None:
        raise SystemExit("Stage 2 processed tensors are missing.")
    report = Path("experiments/stage4_weighted_bce_report.md")
    payload = train_stage4a(config, bundle, report)
    print(f"best_epoch={payload['best_epoch']} checkpoint={payload['checkpoint']}")
    print(f"report={report}")


if __name__ == "__main__":
    main()
