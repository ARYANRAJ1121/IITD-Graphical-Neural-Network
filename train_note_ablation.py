from pathlib import Path

from src.config import load_config
from src.data.processed_bundle import load_processed_bundle
from src.training.train_note_ablation import train_note_ablation


def main() -> None:
    config = load_config(Path("configs/base.yaml"))
    bundle = load_processed_bundle(
        Path(config["data"]["processed_dir"]),
        int(config["training"]["seed"]),
        tuple(config["training"]["patient_split"]),
    )
    if bundle is None:
        raise SystemExit("Processed tensors are missing.")
    report = Path("experiments/note_ablation_report.md")
    payload = train_note_ablation(config, bundle, report)
    print(f"best_epoch={payload['best_epoch']} checkpoint={payload['checkpoint']}")
    print(f"early_stopping={payload['early_stopping']}")
    print(f"report={report}")


if __name__ == "__main__":
    main()
