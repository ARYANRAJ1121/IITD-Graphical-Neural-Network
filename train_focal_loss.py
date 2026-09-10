from pathlib import Path

from src.config import load_config
from src.data.processed_bundle import load_processed_bundle
from src.training.train_focal_loss import train_focal_loss


def main() -> None:
    config = load_config(Path("configs/base.yaml"))
    bundle = load_processed_bundle(
        Path(config["data"]["processed_dir"]),
        int(config["training"]["seed"]),
        tuple(config["training"]["patient_split"]),
    )
    if bundle is None:
        raise SystemExit("Processed tensors are missing.")
    report = Path("experiments/focal_loss_report.md")
    payload = train_focal_loss(
        config,
        bundle,
        report,
        epochs=50,
        checkpoint_name="focal_loss_best.pt",
        history_name="focal_loss_history.json",
        early_stop_patience=10,
    )
    print(f"best_epoch={payload['best_epoch']} checkpoint={payload['checkpoint']}")
    print(f"early_stopping={payload['early_stopping']}")
    print(f"report={report}")


if __name__ == "__main__":
    main()
