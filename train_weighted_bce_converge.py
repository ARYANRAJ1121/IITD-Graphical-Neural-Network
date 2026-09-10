from pathlib import Path

from src.config import load_config
from src.data.processed_bundle import load_processed_bundle
from src.training.train_weighted_bce import train_weighted_bce


def main() -> None:
    config = load_config(Path("configs/base.yaml"))
    bundle = load_processed_bundle(
        Path(config["data"]["processed_dir"]),
        int(config["training"]["seed"]),
        tuple(config["training"]["patient_split"]),
    )
    if bundle is None:
        raise SystemExit("Processed tensors are missing.")
    report = Path("experiments/weighted_bce_convergence_report.md")
    payload = train_weighted_bce(
        config,
        bundle,
        report,
        epochs=50,
        checkpoint_name="weighted_bce_converged_best.pt",
        history_name="weighted_bce_converged_history.json",
        early_stop_patience=10,
        experiment_name="weighted_bce_convergence",
        compare_weighted_bce_20=True,
    )
    print(f"best_epoch={payload['best_epoch']} checkpoint={payload['checkpoint']}")
    print(f"early_stopping={payload['early_stopping']}")
    print(f"report={report}")


if __name__ == "__main__":
    main()
