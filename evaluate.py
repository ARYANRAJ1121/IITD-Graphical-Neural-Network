from pathlib import Path

from src.config import load_config


def main() -> None:
    config_path = Path("configs/base.yaml")
    config = load_config(config_path)
    print("Evaluation metrics are not run until training is authorized.")
    print(f"Processed directory: {config['data']['processed_dir']}")


if __name__ == "__main__":
    main()
