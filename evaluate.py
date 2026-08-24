from pathlib import Path

from src.config import load_config


def main() -> None:
    config_path = Path("configs/base.yaml")
    config = load_config(config_path)
    print(f"Stage 0 setup complete. Evaluation is not implemented yet. Loaded config from {config_path}.")
    print(f"Configured processed directory: {config['data']['processed_dir']}")


if __name__ == "__main__":
    main()

