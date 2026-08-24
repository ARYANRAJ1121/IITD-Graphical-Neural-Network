from pathlib import Path

from src.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_config_loads_correctly() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    assert config["dataset"]["fhir_dir"]
    assert config["data"]["processed_dir"] == "data/processed"
    assert config["model"]["hidden_dim"] == 128
    assert config["training"]["seed"] == 42


def test_fhir_directory_exists() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    fhir_dir = Path(config["dataset"]["fhir_dir"])
    assert fhir_dir.exists()
    assert fhir_dir.is_dir()


def test_required_project_directories_exist() -> None:
    required_dirs = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "src" / "data",
        PROJECT_ROOT / "src" / "graph",
        PROJECT_ROOT / "src" / "embeddings",
        PROJECT_ROOT / "src" / "models",
        PROJECT_ROOT / "src" / "training",
        PROJECT_ROOT / "src" / "evaluation",
        PROJECT_ROOT / "configs",
        PROJECT_ROOT / "experiments",
        PROJECT_ROOT / "notebooks",
        PROJECT_ROOT / "tests",
    ]
    for directory in required_dirs:
        assert directory.exists()
        assert directory.is_dir()
