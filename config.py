from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).parent

config = {
    "general": {
        "experiment_name": "baseline",
        "seed": 42,
    },
    "paths": {
        "train": str(ROOT / "data" / "train.csv"),
        "test": str(ROOT / "data" / "test.csv"),
    },
    "columns": {
        "target": "SalePrice",
        "id": "Id",
    },
    "features": {
        "drop": ["PassengerId", "Ticket", "Cabin", "Name"],
        "categorical": ["Sex", "Embarked", "Title"],
    },
    "split": {"n_splits": 5},
    "mlp": {
        "lr": 0.0005,
        "epochs": 400,
        "batch_size": 32,
        "weight_decay": 0.01,
    },
}

config = OmegaConf.create(config)
