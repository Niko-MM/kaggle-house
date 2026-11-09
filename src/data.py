import pandas as pd


def load_data(cfg):
    """Читаем train.csv и test.csv"""
    train = pd.read_csv(cfg.paths.train)
    test = pd.read_csv(cfg.paths.test)

    return train, test
