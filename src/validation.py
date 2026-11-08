import pandas as pd
from sklearn.model_selection import KFold


def create_folds(df: pd.DataFrame, cfg):
    """Нарезаем фолды"""
    df = df.copy()

    cv = KFold(n_splits=cfg.split.n_splits, shuffle=True, random_state=cfg.general.seed)

    return cv
