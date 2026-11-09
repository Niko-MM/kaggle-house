from sklearn.model_selection import KFold


def create_folds(cfg):
    """
    Нарезаем фолды. Количество и seed в конфиге
    """

    cv = KFold(n_splits=cfg.split.n_splits, shuffle=True, random_state=cfg.general.seed)

    return cv
