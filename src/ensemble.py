from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression


def ensemble(cfg):
    """Линия и два бустинга. Голос у каждого равный, предсказание — среднее."""
    linal = LinearRegression()

    catboost = CatBoostRegressor(
        iterations=400,
        depth=7,
        random_seed=cfg.general.seed,
        verbose=0,
        allow_writing_files=False,
    )

    light = LGBMRegressor(
        n_estimators=400,
        max_depth=7,
        random_state=cfg.general.seed,
        verbose=-1,
    )

    committee = VotingRegressor(
        estimators=[
            ("linal", linal),
            ("catboost", catboost),
            ("light", light),
        ]  # type: ignore
    )
    return committee
