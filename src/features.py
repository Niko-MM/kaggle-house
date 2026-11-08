import pandas as pd


def build_features(df_train: pd.DataFrame, df_test: pd.DataFrame, cfg):
    """Создаем фичи"""
    features = [
        "GrLivArea",
        "OverallQual",
        "OverallCond",
        "GarageCond",
        "GarageQual",
        "GarageCars",
        "GarageArea",
        "TotalBsmtSF",
        "1stFlrSF",
        "FullBath",
        "YearBuilt",
        "YearRemodAdd",
        "BsmtCond",
        "BsmtQual",
        "BsmtExposure",
        "Fireplaces",
        "HalfBath",
        "WoodDeckSF",
        "OpenPorchSF",
        "HeatingQC",
        "KitchenQual",
        "LotArea",
        "BsmtFullBath",
        "MSSubClass",
        "Neighborhood",
    ]
    df_train = df_train[features + [cfg.columns.target] + ["GarageType"]].copy()
    df_test = df_test[features + ["GarageType"]].copy()

    return df_train, df_test, features
