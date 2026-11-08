import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def fill_train_and_test(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """Чистим, мапим, логарифмируем"""

    # У одного дома пропуск, заполняем его самой частой
    if 948 in df_train.index:
        df_train.loc[948, "BsmtExposure"] = "No"

    # Убираем аномальные дома
    # при высокой площади, стоимость низкая
    df_train = df_train[df_train["GrLivArea"] < 4600]

    # создаем мапу
    quality_mapping = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
    bsmt_exposure_mapping = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1}

    # Заполним нулями тестовые пропуски (отсутствие подвалов)
    # чтобы не заполнить медианой на селдующих шагах
    df_test["TotalBsmtSF"] = df_test["TotalBsmtSF"].fillna(0)
    df_test["BsmtFullBath"] = df_test["BsmtFullBath"].fillna(0)

    # Заполняем пропуски машиномест в гараже медианой по типу гаража
    cars_median = df_train.groupby("GarageType")["GarageCars"].median()
    df_test["GarageCars"] = (
        df_test["GarageCars"].fillna(df_test["GarageType"].map(cars_median)).fillna(0)
    )

    # Заполняем пропуски площади гаража медианой по типу гаража
    area_median = df_train.groupby("GarageType")["GarageArea"].median()
    df_test["GarageArea"] = (
        df_test["GarageArea"].fillna(df_test["GarageType"].map(area_median)).fillna(0)
    )
    # Удаляем не нужные фичи
    df_train = df_train.drop(columns=["GarageType"])
    df_test = df_test.drop(columns=["GarageType"])

    # логарифмируем больную цирфу
    df_train["LotArea"] = np.log1p(df_train["LotArea"])
    df_test["LotArea"] = np.log1p(df_test["LotArea"])

    # Качество состояния отопления из порядкового в числовое
    df_train["HeatingQC"] = df_train["HeatingQC"].map(quality_mapping).fillna(0)
    df_test["HeatingQC"] = df_test["HeatingQC"].map(quality_mapping).fillna(0)

    # Качество кухни из пордякового в цифру, пропуск медианой
    df_train["KitchenQual"] = df_train["KitchenQual"].map(quality_mapping).fillna(0)
    qual_medians = df_train.groupby("OverallQual")["KitchenQual"].median()
    df_test["KitchenQual"] = df_test["KitchenQual"].map(quality_mapping)
    df_test["KitchenQual"] = (
        df_test["KitchenQual"]
        .fillna(df_test["OverallQual"].map(qual_medians))
        .fillna(0)
    )

    # Качество гаража. Сначала цифры, потом медиана домов, где гараж есть.
    df_train["GarageQual"] = df_train["GarageQual"].map(quality_mapping).fillna(0)
    garage_qual_medians = (
        df_train.loc[df_train["GarageCars"] > 0]
        .groupby("GarageCars")["GarageQual"]
        .median()
    )
    # Из порядкового в числовой
    df_test["GarageQual"] = df_test["GarageQual"].map(quality_mapping)
    # Заполняем пропуски
    df_test["GarageQual"] = df_test["GarageQual"].fillna(
        df_test["GarageCars"].map(garage_qual_medians)
    )
    # Присваиваем оценкам гаража 0, у которых машинамест 0.
    df_test.loc[df_test["GarageCars"] == 0, "GarageQual"] = 0
    df_test["GarageQual"] = df_test["GarageQual"].fillna(0)

    # Состояние гаража. Тот же порядок: цифры, потом медиана живых гаражей.
    df_train["GarageCond"] = df_train["GarageCond"].map(quality_mapping).fillna(0)
    garage_cond_medians = (
        df_train.loc[df_train["GarageCars"] > 0]
        .groupby("GarageCars")["GarageCond"]
        .median()
    )
    df_test["GarageCond"] = df_test["GarageCond"].map(quality_mapping)
    df_test["GarageCond"] = df_test["GarageCond"].fillna(
        df_test["GarageCars"].map(garage_cond_medians)
    )
    df_test.loc[df_test["GarageCars"] == 0, "GarageCond"] = 0
    df_test["GarageCond"] = df_test["GarageCond"].fillna(0)

    # Состояние подвала из порядквоого в цифру на трейне
    df_train["BsmtCond"] = df_train["BsmtCond"].map(quality_mapping).fillna(0)
    # Считаем медианы домов по трейну
    bsmt_cond_median = (
        df_train.loc[df_train["TotalBsmtSF"] > 0]
        .groupby("OverallQual")["BsmtCond"]
        .median()
    )
    # Тест переводим в числа
    df_test["BsmtCond"] = df_test["BsmtCond"].map(quality_mapping)
    # Заполняем пустые тренировочной медианой
    df_test["BsmtCond"] = df_test["BsmtCond"].fillna(
        df_test["OverallQual"].map(bsmt_cond_median)
    )
    # Если площадь подвала 0, то его нигде нет
    df_test.loc[df_test["TotalBsmtSF"] == 0, "BsmtCond"] = 0
    df_test["BsmtCond"] = df_test["BsmtCond"].fillna(0)

    # Качество подвала. Тот же порядок, что у состояния подвала.
    df_train["BsmtQual"] = df_train["BsmtQual"].map(quality_mapping).fillna(0)
    bsmt_qual_median = (
        df_train.loc[df_train["TotalBsmtSF"] > 0]
        .groupby("OverallQual")["BsmtQual"]
        .median()
    )
    df_test["BsmtQual"] = df_test["BsmtQual"].map(quality_mapping)
    df_test["BsmtQual"] = df_test["BsmtQual"].fillna(
        df_test["OverallQual"].map(bsmt_qual_median)
    )
    df_test.loc[df_test["TotalBsmtSF"] == 0, "BsmtQual"] = 0
    df_test["BsmtQual"] = df_test["BsmtQual"].fillna(0)

    # Выход на улицу или стена из подвала, заполняем по аналогии
    df_train["BsmtExposure"] = (
        df_train["BsmtExposure"].map(bsmt_exposure_mapping).fillna(0)
    )
    bsmt_expo_median = (
        df_train.loc[df_train["TotalBsmtSF"] > 0]
        .groupby("OverallQual")["BsmtExposure"]
        .median()
    )
    df_test["BsmtExposure"] = df_test["BsmtExposure"].map(bsmt_exposure_mapping)
    df_test["BsmtExposure"] = df_test["BsmtExposure"].fillna(
        df_test["OverallQual"].map(bsmt_expo_median)
    )
    df_test.loc[df_test["TotalBsmtSF"] == 0, "BsmtExposure"] = 0
    df_test["BsmtExposure"] = df_test["BsmtExposure"].fillna(0)

    return df_train, df_test


def encode_categorical_features(df_train: pd.DataFrame, df_test: pd.DataFrame, cfg):
    """Кодируем категориальные признаки через One-Hot Encoding"""

    cat_cols = ["MSSubClass", "Neighborhood"]
    # Создаем кодировщик
    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    # Обучаем кодировщик на тренировочных данных
    encoder.fit(df_train[cat_cols].astype(str))
    # Трансформируем в бинарные числа
    train_ohe_raw = encoder.transform(df_train[cat_cols].astype(str))
    test_ohe_raw = encoder.transform(df_test[cat_cols].astype(str))
    # Достаем имена
    ohe_col_names = encoder.get_feature_names_out(cat_cols)
    # Переводим в pd
    df_train_ohe = pd.DataFrame(
        np.asarray(train_ohe_raw), columns=ohe_col_names, index=df_train.index
    )
    df_test_ohe = pd.DataFrame(
        np.asarray(test_ohe_raw), columns=ohe_col_names, index=df_test.index
    )
    # Удаляем старую колонку
    df_train_clean = df_train.drop(columns=cat_cols)
    df_test_clean = df_test.drop(columns=cat_cols)
    # Склеиваем
    df_train_final = pd.concat([df_train_clean, df_train_ohe], axis=1)
    df_test_final = pd.concat([df_test_clean, df_test_ohe], axis=1)
    # Получаем финальные фичи без целевой
    final_features = [
        col for col in df_train_final.columns if col != cfg.columns.target
    ]

    return df_train_final, df_test_final, final_features
