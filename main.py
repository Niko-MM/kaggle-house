from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import config
from src.data import load_data
from src.features import build_features
from src.preprocessing import encode_categorical_features, fill_train_and_test
from src.train import run
from src.utils import log_experiment
from src.validation import create_folds


def run_cv_txt(df_train, features, cfg, model):
    """Смотрим результаты модели по фолдам
    Возарщаем среднюю точность и среднее отклонение
    """
    # Делим датасет на фолды
    cv_splitter = create_folds(df_train, cfg)
    # Запускаем тренировку моделей
    mean_dollar_rmse, std_dollar_rmse, mean_log_rmse, std_log_rmse, model_name = run(
        df_train, cfg, features, cv_splitter, model=model
    )
    # В журнал кладём доллары: их можно ставить рядом с прошлыми прогонами.
    # RMSE логарифма печатается в консоли из run(), к долларам его не прикладывают.
    log_experiment(
        experiment_name="mlp_2line_drop1",
        model_name=model_name,
        features=features,
        mean_rmse=mean_dollar_rmse,
        std_rmse=std_dollar_rmse,
        seed=cfg.general.seed,
    )
    print(f"RMSE log: среднее {mean_log_rmse:.4f} | разброс {std_log_rmse:.4f}")
    return mean_dollar_rmse, std_dollar_rmse


def prepare(train, test, cfg):

    test_ids = test[cfg.columns.id].copy()
    # Создаем фичи
    df_train, df_test, features = build_features(train, test, cfg)
    # Заполняем пропуски
    df_train, df_test = fill_train_and_test(df_train, df_test)
    # Фичи кроме таргета
    features = [col for col in df_train.columns if col != cfg.columns.target]
    # Подгоняем колонки теста под features
    df_test = df_test.reindex(columns=features)
    # Кодируем категориальные фичи
    df_train, df_test, features = encode_categorical_features(df_train, df_test, cfg)

    return df_train, features, test_ids, df_test


def write_submission(df_train, df_test, test_ids, features, cfg, model):
    """Финальное обучение"""
    # Тренировочный сет
    X_train = df_train[features]
    y_train_log = np.log1p(df_train[cfg.columns.target])
    # Тестовый сет
    X_test = df_test[features]
    # Обучаем на тренировочном сете
    model.fit(X_train, y_train_log)
    # Делаем предсказание
    predictions_log = model.predict(X_test)
    # Возвращаем цену в $
    predictions_dollars = np.expm1(predictions_log)
    # Собираем финальный сет
    submission = pd.DataFrame(
        {cfg.columns.id: test_ids.values, cfg.columns.target: predictions_dollars}
    )
    # Путь до файла
    output_path = Path(__file__).parent / "outputs" / "submission.csv"
    # Гарантируем, что файл существует
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Записываем результаты
    submission.to_csv(output_path, index=False)
    print(f"Файл ответа: {output_path}, строк: {len(submission)}")

    return output_path


if __name__ == "__main__":
    train_raw, test_raw = load_data(config)
    df_train, features, test_ids, df_test = prepare(train_raw, test_raw, config)
    model = LinearRegression()

    # Прогоняем модель, пишем логи csv, txt
    # run_cv_txt(df_train, features, config, model)

    # Пишем submission.csv. На фолдах учились копии, эта модель ещё пустая
    write_submission(df_train, df_test, test_ids, features, config, model)
