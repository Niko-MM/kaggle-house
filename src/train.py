import numpy as np
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def run(df, cfg, features, cv_splitter, model):
    """Линия учится на log1p(цены).

    log1p и expm1 — пара: второе отменяет первое.
    RMSE в долларах можно сравнивать с прошлыми прогонами.
    RMSE логарифма — другая единица, к долларам её не прикладывают.
    """
    fold_dollar_rmses = []
    fold_log_rmses = []
    model_name = model.__class__.__name__

    X = df[features]
    y = df[cfg.columns.target]

    for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        y_train_log = np.log1p(y_train)
        y_val_log = np.log1p(y_val)

        fold_model = clone(model)

        fold_model.fit(X_train, y_train_log)

        log_preds = fold_model.predict(X_val)
        log_rmse = root_mean_squared_error(y_val_log, log_preds)

        # В доллары возвращаем предсказание
        dollar_preds = np.expm1(log_preds)
        dollar_rmse = root_mean_squared_error(y_val, dollar_preds)

        fold_log_rmses.append(log_rmse)
        fold_dollar_rmses.append(dollar_rmse)
        print(f"Фолд {fold}: RMSE $ = {dollar_rmse:.1f} | RMSE log = {log_rmse:.4f}")

    mean_dollar_rmse = float(np.mean(fold_dollar_rmses))
    std_dollar_rmse = float(np.std(fold_dollar_rmses))
    mean_log_rmse = float(np.mean(fold_log_rmses))
    std_log_rmse = float(np.std(fold_log_rmses))

    return (
        mean_dollar_rmse,
        std_dollar_rmse,
        mean_log_rmse,
        std_log_rmse,
        model_name,
    )
