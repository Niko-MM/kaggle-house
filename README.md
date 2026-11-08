# House Prices

Задача: предсказать цену дома. На Kaggle считают RMSE по логарифму цены.

На тест уехала линейная регрессия. Деревья, бустинги и сеть на тех же колонках были хуже: смотрел среднее и разброс по фолдам, в долларах.
На Kaggle получилось 0.13470.
Все прогоны лежат в `outputs/results.txt`.

## Запуск

Скачать `train.csv` и `test.csv` в папку `data/` со страницы соревнования:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Команда из корня проекта. Она учит линейку на всех очищенных домах и пишет `outputs/submission.csv`.

Ноутбук `house_prices.ipynb` — разведка и кривая потерь сети.
