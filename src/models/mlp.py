import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class HousModel(nn.Module):
    """Архитектура модели"""

    def __init__(self, input_dim) -> None:
        super().__init__()
        self.line1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.line2 = nn.Linear(64, 1)
        self.drop1 = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.line1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.line2(x)
        return x


# Наследуемся, чтобы работал run
class ModelEngine(BaseEstimator, RegressorMixin):
    """
    Интерфейс модели, адаптирован под pipline sklearn
    """

    def __init__(self, lr, epochs, batch_size, seed) -> None:
        # Параметры модели
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        # Ошибка
        self.criterion = nn.MSELoss()
        # История учебной ошибки по эпохам
        self.loss_history = None
        # Нормализация данных
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """Обучаем модель на переданной таблице."""
        # Нормализуем данные
        X_scaled = self.scaler.fit_transform(X)
        torch.manual_seed(self.seed)
        # Создаем модель и оптимизатор
        self.model = HousModel(X_scaled.shape[1])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        # Переводим в тензор
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

        # Упаковываем данные
        dataset = TensorDataset(X_tensor, y_tensor)
        # Отдельный кубик порядка батчей, тот же seed. Иначе shuffle зависит от того,
        # сколько случайных чисел съела инициализация весов.
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, generator=generator
        )

        # Переводим модель в режим обучения
        self.model.train()
        # Список для хранения лоса по эпохам
        self.loss_history = []

        for epoch in range(self.epochs):
            # лосы батча внутри эпохи
            epoch_losses = []
            for batch_X, batch_y in dataloader:
                # Обнуляем градиенты
                self.optimizer.zero_grad()
                # Прямой проход
                preds = self.model(batch_X)
                # Считаем ошибку
                loss = self.criterion(preds, batch_y)
                # Делаем обратный проход
                loss.backward()
                # Настраиваем веса
                self.optimizer.step()
                # Чистый лосс батча
                epoch_losses.append(loss.item())
            # Средний лосс за всю эпоху. Добавляем в общую
            mean_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            self.loss_history.append(mean_epoch_loss)
        return self

    def predict(self, X):
        """Делаем предсказание"""
        # Используем трансформ, чтобы не переобучать
        X_scaled = self.scaler.transform(X)
        # Переводим в тензор
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Переводим в режим предсказания
        self.model.eval()

        # Отсоединяем граф
        with torch.no_grad():
            # Делаем предсказание
            preds = self.model(X_tensor)
            # Переводим на cpu (Если модель на cuda/mps), переводим в numpy, растягиваем в вектор
            preds_numpy = preds.cpu().numpy().flatten()
        return preds_numpy
