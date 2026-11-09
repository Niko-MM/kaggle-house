import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
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
class ModelEngine(BaseEstimator):
    """
    Интерфейс модели, адаптирован под pipeline sklearn
    """

    def __init__(self, lr, epochs, batch_size, seed) -> None:
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.criterion = nn.MSELoss()
        self.loss_history = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """Обучаем модель на переданной таблице"""

        X_scaled = self.scaler.fit_transform(X)
        # Веса сети: каждый фолд стартует с одного и того же места
        torch.manual_seed(self.seed)

        self.model = HousModel(X_scaled.shape[1])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X_tensor, y_tensor)

        # Порядок батчей
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, generator=generator
        )

        self.model.train()
        self.loss_history = []

        for epoch in range(self.epochs):
            epoch_losses = []
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                preds = self.model(batch_X)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())
            mean_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            self.loss_history.append(mean_epoch_loss)
        return self

    def predict(self, X):
        """Делаем предсказание"""
        # Масштабируем новые данные, по тренировочному сету
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        self.model.eval()
        # Отсоединяем граф, для экономии памяти
        with torch.no_grad():
            preds = self.model(X_tensor)
            preds_numpy = preds.cpu().numpy().flatten()
        return preds_numpy
