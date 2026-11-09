import matplotlib.pyplot as plt


def plot_loss_curve(loss_history, title="Кривая обучения"):
    """График ошибки сети по эпохам"""
    plt.figure(figsize=(9, 4.5))
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.title(title)
    plt.xlabel("Эпоха")
    plt.ylabel("Ошибка (MSE по log цены)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
