import matplotlib.pyplot as plt


def plot_loss_curve(loss_history, title="Кривая обучения (Loss Curve)"):
    """Одна линия: MSE по log цены на тех домах, по которым шагали веса"""
    plt.figure(figsize=(9, 4.5))
    plt.plot(
        range(1, len(loss_history) + 1),
        loss_history,
        color="crimson",
        linewidth=2,
    )
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Эпоха", fontsize=11)
    plt.ylabel("Loss (MSE по log цены)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
