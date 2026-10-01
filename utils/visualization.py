"""Save epoch-level training and held-out validation curves."""


def save_training_curves(history, output, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for split, label in (("train", "Train"), ("validation", "Validation")):
        axes[0].plot(epochs, [row[split]["loss"] for row in history], marker="o", label=label)
        axes[1].plot(epochs, [100 * row[split]["accuracy"] for row in history], marker="o", label=label)
    axes[0].set(ylabel="Cross-entropy loss", title="Loss")
    axes[1].set(ylabel="Accuracy (%)", ylim=(-2, 102), title="Accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        if len(epochs) <= 10:
            axis.set_xticks(epochs)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(title)
    fig.savefig(output, dpi=160)
    plt.close(fig)
