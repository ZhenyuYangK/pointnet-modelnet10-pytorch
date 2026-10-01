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


def save_confusion_matrix(matrix, classes, output, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(10, 9), layout="constrained")
    picture = ax.imshow(matrix, cmap="Blues", vmin=0)
    fig.colorbar(picture, ax=ax, shrink=0.8, label="Number of models")
    ax.set_xticks(range(len(classes)), labels=classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), labels=classes)
    ax.set(xlabel="Predicted class", ylabel="True class", title=title)
    for row in range(len(classes)):
        for column in range(len(classes)):
            value = matrix[row, column]
            ax.text(column, row, str(value), ha="center", va="center",
                    color="white" if value > matrix.max() / 2 else "black", fontsize=10)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_prediction_examples(examples, output, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 8), layout="constrained")
    for panel, example in enumerate(examples, start=1):
        points = example["points"].numpy()
        ax = fig.add_subplot(2, 3, panel, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="viridis", s=3)
        ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=-55)
        ax.set_axis_off()
        correct = example["true_label"] == example["predicted_label"]
        ax.set_title(f"{example['name']} | {'correct' if correct else 'incorrect'}\n"
                     f"True: {example['true_class']}\nPredicted: {example['predicted_class']}",
                     color="darkgreen" if correct else "darkred", fontsize=10)
    fig.suptitle(title, fontsize=12)
    fig.savefig(output, dpi=160)
    plt.close(fig)
