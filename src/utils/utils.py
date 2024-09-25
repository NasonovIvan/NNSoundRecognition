import matplotlib.pyplot as plt
from src.utils.path_names import TRAIN_HISTORY
import pickle


def exponential_lr(
    epoch: int,
    start_lr: float = 0.001,
    min_lr: float = 0.00001,
    max_lr: float = 0.005,
    rampup_epochs: int = 5,
    sustain_epochs: int = 0,
    exp_decay: float = 0.85,
) -> float:
    """
    Calculates the learning rate for each epoch using an exponential decay schedule.

    Args:
        epoch (int): Current epoch number.
        start_lr (float, optional): Starting learning rate. Defaults to 0.001.
        min_lr (float, optional): Minimum learning rate. Defaults to 0.00001.
        max_lr (float, optional): Maximum learning rate. Defaults to 0.005.
        rampup_epochs (int, optional): Number of epochs for linear ramp-up. Defaults to 5.
        sustain_epochs (int, optional): Number of epochs to sustain max learning rate. Defaults to 0.
        exp_decay (float, optional): Exponential decay rate. Defaults to 0.85.

    Returns:
        float: Calculated learning rate for the given epoch.
    """
    if epoch < rampup_epochs:
        lr = (max_lr - start_lr) / rampup_epochs * epoch + start_lr
    elif epoch < rampup_epochs + sustain_epochs:
        lr = max_lr
    else:
        lr = (max_lr - min_lr) * exp_decay ** (
            epoch - rampup_epochs - sustain_epochs
        ) + min_lr
    return lr


def plot_metrics() -> None:
    """
    Plots training and validation metrics (loss, accuracy, and learning rate) from saved history.
    """
    with open(TRAIN_HISTORY + "HistoryInceptionDict", "rb") as file_pi:
        history_dict = pickle.load(file_pi)

    epochs = range(1, len(history_dict["accuracy"]) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history_dict["loss"], "bo", label="Training loss")
    plt.plot(epochs, history_dict["val_loss"], "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history_dict["accuracy"], "bo", label="Training accuracy")
    plt.plot(epochs, history_dict["val_accuracy"], "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history_dict["lr"], "b", label="Learning Rate")
    plt.title("Learning Rate")
    plt.legend()

    plt.tight_layout()
    plt.show()
