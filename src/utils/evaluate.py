from typing import Any
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
import torch


def evaluate_model(model: Model, test_data_hc: Sequence) -> None:
    """
    Evaluate the model on the test dataset and print performance metrics.

    Args:
        model (Model): The trained Keras model to evaluate.
        test_data_hc (Sequence): The test dataset in the form of a Keras Sequence.

    Returns:
        None: This function prints the evaluation results but does not return any value.
    """
    test_results = model.evaluate(test_data_hc)
    print("Test loss", test_results[0])
    print("Test accuracy", test_results[1])
    print("Test f1-score", test_results[2])
    print("Test recall", test_results[3])
    print("Test precision", test_results[4])


def evaluate_img_model(model: tf.keras.Model, test_dataset: tf.data.Dataset) -> None:
    """
    Evaluate the Xception model on the test dataset.

    Args:
        model (tf.keras.Model): The model to evaluate
        test_dataset (tf.data.Dataset): The test dataset
    """
    score = model.evaluate(test_dataset, verbose=0)
    print(f"Test loss: {score[0]:.4f}")
    print(f"Test accuracy: {score[1]:.4f}")
    print(f"Test Recall: {score[2]:.4f}")
    print(f"Test Precision: {score[3]:.4f}")
    f1_score = 2 * score[2] * score[3] / (score[2] + score[3])
    print(f"Test F1-score: {f1_score:.4f}")


def evaluate_vit_model(model, test_dataset):
    """
    Evaluate the ViT model on the test dataset.

    Args:
        model (nn.Module): The model to evaluate
        test_dataset (Dataset): The test dataset
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs > 0).squeeze()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
