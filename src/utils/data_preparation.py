import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import src.utils.functions as fn
from typing import Tuple, Callable

from torch.utils.data import Dataset
from torch.utils.data import random_split
from PIL import Image
import os

from src.utils.path_names import X_DATA_HC_PATH, Y_DATA_HC_PATH
from src.utils.path_names import TRAIN_CSV_PATH, SPECTROGRAMS_FOLDER


def load_data(
    train_csv_path: str, train_wav_path: str, train_wav_noised_path: str
) -> Tuple[pd.DataFrame, str, str]:
    """
    Loads the data from CSV file and returns paths to wav files.

    Args:
        train_csv_path (str): Path to the training CSV file.
        train_wav_path (str): Path to the directory containing training wav files.
        train_wav_noised_path (str): Path to the directory containing noised training wav files.

    Returns:
        Tuple[pd.DataFrame, str, str]: DataFrame with loaded data, path to train wav files, path to noised train wav files.
    """
    df = pd.read_csv(train_csv_path)
    return df, train_wav_path, train_wav_noised_path


def prepare_datasets(
    df: pd.DataFrame,
    path_train: str,
    augment_data_fn: Callable,
    augmentation_factor: int = 12000,
    create_dataset: bool = False,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Prepares training, validation, and test datasets.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        path_train (str): Path to the training wav files.
        augment_data_fn (Callable): Function to augment the data.
        augmentation_factor (int, optional): Factor for data augmentation. Defaults to 12000.
        create_dataset (bool, optional): Whether to create a new dataset or load existing. Defaults to False.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: Training, validation, and test datasets.
    """
    if create_dataset:
        x_data_hc, y_data_hc = fn.create_HC_dataset_wavs(df, path_train, Noised=False)
    else:
        x_data_hc = np.load(X_DATA_HC_PATH)
        y_data_hc = np.load(Y_DATA_HC_PATH)

    x_data_hc_augmented, y_data_hc_augmented = augment_data_fn(
        x_data_hc, y_data_hc, augmentation_factor
    )

    x_train_val, x_test_hc, y_train_val, y_test_hc = train_test_split(
        x_data_hc_augmented,
        y_data_hc_augmented,
        test_size=0.1,
        random_state=42,
        shuffle=True,
        stratify=y_data_hc_augmented,
    )

    x_train_hc, x_val_hc, y_train_hc, y_val_hc = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.1,
        random_state=42,
        shuffle=True,
        stratify=y_train_val,
    )

    train_data_hc = tf.data.Dataset.from_tensor_slices((x_train_hc, y_train_hc)).batch(
        64
    )
    val_data_hc = tf.data.Dataset.from_tensor_slices((x_val_hc, y_val_hc)).batch(64)
    test_data_hc = tf.data.Dataset.from_tensor_slices((x_test_hc, y_test_hc)).batch(64)

    return train_data_hc, val_data_hc, test_data_hc

def prepare_img_dataset():
    # Create dataset with augmentation images. It takes about 2-3 minutes
    combined_dataset = fn.CreateKaggleDatasetNew(csv_file=TRAIN_CSV_PATH, image_folder=SPECTROGRAMS_FOLDER)

    combined_dataset = combined_dataset.shuffle(1024*2, seed=42)

    # Define the sizes for train, validation, and test sets
    train_size = int(0.9 * len(combined_dataset))  # 90% for training
    val_size = int(0.05 * len(combined_dataset))   # 5% for validation
    test_size = len(combined_dataset) - train_size - val_size  # Remaining for testing

    # Shuffle the dataset before splitting
    combined_dataset = combined_dataset.shuffle(len(combined_dataset))

    # Split the dataset
    train_dataset = combined_dataset.take(train_size)
    temp_dataset = combined_dataset.skip(train_size)
    val_dataset = temp_dataset.take(val_size)
    test_dataset = temp_dataset.skip(val_size)

    # Define batch size
    batch_size = 64

    # Batch and preprocess the datasets
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.repeat()  # The training dataset must repeat for several epochs
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)  # Prefetch next batch while training

    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset


def prepare_vit_dataset(dataset, train_ratio=0.7, val_ratio=0.15):
    """
    Prepare and return train, validation and test datasets.
    
    Args:
        dataset: The full dataset
        train_ratio (float): Ratio of the training set (default: 0.7)
        val_ratio (float): Ratio of the validation set (default: 0.15)
    
    Returns:
        tuple: Train, validation and test datasets
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    return train_dataset, val_dataset, test_dataset

class CustomDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_name = os.path.splitext(img_name)[0] + '.png'
        img_path = os.path.join(self.image_dir, img_name)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

class ArticleDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label