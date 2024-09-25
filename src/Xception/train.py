import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.path_names import WEIGHTS_PATH, TRAIN_HISTORY

import datetime
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import custom_object_scope
from typing import Tuple
import src.utils.functions as fn
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from model import build_xception_model

def train_model(
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    train_size: int,
    batch_size: int
) -> Tuple[Model, keras.callbacks.History]:
    """
    Trains the Xception model on the given data.

    Args:
        train_dataset (tf.data.Dataset): Training dataset.
        val_dataset (tf.data.Dataset): Validation dataset.
        train_size (int): Size of the training dataset.
        batch_size (int): Batch size for training.

    Returns:
        Tuple[Model, keras.callbacks.History]: Trained model and training history.
    """
    # Build and compile the model
    model = build_xception_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    )

    # Callbacks
    weights_file = WEIGHTS_PATH + 'Xception_weights.h5'
    logdir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        ModelCheckpoint(weights_file, save_best_only=True, monitor="val_loss"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=3, mode='min', min_lr=0),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        TensorBoard(log_dir=logdir)
    ]

    # Training
    epochs = 20
    steps_per_epoch = train_size // batch_size
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks
    )

    # Fine-tuning
    base_model = model.layers[2]
    base_model.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    )

    weights_file_finetune = WEIGHTS_PATH + 'Xception_weights_finetune.h5'
    callbacks_finetune = [
        ModelCheckpoint(weights_file_finetune, mode='max', save_best_only=True, monitor='val_accuracy'),
        ReduceLROnPlateau(monitor="val_loss", factor=0.75, patience=3, mode='min', min_lr=0)
    ]

    epochs_finetune = 8
    history_finetune = model.fit(
        train_dataset,
        epochs=epochs_finetune,
        validation_data=val_dataset,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks_finetune
    )

    # Combine histories
    for k in history.history.keys():
        history.history[k].extend(history_finetune.history[k])

    with open(TRAIN_HISTORY + "HistoryXceptionDict", "wb") as file_pi:
        pickle.dump(history.history, file_pi)

    model.trainable = False

    return model, history

def define_model() -> Model:
    """
    Defines and loads a pre-trained Xception model.

    Returns:
        Model: Loaded pre-trained model.
    """
    model = build_xception_model()
    model.load_weights(WEIGHTS_PATH + 'Xception_weights_finetune.h5')
    return model