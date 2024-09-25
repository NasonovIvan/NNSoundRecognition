import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.path_names import WEIGHTS_PATH, TRAIN_HISTORY

import datetime
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import custom_object_scope
from typing import Tuple, Any
from model import build_attention_lstm_model
from src.utils.utils import exponential_lr
import src.utils.functions as fn


def train_model(
    train_data_hc: tf.data.Dataset, val_data_hc: tf.data.Dataset
) -> Tuple[keras.Model, keras.callbacks.History]:
    """
    Trains the AttentionLSTM model on the given data.

    Args:
        train_data_hc (tf.data.Dataset): Training dataset.
        val_data_hc (tf.data.Dataset): Validation dataset.

    Returns:
        Tuple[keras.Model, keras.callbacks.History]: Trained model and training history.
    """
    # Define input shape and number of classes
    input_shape = train_data_hc.element_spec[0].shape[1:]

    # Build and compile the model
    model = build_attention_lstm_model(input_shape)
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["accuracy", fn.f1_score, fn.recall, fn.precision],
    )

    weights_inception = WEIGHTS_PATH + 'AttentionLstm.h5'
    logdir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            weights_inception, save_best_only=True, monitor="val_loss"
        ),
        tf.keras.callbacks.LearningRateScheduler(exponential_lr, verbose=True),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.001, patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(log_dir=logdir),
    ]

    history = model.fit(
        train_data_hc,
        epochs=15,
        callbacks=callbacks,
        validation_data=val_data_hc,
        verbose=1,
    )

    with open(TRAIN_HISTORY + "HistoryAttentionDict", "wb") as file_pi:
        pickle.dump(history.history, file_pi)

    return model, history


def define_model(train_data_hc: tf.data.Dataset) -> keras.Model:
    """
    Defines and loads a pre-trained AttentionLSTM model.

    Args:
        train_data_hc (tf.data.Dataset): Training dataset (used to determine input shape).

    Returns:
        keras.Model: Loaded pre-trained model.
    """

    custom_objects = {
        "f1_score": fn.f1_score,
        "recall": fn.recall,
        "precision": fn.precision,
    }

    input_shape = train_data_hc.element_spec[0].shape[1:]

    model = build_attention_lstm_model(input_shape)
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["accuracy", fn.f1_score, fn.recall, fn.precision],
    )
    with custom_object_scope(custom_objects):
        model = keras.models.load_model(WEIGHTS_PATH + 'AttentionLstm.h5')

    return model
