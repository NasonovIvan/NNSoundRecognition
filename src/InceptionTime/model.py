from typing import List, Tuple
from tensorflow.keras.layers import (
    Conv1D,
    Dropout,
    MaxPooling1D,
    Flatten,
    Dense,
    Input,
    concatenate,
)
from tensorflow.keras.models import Model
import tensorflow as tf


def inception_module(x: tf.Tensor, filters: List[int]) -> tf.Tensor:
    """
    Creates an inception module for the InceptionTime network.

    Args:
        x (tf.Tensor): Input tensor.
        filters (List[int]): List of filter sizes for each branch.

    Returns:
        tf.Tensor: Concatenated output of all branches.
    """
    branches = []

    branch1 = Conv1D(filters=filters[0], kernel_size=1, activation="relu")(x)
    drop1 = Dropout(0.2)(branch1)
    branches.append(drop1)

    branch2 = Conv1D(
        filters=filters[1], kernel_size=2, padding="same", activation="relu"
    )(x)
    drop2 = Dropout(0.2)(branch2)
    branches.append(drop2)

    branch4 = Conv1D(
        filters=filters[2], kernel_size=4, padding="same", activation="relu"
    )(x)
    drop4 = Dropout(0.2)(branch4)
    branches.append(drop4)

    branches.append(x)

    concatenated = concatenate(branches)
    return concatenated


def build_inception_time(input_shape: Tuple[int, int], num_classes: int) -> Model:
    """
    Builds the InceptionTime network.

    Args:
        input_shape (Tuple[int, int]): Shape of the input data.
        num_classes (int): Number of classes for classification.

    Returns:
        Model: Compiled Keras model of InceptionTime network.
    """
    inputs = Input(shape=input_shape)

    x = Conv1D(filters=128, kernel_size=1, activation="relu")(inputs)

    x = inception_module(x, [32, 64, 128])
    x = inception_module(x, [32, 64, 128])
    x = inception_module(x, [32, 64, 128])

    x = MaxPooling1D(pool_size=3, strides=2)(x)
    x = Flatten()(x)

    x = Dense(256, activation="relu")(x)
    drop = Dropout(0.1)(x)
    # x = im.Dense(num_classes, activation='softmax')(drop)
    x = Dense(1, activation="sigmoid")(drop)

    model = Model(inputs=inputs, outputs=x)
    return model
