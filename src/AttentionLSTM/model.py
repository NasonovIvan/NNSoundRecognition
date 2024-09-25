from typing import Tuple
from tensorflow.keras.layers import (
    Input,
    LSTM,
    MultiHeadAttention,
    LayerNormalization,
    TimeDistributed,
    Dense,
    Flatten,
    concatenate,
)
from tensorflow.keras.models import Model
import tensorflow as tf


def build_attention_lstm_model(
    input_shape: Tuple[int, int],
    num_recurrent_units: int = 64,
    num_attention_heads: int = 8,
) -> Model:
    """
    Builds an AttentionLSTM model.

    Args:
        input_shape (Tuple[int, int]): Shape of the input data.
        num_recurrent_units (int): Number of units in the LSTM layer.
        num_attention_heads (int): Number of attention heads.

    Returns:
        Model: Compiled Keras model of Attention-LSTM network.
    """
    inputs = Input(shape=input_shape)

    recurrent_layer = LSTM(num_recurrent_units, return_sequences=True)(inputs)
    attention_layer = MultiHeadAttention(
        num_heads=num_attention_heads, key_dim=num_recurrent_units
    )(inputs, inputs)

    attention_output = LayerNormalization()(
        concatenate([attention_layer, recurrent_layer])
    )
    
    dense_layer = TimeDistributed(Dense(512, activation='relu'))(attention_output)
    dense_layer = TimeDistributed(Dense(64, activation='relu'))(dense_layer)
    dense_layer = Flatten()(dense_layer)
    outputs = Dense(1, activation='sigmoid')(dense_layer)

    model = Model(inputs=inputs, outputs=outputs)
    return model
