from typing import Tuple
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
import src.utils.imports as im

im.ssl._create_default_https_context = im.ssl._create_unverified_context

def build_xception_model(
    input_shape: Tuple[int, int, int] = (255, 255, 3),
    num_classes: int = 1
) -> Model:
    """
    Builds a Xception model.

    Args:
        input_shape (Tuple[int, int, int]): Shape of the input images.
        num_classes (int): Number of output classes.

    Returns:
        Model: Compiled Keras model of Xception network.
    """
    inputs = Input(shape=input_shape)
    
    # Rescaling layer
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1)(inputs)
    
    # Base Xception model
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.1)(x)
    
    if num_classes == 1:
        outputs = Dense(1, activation='sigmoid')(x)
    else:
        outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model