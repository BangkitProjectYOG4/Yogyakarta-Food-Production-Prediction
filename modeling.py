import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def building_model(train_data):
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu,
                     input_shape=[train_data.shape[1]]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.optimizers.Adam(learning_rate=0.001)

    model.compile(loss='mae', optimizer=optimizer, metrics=['mse'])

    return model
