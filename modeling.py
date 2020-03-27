import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def building_model(train_data):
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu,
                     input_shape=[len(train_data.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.optimizers.RMSprop(0.001)

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    return model
