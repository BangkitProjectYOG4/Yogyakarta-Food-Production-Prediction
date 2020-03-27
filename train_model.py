import tensorflow as tf
from tensorflow import keras

from modeling import building_model


def training(model, train_data, train_labels, epochs):
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0:
                print('')
            print('.', end='')

    history = model.fit(
        train_data, train_labels, epochs=epochs, validation_split=0.2, verbose=0, callbacks=[PrintDot()]
    )

    return history
