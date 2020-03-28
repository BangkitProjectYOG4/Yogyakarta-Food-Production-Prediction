import tensorflow as tf
from tensorflow import keras

from modeling import building_model


def training(model, train_data, train_labels, epochs, early_stop=False, patience=10):
    if early_stop:
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        history = model.fit(
            train_data, train_labels, epochs=epochs, validation_split=0.2, verbose=1, callbacks=[early_stop]
        )
    else:
        history = model.fit(
            train_data, train_labels, epochs=epochs, validation_split=0.2, verbose=1, callbacks=[]
        )
        

    return history
