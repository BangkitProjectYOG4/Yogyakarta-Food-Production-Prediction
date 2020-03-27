import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# import matplotlib.pyplot as plt

from preprocessing import preprocess
from modeling import building_model
from train_model import training
from ploting import ploting_history
from evaluation_model import evaluation


# preprocess('data/padi.csv', 1)
preprocessed_dataset = pd.read_csv(
    "data/preprocessed.csv", names=['Year-3', 'Year-2', 'Year-1', 'Target'], sep=",")

train_data = preprocessed_dataset.sample(frac=0.9, random_state=0)
test_data = preprocessed_dataset.drop(train_data.index)

train_labels = train_data.pop('Target')
test_labels = test_data.pop('Target')

model = building_model(train_data)
trained_model = training(model, train_data, train_labels, epochs=1000)
ploting_history(trained_model)

print('')
evaluation(model, test_data, test_labels)

predict_result = model.predict(test_data).flatten()
print(predict_result)
