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
path_to_file = 'data/padi.csv'
province = 'DI YOGYAKARTA'
sliding_window = 3
train_test_proportion = 0.8

data = pd.read_csv(path_to_file)
preprocessed = preprocess(data[data['Provinsi'] == province],\
                            sliding_window)

train_data = preprocessed[:int(train_test_proportion*len(preprocessed))]
test_data = preprocessed.drop(train_data.index)

train_labels = train_data.pop(train_data.columns[-1])
test_labels = test_data.pop(test_data.columns[-1])

model = building_model(train_data)
tf.random.set_seed(28)
trained_model = training(model, train_data, train_labels, epochs=1000, early_stop=False)
ploting_history(trained_model)

print('')
evaluation(model, test_data, test_labels)

predict_result = model.predict(test_data).flatten()
print(predict_result)
