# Heart Disease Prediction with Neural Networks

import sys
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import keras
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

# import dataset

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
         'oldpeak', 'slope', 'ca', 'thal', 'class']

# read the datfile

df = pd.read_csv(url, names=names)
df = df[~df.isin(['?'])]
df = df.dropna(axis=0)
# transform to numeric
df = df.apply(pd.to_numeric)

df.hist(figsize=(12, 12))
# plt.show()

# Create X and Y datasets for training

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# convert data to categorical  labels


Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)


# define function to build keras model

def create_model():
    model = Sequential()
    model.add(Dense(8, input_dim=13, kernel_initializer='normal', activation='relu'))  # 13 attributes (14-1), 8 neurons
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))  # hidden layer
    model.add(Dense(5, activation='softmax'))  # output

    # compile model
    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


model = create_model()

# fit model to training data

model.fit(X_train, Y_train, epochs=100, batch_size=10, verbose=1)
