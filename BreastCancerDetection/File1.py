import sys
import numpy as np
import matplotlib
import pandas as pd
import sklearn
from sklearn.model_selection import cross_validate
from sklearn import preprocessing, model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

names = ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion',
         'single_epithelial_size',
         'bear_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']

df = pd.read_csv(url, names=names)
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

# split into x and y for training
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])  # malignant or benign classification

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

seed = 8
scoring = 'accuracy'

# Define the models

models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))
models.append(('SVM', SVC()))

# model evalution
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    #print(name, cv_results.mean(), cv_results.std())

# make prediction on validation dataset

for name, model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(name)
    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test,predictions))
