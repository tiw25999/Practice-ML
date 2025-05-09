# -*- coding: utf-8 -*-
"""Template - ANN.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vaSTVWwG_8adtMryAzxtgZkgMPMgowKp

# Artificial Neural Network

### Importing the libraries
"""

# Import necessary libraries
import tensorflow as tf
import numpy as np
import pandas as pd

print(tf.__version__)

"""## Part 1 - Data Preprocessing

Import dataset and assign to variables.
"""

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)

"""Encoding 'Gender'"""

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

print(X)

"""Encoding 'Geography' (one hot encoding)
France -> 1 0 0
Spain -> 0 1 0
Germany -> 0 0 1
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)

"""Splitting the dataset into the training and test dataset"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

X_train.shape

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""## Part 2 - Building the ANN"""

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units = 8, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

"""## Part 3 - Training the ANN"""

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics =['accuracy'])

ann.fit(X_train, y_train, batch_size = 32, epochs = 500)

print(y_train)

"""## Part 4 - Making the predictions and evaluating the model"""