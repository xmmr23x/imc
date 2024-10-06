#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 10:14:36 2022

@author: pedroa
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import svm

from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('../BasesDatos/csv/dataset3.csv',header=None)
X    = data.iloc[:,:-1].values
y    = data.iloc[:,-1].values

# Separar los datos
sss = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.25)

for train_index, test_index in sss.split(X,y):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

# Estandarizacion
scaler         = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train the SVM model
svm_model = svm.SVC(kernel='rbf',C=5, gamma=1)
svm_model.fit(X_train_scaled, y_train)

# Hacer predicciones con el modelo entrenado
y_pred = svm_model.predict(X_test_scaled)

# Evaluar el rendimiento
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')


# Show the points
X = X_train_scaled
y = y_train
plt.figure(1)
plt.clf()
plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)

# Show the separating hyperplane
plt.axis('tight')
# Extract the limit of the data to construct the mesh
x_min = X[:, 0].min()
x_max = X[:, 0].max()
y_min = X[:, 1].min()
y_max = X[:, 1].max()

# Create the mesh and obtain the Z value returned by the SVM
XX, YY = np.mgrid[x_min:x_max:500j, y_min:y_max:500j]
Z = svm_model.decision_function(np.c_[XX.ravel(), YY.ravel()])

# Make a color plot including the margin hyperplanes (Z=-1 and Z=1) and the
# separating hyperplane (Z=0)
Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z > 0, cmap='tab20b')
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
				levels=[-1, 0, 1])

plt.show()

