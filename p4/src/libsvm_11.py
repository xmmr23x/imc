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
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('../BasesDatos/csv/dataset3.csv',header=None)
X    = data.iloc[:,:-1].values
y    = data.iloc[:,-1].values

# Valores de hiperparámetros a probar manualmente
Cs = np.logspace(-5, 15, num=11, base=2)
Gs = np.logspace(-15, 3, num=9, base=2)

# Validacion cruzada anidada tipo K-fold
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25)

# Separar los datos
for train_index, test_index in sss.split(X,y):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

# Estandarizacion
scaler         = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Realizamos una particion tipo K-fold de los datos de entrenamiento
skf = StratifiedKFold(n_splits=5, shuffle=True)

# Búsqueda de hiperparámetros

best_acc    = 0
best_params = {'C': None, 'gamma': None}

# Para cada combinacion de parametros, realizamos K entrenamientos
for C in Cs:
	for gamma in Gs:
		scores    = []
		svm_model = svm.SVC(kernel='rbf', C=C, gamma=gamma)

		# Para cada entrenamiento k
		for ktrain_index, ktest_index in skf.split(X_train_scaled,y_train):
			# Utilizamos el subconjunto k como conjunto de test
			# y el resto de subconjuntos como conjunto de entrenamiento
			X_train_k, X_test_k = X_train_scaled[ktrain_index], X_train_scaled[ktest_index]
			y_train_k, y_test_k = y_train[ktrain_index], y_train[ktest_index]

			# se entrena el modelo con el resto de subconjuntos
			svm_model.fit(X_train_k, y_train_k)
			y_pred_k = svm_model.predict(X_test_k)

			# se almacenan los errores en un array de numpy
			# para despues calcular la media
			scores.append(accuracy_score(y_test_k, y_pred_k))

		if np.mean(scores) > best_acc:
			best_acc = np.mean(scores)
			best_params = {'C': C, 'gamma': gamma}

# Train the SVM model
svm_model = svm.SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'])
svm_model.fit(X_train_scaled, y_train)

# Hacer predicciones con el modelo entrenado
y_pred = svm_model.predict(X_test_scaled)

# Evaluar el rendimiento
accuracy = accuracy_score(y_test, y_pred)
print(f'C = {best_params["C"]}, gamma = {best_params["gamma"]}')
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
