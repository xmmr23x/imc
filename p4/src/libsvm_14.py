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
test    = pd.read_csv('../BasesDatos/csv/test_spam.csv',header=None)
X_test  = test.iloc[:,:-1].values
y_test  = test.iloc[:,-1].values

train   = pd.read_csv('../BasesDatos/csv/train_spam.csv',header=None)
X_train = train.iloc[:,:-1].values
y_train = train.iloc[:,-1].values

# SVM model
svm_model = svm.SVC(kernel='linear')

# Obtencion de los valores para C y gamma
# genera arrays de numeros distribuidos uniformemente en escala logaritmica
# pimer parametro: inicio
# segundo parametro: fin
# num: cantidad de elementos del array
# base: base
# ejemplo desde 2⁻⁵ hasta 2¹⁵
Hp     = np.logspace(-2, 1, num=4, base=10)
optimo = model_selection.GridSearchCV(estimator=svm_model, param_grid=dict(C=Hp,gamma=Hp), n_jobs=-1,cv=3)

# Entrenamiento
optimo.fit(X_train, y_train)

# Hacer predicciones con el modelo entrenado
y_pred = optimo.predict(X_test)

# Evaluar el rendimiento (usamos accuracy, puedes añadir otras métricas si lo deseas)
accuracy = accuracy_score(y_test, y_pred)
print('C = ', optimo.best_params_['C'])
print('gamma = ', optimo.best_params_['gamma'])
print('k = ', optimo.cv)
print(f'Accuracy: {accuracy:.4f}')

# Show the points
X = X_train
y = y_train
plt.figure(1)
plt.clf()
plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)
