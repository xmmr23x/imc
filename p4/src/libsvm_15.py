#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 10:14:36 2022

@author: pedroa
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn import svm
from sklearn import preprocessing
from sklearn import model_selection

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Load the dataset
test    = pd.read_csv('../BasesDatos/csv/test_spam.csv',header=None)
X_test  = test.iloc[:,:-1].values
y_test  = test.iloc[:,-1].values

train   = pd.read_csv('../BasesDatos/csv/train_spam.csv',header=None)
X_train = train.iloc[:,:-1].values
y_train = train.iloc[:,-1].values

# SVM model
svm_model = svm.SVC(kernel='linear', C=0.1)
svm_model.fit(X_train, y_train)

# Hacer predicciones con el modelo entrenado
y_pred = svm_model.predict(X_test)

# Matriz de confusion
conf_matrix = confusion_matrix(y_test, y_pred)

# Analizamos los correos en los que se equivoca
errores = []

for i in range(y_pred.shape[0]):
	if y_pred[i] != y_test[i]:
		print(f"Correo {i}: Valor estimado {y_pred[i]}, valor real {y_test[i]}")
		errores.append(i)

for correo in errores:
	print(f"Correo {correo}:")
	print(X_test[correo].size)

# Visualizar la matriz de confusión
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.show()
