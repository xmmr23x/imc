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

Cs = np.logspace(-2, 1, num=4, base=10)

for C in Cs:
	# Entrenar el modelo para cada valor de C y hacer predicciones
	svm_model = svm.SVC(kernel='linear', C=C)
	svm_model.fit(X_train, y_train)
	y_pred = svm_model.predict(X_test)

	# Mostrar los resultados para cada valor de C
	print('C = ', C)
	print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
