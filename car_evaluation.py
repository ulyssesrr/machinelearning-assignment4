#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

dataset = np.genfromtxt('car.data', delimiter=',', dtype=None)

n_samples = dataset.shape[0]
n_features = dataset.shape[1]
print("Evaluation dataset: %d amostras(%d características)" % (dataset.shape[0], n_features))

target = dataset[:,-1]
dataset = dataset[:,0:-1]

tle = LabelEncoder()
target = tle.fit_transform(target)
print(target)
print(list(tle.classes_))

labels_encoders = []
for idx in range(0, n_features-1):
    le = LabelEncoder()
    labels_encoders += [le]
    e = le.fit_transform(dataset[:,idx])
    dataset[:,idx] = e
    print(list(le.classes_))
print(len(dataset[0]))


enc=OneHotEncoder(sparse=False)
dataset = enc.fit_transform(dataset)

kf = StratifiedKFold(target, n_folds=3)

round_count = 0
for calibration_indexes, validation_indexes in kf:
	round_count += 1
	if round_count == 1:
		continue
	n_calibration = len(calibration_indexes)
	print("Tamanho Calibração/Validação: %d/%d" % (n_calibration, len(validation_indexes)))
	print("Rodada: %d" % round_count)
	for k in range(1,20):
		n_train_indexes = int(n_calibration/2)
		train_indexes = calibration_indexes[:n_train_indexes]
		test_indexes = calibration_indexes[n_train_indexes:]
		print("Tamanho Treino/Teste: %d/%d" % (len(train_indexes), len(test_indexes)))
		knn = KNeighborsClassifier(n_neighbors=k)
		knn.fit(dataset[train_indexes], target[train_indexes])
		target_pred_train = knn.predict(dataset[train_indexes])
		target_pred_test = knn.predict(dataset[test_indexes])
		accuracy_train = accuracy_score(target[train_indexes], target_pred_train)
		accuracy_test = accuracy_score(target[test_indexes], target_pred_test)
		#print("precision_score([%s], [%s], average='macro')" % ("','".join(target[train_indexes]), "','".join(target_pred_train)))
		precision_macro_averaged_train = precision_score(target[train_indexes], target_pred_train, average='macro')
		precision_macro_averaged_test = precision_score(target[test_indexes], target_pred_test, average='macro')
		print("kNN: k: %d - Acurácia (Treino/Teste): %0.2f/%0.2f - Precisão Macro Averaged(Treino/Teste): %0.2f/%0.2f" % (k, accuracy_train, accuracy_test, precision_macro_averaged_train, precision_macro_averaged_test))
			
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(dataset, target)
 
print(knn.predict(dataset))
print(knn.predict([dataset[-1]]))
