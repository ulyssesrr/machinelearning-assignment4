#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from tabulate import tabulate

dataset = np.genfromtxt('car.data', delimiter=',', dtype=None)

n_samples = dataset.shape[0]
n_features = dataset.shape[1]
print("Car Evaluation dataset: %d amostras(%d características)" % (dataset.shape[0], n_features))

target = dataset[:,-1]
dataset = dataset[:,0:-1]

tle = LabelEncoder()
target = tle.fit_transform(target)
print(np.unique(target))
print(list(tle.classes_))
#exit()

labels_encoders = []
for idx in range(0, n_features-1):
    le = LabelEncoder()
    labels_encoders += [le]
    e = le.fit_transform(dataset[:,idx])
    dataset[:,idx] = e
    #print(list(le.classes_))
#print(len(dataset[0]))


enc=OneHotEncoder(sparse=False)
dataset = enc.fit_transform(dataset)

kf = StratifiedKFold(target, n_folds=3)



folds_accuracy = []
folds_precision_macro_averaged = []
folds_recall_macro_averaged = []
folds_confusion_matrix = []
for fold_idx, (calibration_indexes, test_indexes) in enumerate(kf):
	n_calibration = len(calibration_indexes)
	print("Tamanho Calibração/Validação: %d/%d" % (n_calibration, len(test_indexes)))
	print("Fold: %d" % (fold_idx+1))
	calibration_target = target[calibration_indexes]
	calibration_dataset = dataset[calibration_indexes]
	subkf = StratifiedKFold(calibration_target, n_folds=2)
	best_accuracy =  -1
	train_indexes, validation_indexes = next(iter(subkf))
	for k in range(1,21):
		knn = KNeighborsClassifier(n_neighbors=k)
		knn.fit(calibration_dataset[train_indexes], calibration_target[train_indexes])
		target_pred_train = knn.predict(calibration_dataset[train_indexes])
		target_pred_validation = knn.predict(calibration_dataset[validation_indexes])
		accuracy_train = accuracy_score(calibration_target[train_indexes], target_pred_train)
		accuracy_validation = accuracy_score(calibration_target[validation_indexes], target_pred_validation)
		precision_macro_averaged_train = precision_score(calibration_target[train_indexes], target_pred_train, labels=np.unique(target), average='macro')
		precision_macro_averaged_validation = precision_score(calibration_target[validation_indexes], target_pred_validation, labels=np.unique(target), average='macro')
		recall_macro_averaged_train = recall_score(calibration_target[train_indexes], target_pred_train, labels=np.unique(target), average='macro')
		recall_macro_averaged_validation = recall_score(calibration_target[validation_indexes], target_pred_validation, labels=np.unique(target), average='macro')
		if accuracy_validation > best_accuracy:
			best_k = k
			best_accuracy = accuracy_validation
		print("kNN: k: %d\n|-Acurácia (Treino/Validação): %2.2f%%/%2.2f%%\n|-Macro-precision Médio (Treino/Validação): %2.2f%%/%2.2f%%\n|-Macro-recall Médio (Treino/Validação): %2.2f%%/%2.2f%%" % (k, accuracy_train*100, accuracy_validation*100, precision_macro_averaged_train*100, precision_macro_averaged_validation*100,recall_macro_averaged_train*100,recall_macro_averaged_validation*100))
	print("\nFold: %d" % (fold_idx+1))
	print("Melhor: k: %d - Acurácia (Validação): %2.2f%%" % (best_k, best_accuracy*100))
	knn = KNeighborsClassifier(n_neighbors=best_k)
	knn.fit(calibration_dataset[train_indexes], calibration_target[train_indexes])
	target_pred_test = knn.predict(dataset[test_indexes])
	accuracy_test = accuracy_score(target[test_indexes], target_pred_test)
	folds_accuracy += [accuracy_test]
	precision_macro_averaged_test = precision_score(target[test_indexes], target_pred_test, labels=np.unique(target), average='macro')
	folds_precision_macro_averaged += [precision_macro_averaged_test]
	recall_macro_averaged_test = recall_score(target[test_indexes], target_pred_test, labels=np.unique(target), average='macro')
	folds_recall_macro_averaged += [recall_macro_averaged_test]
	confusion_matrix_test = confusion_matrix(target[test_indexes], target_pred_test)
	folds_confusion_matrix += [np.mat(confusion_matrix_test)]
	print("kNN: k: %d\n|-Acurácia (Teste): %2.2f%%\n|-Macro-precision Médio (Teste): %2.2f%%\n|-Macro-recall Médio (Teste): %2.2f%%\n" % (best_k, accuracy_test*100, precision_macro_averaged_test*100, recall_macro_averaged_test*100))
	print(tabulate(confusion_matrix_test/confusion_matrix_test.sum(axis=1)[:,None]*100, headers=tle.classes_, tablefmt="grid"))
	print(tabulate(confusion_matrix_test, headers=tle.classes_, tablefmt="grid"))
	
#print(folds_accuracy)
#print(folds_precision_macro_averaged)
#print(folds_recall_macro_averaged)
#print(tabulate(np.array(sum(folds_confusion_matrix)), headers=tle.classes_, tablefmt="grid"))
