#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import combinations

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

dataset = np.genfromtxt('wine.data', delimiter=',', dtype=np.float64)

n_samples = dataset.shape[0]
n_features = dataset.shape[1]
print("Wine dataset: %d amostras(%d características)" % (dataset.shape[0], n_features-1))

target = dataset[:,0]
dataset = dataset[:,1:]

kf = StratifiedKFold(target, n_folds=3)

def rankAttributesByAccuracy(attributes, train_data, train_target, validation_data, validation_target):
	rank = []
	for attrs in attributes:
		train_data_view = train_data[:,attrs]
		validation_data_view = validation_data[:,attrs]
		nn = KNeighborsClassifier(n_neighbors=1)
		nn.fit(train_data_view, train_target)
		#print(train_data_view)
		#print(train_target)
		#print(validation_data_view)
		target_pred_validation = nn.predict(validation_data_view)
		
		accuracy_validation = accuracy_score(validation_target, target_pred_validation)
		rank += [[attrs , accuracy_validation]]
		
	rank.sort(key=lambda tup: tup[1], reverse=True)
	return rank

def selectAttributesSFS(n_attributes, train_data, train_target, validation_data, validation_target):
	attributes = range(0, train_data.shape[1])
	attributes_selected = []
	for i in range(0, n_attributes):
		attributes_temp = []
		for attr in attributes:
			if not attr in attributes_selected:
				attributes_temp += [attributes_selected + [attr]]
		rank = rankAttributesByAccuracy(attributes_temp, train_data, train_target, validation_data, validation_target)
		#print(rank[0])
		attributes_selected = rank[0][0]
	return attributes_selected
	
def selectAttributesSBE(n_attributes, train_data, train_target, validation_data, validation_target):
	attributes = range(0, train_data.shape[1])
	attributes_selected = list(attributes)
	for i in range(0, train_data.shape[1] - n_attributes + 1):
		attributes_temp = []
		for attr in attributes:
			if attr in attributes_selected:
				attributes_temp += [np.delete(attributes_selected, np.where(attributes_selected == attr), 0)]
		
		rank = rankAttributesByAccuracy(attributes_temp, train_data, train_target, validation_data, validation_target)
		#print(rank[0])
		attributes_selected = rank[0][0]
	return attributes_selected
	


round_count = 0
calibration_indexes, test_indexes = next(iter(kf))
calibration_target = target[calibration_indexes]
calibration_dataset = dataset[calibration_indexes]
test_target = target[test_indexes]
test_dataset = dataset[test_indexes]
subkf = StratifiedKFold(calibration_target, n_folds=2)
train_indexes, validation_indexes = next(iter(subkf))
train_target = calibration_target[train_indexes]
train_dataset = calibration_dataset[train_indexes]
validation_target = calibration_target[validation_indexes]
validation_dataset = calibration_dataset[validation_indexes]

final_train_dataset = np.concatenate([train_dataset, validation_dataset])
final_train_target = np.concatenate([train_target, validation_target])
print("Letra A")
n_attributes = 5

attributes_selected_sfs = selectAttributesSFS(n_attributes, train_dataset, train_target, validation_dataset, validation_target)
accuracy = rankAttributesByAccuracy([attributes_selected_sfs], final_train_dataset, final_train_target, test_dataset, test_target)
accuracy_test_sfs = accuracy[0][1]
attributes_selected_sfs.sort()
print("SFS: %d Atributos selecionados: %s - Acurácia (Teste): %.2f" % (n_attributes, attributes_selected_sfs, accuracy_test_sfs))

attributes_selected_sbe = selectAttributesSBE(n_attributes, train_dataset, train_target, validation_dataset, validation_target)
accuracy = rankAttributesByAccuracy([attributes_selected_sbe], final_train_dataset, final_train_target, test_dataset, test_target)
accuracy_test_sbe = accuracy[0][1]
attributes_selected_sbe.sort()
print("SBE: %d Atributos selecionados: %s - Acurácia (Teste): %.2f" % (n_attributes, attributes_selected_sbe, accuracy_test_sbe))

print("Letra B")
n_attributes = 10

attributes_selected_sfs = selectAttributesSFS(n_attributes, train_dataset, train_target, validation_dataset, validation_target)
accuracy = rankAttributesByAccuracy([attributes_selected_sfs], final_train_dataset, final_train_target, test_dataset, test_target)
accuracy_test_sfs = accuracy[0][1]
attributes_selected_sfs.sort()
print("SFS: %d Atributos selecionados: %s - Acurácia (Teste): %.2f" % (n_attributes, attributes_selected_sfs, accuracy_test_sfs))

attributes_selected_sbe = selectAttributesSBE(n_attributes, train_dataset, train_target, validation_dataset, validation_target)
accuracy = rankAttributesByAccuracy([attributes_selected_sbe], final_train_dataset, final_train_target, test_dataset, test_target)
accuracy_test_sbe = accuracy[0][1]
attributes_selected_sbe.sort()
print("SBE: %d Atributos selecionados: %s - Acurácia (Teste): %.2f" % (n_attributes, attributes_selected_sbe, accuracy_test_sbe))

print("Letra C")

n_attributes = 5

attributes_selected_sfs = selectAttributesSFS(n_attributes, final_train_dataset, final_train_target, final_train_dataset, final_train_target)
accuracy = rankAttributesByAccuracy([attributes_selected_sfs], final_train_dataset, final_train_target, test_dataset, test_target)
accuracy_test_sfs = accuracy[0][1]
attributes_selected_sfs.sort()
print("SFS: %d Atributos selecionados: %s - Acurácia (Teste): %.2f" % (n_attributes, attributes_selected_sfs, accuracy_test_sfs))

attributes_selected_sbe = selectAttributesSBE(n_attributes, final_train_dataset, final_train_target, final_train_dataset, final_train_target)
accuracy = rankAttributesByAccuracy([attributes_selected_sbe], final_train_dataset, final_train_target, test_dataset, test_target)
accuracy_test_sbe = accuracy[0][1]
attributes_selected_sbe.sort()
print("SBE: %d Atributos selecionados: %s - Acurácia (Teste): %.2f" % (n_attributes, attributes_selected_sbe, accuracy_test_sbe))


n_attributes = 10

attributes_selected_sfs = selectAttributesSFS(n_attributes, final_train_dataset, final_train_target, final_train_dataset, final_train_target)
accuracy = rankAttributesByAccuracy([attributes_selected_sfs], final_train_dataset, final_train_target, test_dataset, test_target)
accuracy_test_sfs = accuracy[0][1]
attributes_selected_sfs.sort()
print("SFS: %d Atributos selecionados: %s - Acurácia (Teste): %.2f" % (n_attributes, attributes_selected_sfs, accuracy_test_sfs))

attributes_selected_sbe = selectAttributesSBE(n_attributes, final_train_dataset, final_train_target, final_train_dataset, final_train_target)
accuracy = rankAttributesByAccuracy([attributes_selected_sbe], final_train_dataset, final_train_target, test_dataset, test_target)
accuracy_test_sbe = accuracy[0][1]
attributes_selected_sbe.sort()
print("SBE: %d Atributos selecionados: %s - Acurácia (Teste): %.2f" % (n_attributes, attributes_selected_sbe, accuracy_test_sbe))
# Stratificar 3 folds
#Stratificar o train em 2 folds
