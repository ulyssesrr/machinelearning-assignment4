#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import combinations

import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid

def load_nebulosa_train():
	return np.genfromtxt('nebulosa_train.txt', delimiter=' ', dtype=np.float64)
	
def load_nebulosa_test():
	return np.genfromtxt('nebulosa_test.txt', delimiter=' ', dtype=np.float64)



def itemA():
	train_dataset = load_nebulosa_train();
	
	train_target = train_dataset[:,-1]
	train_dataset = train_dataset[:,:-1]
	
	nam_target = np.where(np.isnan(train_target))
	train_target = np.delete(train_target, nam_target)
	train_dataset = np.delete(train_dataset, nam_target, 0)
	train_dataset = np.nan_to_num(train_dataset)
	
	
	test_dataset = load_nebulosa_test()
	
	test_target = test_dataset[:,-1]
	test_dataset = test_dataset[:,:-1]
	
	nam_target = np.where(np.isnan(test_target))
	test_target = np.delete(test_target, nam_target)
	test_dataset = np.delete(test_dataset, nam_target, 0)
	test_dataset = np.nan_to_num(test_dataset)
	
	n_train_samples = train_dataset.shape[0]
	n_train_features = train_dataset.shape[1]
	print("Nebulosa Train dataset: %d amostras(%d características)" % (n_train_samples, n_train_features))
		
	n_test_samples = test_dataset.shape[0]
	n_test_features = test_dataset.shape[1]
	print("Nebulosa Test dataset: %d amostras(%d características)" % (n_test_samples, n_test_features))
	
	
	nn = KNeighborsClassifier(n_neighbors=1)
	nn.fit(train_dataset, train_target)
	nn_target_pred_test = nn.predict(test_dataset)

	nn_accuracy_test = accuracy_score(test_target, nn_target_pred_test)
	print("NN: Acurácia (Teste): %.2f" % (nn_accuracy_test))

	#train_target[18] = 1
	nc = NearestCentroid(metric='euclidean')
	nc.fit(train_dataset, train_target)
	nc_target_pred_test = nc.predict(test_dataset)
	#print(nc_target_pred_test)
	
	nc_accuracy_test = accuracy_score(test_target, nc_target_pred_test)
	print("Rocchio: Acurácia (Teste): %.2f" % (nc_accuracy_test))

def itemB():
	train_dataset = load_nebulosa_train();
	# remover missing values
	#print(train_dataset)
	train_dataset = train_dataset[~np.isnan(train_dataset).any(axis=1)]
	train_dataset = train_dataset[:,2:]
	
	train_target = train_dataset[:,-1]
	train_dataset = train_dataset[:,:-2]
	
	#train_dataset = normalize(train_dataset, axis=0)
	
	test_dataset = load_nebulosa_test();
	#remover mising values
	test_dataset = test_dataset[~np.isnan(test_dataset).any(axis=1)]
	test_dataset = test_dataset[:,2:]
	
	test_target = test_dataset[:,-1]
	test_dataset = test_dataset[:,:-2]
	#print(test_dataset)
	#test_dataset = normalize(test_dataset, axis=1)
	#print(test_dataset)
	
	kbest = SelectKBest(f_classif, k=3).fit(train_dataset, train_target)
	train_dataset = kbest.transform(train_dataset)
	test_dataset = kbest.transform(test_dataset)
	
	#print(train_dataset)
	   
	n_train_samples = train_dataset.shape[0]
	n_train_features = train_dataset.shape[1]
	#print("Nebulosa Train dataset: %d amostras(%d características)" % (n_train_samples, n_train_features))
		
	n_test_samples = test_dataset.shape[0]
	n_test_features = test_dataset.shape[1]
	#print("Nebulosa Test dataset: %d amostras(%d características)" % (n_test_samples, n_test_features))
	
	nn = KNeighborsClassifier(n_neighbors=1)
	nn.fit(train_dataset, train_target)
	nn_target_pred_test = nn.predict(test_dataset)

	nn_accuracy_test = accuracy_score(test_target, nn_target_pred_test)
	print("NN: Acurácia (Teste): %.2f" % (nn_accuracy_test))

	nc = NearestCentroid(metric='euclidean')
	nc.fit(train_dataset, train_target)
	nc_target_pred_test = nc.predict(test_dataset)

	nc_accuracy_test = accuracy_score(test_target, nc_target_pred_test)
	print("Rocchio: Acurácia (Teste): %.2f" % (nc_accuracy_test))
	
print("===== Letra A")
itemA();
print("===== Letra B")
itemB();
