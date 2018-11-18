#!/usr/bin/env python
import numpy as np
import scipy
import scipy.linalg as la
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from mnist import MNIST
import pickle
import os
sns.set()
sns.set_style("ticks")
np.random.seed(0)

def load_data():
	savedf = "data.pkl"
	if(os.path.exists(savedf)):
		print("Reading: " + savedf)
		with open(savedf, 'rb') as input:
			X_train = pickle.load(input)
			labels_train = pickle.load(input)
			X_test = pickle.load(input)
			labels_test = pickle.load(input)
	else:
		mndata = MNIST('python-mnist/data/')
		X_train, labels_train = map(np.array, mndata.load_training())
		X_test, labels_test = map(np.array, mndata.load_testing())
		X_train = X_train/255.0
		X_test = X_test/255.0
		with open(savedf, 'wb') as output:
			pickle.dump(X_train, output, pickle.HIGHEST_PROTOCOL)
			pickle.dump(labels_train, output, pickle.HIGHEST_PROTOCOL)
			pickle.dump(X_test, output, pickle.HIGHEST_PROTOCOL)
			pickle.dump(labels_test, output, pickle.HIGHEST_PROTOCOL)
	
	X = np.vstack(X_test, X_train)
	y = np.vstack(labels_test, labels_train)
	return(X, y)


def kfold(X, Y, k = 5):
	idx = np.random.permutation(X.shape[0])
	X_trains = []
	X_tests = []
	Y_trains = []
	Y_tests = []
	for i in range(k):
		start = int(i*X.shape[0]/k)
		end = int((i+1)*X.shape[0]/k)
		idx_test = idx[start:end]
		idx_train = np.concatenate( (idx[0:start], idx[end:]) ) 
		X_trains.append(X[idx_train, :])
		X_tests.append(X[idx_test, :])
		Y_trains.append(Y[idx_train, :])
		Y_tests.append(Y[idx_test, :])
	
	#for i in range(k):
	#	print(X_tests[i].shape, Y_tests[i].shape, X_trains[i].shape, Y_trains[i].shape)

	#num = X.shape[0]	
	#test_idx = np.random.random_integers(0, high=num-1, size=int(num*.2))		
	#mask = np.zeros(num, dtype=bool)
	#mask[test_idx] = True
	#X_test = X[mask, :]
	#Y_test = Y[mask, :]
	#X_train = X[~mask, :]
	#Y_train = Y[~mask, :]
	
	return(X_tests, Y_tests, X_trains, Y_trains)

def SelStart(X, k = 5):
	n = X.shape[0]
	tmp = np.arange(n)
	selidx = np.random.choice(tmp, size=k, replace=False)
	print(selidx)

#
# load / save
#
X, y = load_data()

SelStart(X)


