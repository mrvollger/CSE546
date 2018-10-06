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


def hotones(labels):
	cats = 10 # numer of catigories
	rtn = np.zeros((len(labels), cats) )
	rtn[np.arange(len(labels)), labels] = 1
	return(rtn)


# X=data, Y=labels, L=lambda
def train(X, Y, L):
	# wh = (XtX+ LI)^-1XtY
	# I want to solve 
	# (XtX+ LI) wh = XtY
	# in the form
	# a x = b
	a = (np.transpose(X).dot(X) + L*np.identity(X.shape[1]) )
	b = np.transpose(X).dot(Y)
	w_hat = la.solve(a, b)
	return(w_hat)


def predict(X, W):
	p = X.dot(W)
	p = np.argmax(p, axis=1)
	return(p)


def accuracy(p, Y):
	num = Y.shape[0]
	guess = Y[np.arange(num), p] # selects the indexes where we think the label is
	correct = guess.sum() *1.0
	acc = correct / num
	return(acc)


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

	return(X_train, labels_train, X_test, labels_test)

def splitTrainTest(X, Y):
	num = X.shape[0]	
	test_idx = np.random.random_integers(0, high=num-1, size=int(num*.2))		
	mask = np.zeros(num, dtype=bool)
	mask[test_idx] = True
	X_test = X[mask, :]
	Y_test = Y[mask, :]
	
	X_train = X[~mask, :]
	Y_train = Y[~mask, :]
	
	return(X_test, Y_test, X_train, Y_train)


def moveToCos(X_test, X_train, p):


#
# load / save
#
X_train, labels_train, X_test, labels_test = load_data()


#
# Part B
#
labels_train = hotones(labels_train)
labels_test = hotones(labels_test)

#
# Part C
#
L= 10**(-4)
w_hat = train(X_train, labels_train, L)
p_test = predict(X_test, w_hat)
p_train = predict(X_train, w_hat)

a_test = accuracy(p_test, labels_test)
a_train = accuracy(p_train, labels_train)
print("Test accuracy:{:.03f}\nTrain accuracy:{:.03f}".format(a_test, a_train))


#
# Part d
#
X_test, Y_test, X_train, Y_train = splitTrainTest(X_train, labels_train)
w_hat = train(X_train, Y_train, L)
p_test = predict(X_test, w_hat)
p_train = predict(X_train, w_hat)

a_test = accuracy(p_test, Y_test)
a_train = accuracy(p_train, Y_train)
print("Test accuracy:{:.03f}\nTrain accuracy:{:.03f}".format(a_test, a_train))







