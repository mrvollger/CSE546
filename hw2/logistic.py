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
L=0.1


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
	
	# rmeove not 2, 7
	labels_train = labels_train.astype(np.int16)
	labels_test = labels_test.astype(np.int16)

	mask = ( (labels_test == 7) | (labels_test == 2) )
	labels_test = labels_test[mask]
	X_test = X_test[mask,:]
	labels_test[labels_test == 7]  = 1
	labels_test[labels_test == 2]  = -1

	mask = ( (labels_train == 7) | (labels_train == 2) )
	labels_train = labels_train[mask]
	X_train = X_train[mask,:]
	labels_train[labels_train == 7]  = 1
	labels_train[labels_train == 2]  = -1
	
	print(X_train.shape, labels_train.shape, labels_train.sum())
	return(X_train, labels_train, X_test, labels_test)

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
	
	return(X_tests, Y_tests, X_trains, Y_trains)

def descent(X, Y, w, b, newton=False, eta = 0.5):
	# update b 
	u = 1.0/(1.0+np.exp(-Y*(b + X.dot(w)) ) )
	db  = (-Y * (1-u)).mean()
	vb = -db 
	if(newton):
		uu = u * (1-u)
		ddb = (Y * Y * uu ).mean()
		vb = la.solve(ddb, -db)
	b += eta * vb
	
	# update w	
	u = 1.0/(1.0+np.exp(-Y*(b + X.dot(w)) ) )
	xy = np.multiply(X.T, Y)
	dw  = (- xy * (1-u)).mean(axis=1) + 2 * L * w
	vw = -dw 
	if(newton):
		Y = Y.reshape((Y.shape[0], 1))
		u = u.reshape((u.shape[0],1))
		one =  ( (u*(1-u)*Y*Y) )
		#print("one", one.shape)
		d = X.shape[1]; n = X.shape[0]
		outer = np.zeros((d,d))
		for i in range(len(one)):
			const = one[i] 
			xi = X[i,:]
			xit = X.T[:,i]
			outer += const * np.outer(xi, xit)
		outer = outer / n
		print("outer\n",outer, outer.shape)
		ddw = (outer + 2 * L * np.identity(d))
		vw = la.solve(ddw, -dw)
		print(vw.sum())
		#print(vw[0].shape, vw[1].shape)
	w += eta * vw
	
	return(w,b)
	
def objective(X, Y, w, b):
	inside = np.log( 1.0 + np.exp( -Y * (b + X.dot(w)) ) )
	j = inside.mean() + L * np.linalg.norm(w,ord=2)


	pred = b + X.dot(w)
	pred[pred < 0] = -1
	pred[pred >= 0 ] = 1
	correct = np.sum(pred == Y)
	#print(correct)
	error = 1.0 - float(correct) / float(X.shape[0])

	return(j, error)
	
def makePlots(iters, test_j, train_j, test_e, train_e, name):
	fig, ax = plt.subplots(figsize=(16,9))
	sns.lineplot(x=iters, y=test_j, ax=ax, label="Test")
	sns.lineplot(x=iters, y=train_j, ax=ax, label="Train")
	ax.set_xlabel("Iteration")
	ax.set_ylabel("J(w,b)")
	plt.ylim(min(train_j)-.05, .5)
	plt.legend()
	sns.despine()
	plt.savefig(name + ".objective.pdf")
	
	fig, ax = plt.subplots(figsize=(16,9))
	sns.lineplot(x=iters, y=test_e, ax=ax, label="Test")
	sns.lineplot(x=iters, y=train_e, ax=ax, label="Train")
	ax.set_xlabel("Iteration")
	ax.set_ylabel("Fraction incorrect")
	plt.ylim(0,.2)
	plt.legend()
	sns.despine()
	plt.savefig(name + ".error.pdf")



def run(X_train, Y_train, X_test, Y_test, name, eta, itersize, newton=False, batch=0):
	d=X_train.shape[1]
	w = np.zeros(d)
	b = 0

	iters = []; test_j=[]; train_j=[]; test_e = []; train_e = []
	j, error = objective(X_train, Y_train, w, b)
	tj, terror = objective(X_test, Y_test, w, b)
	test_j.append(tj)
	train_j.append(j)
	test_e.append(terror)
	train_e.append(error)
	iters.append(0)
	
	
	# looping over total training data set
	i = 1
	for dataloop in range(0,itersize):
		n = X_train.shape[0]
		idx = np.random.permutation(n)
		X_train = X_train[idx]
		Y_train = Y_train[idx]
		split = n/batch
		Xs = np.array_split(X_train, split)	
		Ys = np.array_split(Y_train, split)
		#print(split, len(Xs))
		
		# looping over a subset for stochastic gradient decent 
		for X_split, Y_split in zip(Xs, Ys):
			w, b = descent(X_split, Y_split, w, b, newton=newton, eta=eta)
			j, error = objective(X_train, Y_train, w, b)
			tj, terror = objective(X_test, Y_test, w, b)
			
			test_j.append(tj)
			train_j.append(j)
			test_e.append(terror)
			train_e.append(error)
			iters.append(i)
			if(i % 100 == 0 or split == 1):
				print(j, error, i, dataloop, X_split.shape)
			i += 1
		
	makePlots(iters, test_j, train_j, test_e, train_e, name)


X_train, Y_train, X_test, Y_test = load_data()
#run( X_train, Y_train, X_test, Y_test, "5b", 0.5, 50, batch=X_train.shape[0]) 
#run( X_train, Y_train, X_test, Y_test, "5c", 0.001, 1, batch=1 ) 
#run( X_train, Y_train, X_test, Y_test, "5d", 0.01, 10, batch=100 ) 
run( X_train, Y_train, X_test, Y_test, "5e", 1, 10, batch=X_train.shape[0], newton=True) 




