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

	return(X_train, labels_train, X_test, labels_test)


def fakeData(n=500, d=1000, k =100, sig=1):
	noise = np.random.randn(n)
	x = np.random.randn(n, d)
	w = np.arange(1,d+1)/k
	w[ w > 1] = 0
	Y = (x).dot(w.T) + noise 	
	#print(x.shape, Y.shape, w)
	return(x, Y, w)


def lambdaMax(X, Y):
	Ynorm = Y - np.mean(Y)
	ls = Ynorm.dot(X)
	lmax = 2*np.max(ls)
	#print(ls.shape, lmax)
	return(lmax)	


def objective(X,Y,L,b, w):
	term  = X.dot(w.T) + b - Y
	square = term.dot(term.T)
	reg = L* np.abs(w).sum()
	return(square + reg)

def fitLasso(X, Y, L, w=None, delta = 0.001):
	diff = delta + 1
	n = X.shape[0]
	d = X.shape[1]
	if(w is None):
		w = np.zeros(d)
	else:
		w = w.copy()
	
	# x is not updated in the loop so a_k can be calculated once
	aks = 2*np.sum(np.square(X), axis=0)
	#print(aks.shape)

	while(delta < diff):
		b = np.mean(Y - X.dot(w.T))
		w_old = w.copy()	
		
		#xw = X * w
		#xwsum = xw.sum(axis = 1)
		#print(xwsum.shape)	

		for k in np.arange(d):
			ak = aks[k]
			xk = X[:,k]	
			jnotk = X.dot(w.T) - w[k]*xk
			ck = 2*xk.dot(Y - (b + jnotk) )
			
			if(ck < -L):
				w[k] = (ck + L)/ak
			elif(-L <= ck <= L):
				w[k] = 0
			else:
				w[k] = (ck - L)/ak
		
		diff = np.max(np.abs(w - w_old))
		#print(diff, L)
		#print(w, w_new)
		obj = objective(X, Y, L, b, w)
		print("obj:{:.2f}\tlabmda:{:.4f}\tdiff:{:.4f}".format(obj, L, diff))

	return(w, b)

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

def MSE(Y, X, w, b):
	pred = X.dot(w.T) + b 
	mse = ((Y - pred) ** 2).mean()
	return(mse)

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

def GetGandB(p, d):
	G = np.sqrt(0.1) * np.random.randn(p, d)
	b = np.random.uniform(low=0, high=2*np.pi, size=p)
	return(G, b)

def moveToCos(X_test, X_train, G, b):
	C_test = np.cos( np.transpose(G.dot(np.transpose(X_test))) + b ) 
	C_train = np.cos( np.transpose(G.dot(np.transpose(X_train)) )+ b ) 
	return(C_test, C_train)


def Problem3():
	n=500;d=1000;k=100
	X, Y, realW = fakeData(n=n, d=d, k=k)
	lmax= lambdaMax(X, Y)
	print("Lambda max is:{}".format(lmax))

	L = lmax
	nonZeros = []
	FDR = []
	TPR = []
	Ls = []
	w = np.zeros(d)
	while L > 0.05:
		w,b = fitLasso(X, Y, L, w=w, delta=0.01)
		notzero = (w>0).sum()
		nonZeros.append( notzero )
		TPR.append( (w[0:k] > 0.0).sum()/k  )
		FDR.append( (w[k:] > 0.0).sum()/notzero  )
		Ls.append(L)
		L = L/(np.exp(1)/2)
	
	#
	# P3 part 1
	#
	fig, ax = plt.subplots(figsize=(16,9))
	sns.lineplot(Ls, nonZeros)
	sns.scatterplot(Ls, nonZeros)
	plt.xscale('log')
	ax.set_xlabel("Log Labmda")
	ax.set_ylabel("Number of non zeros")
	sns.despine()
	plt.savefig("P3part1.pdf")

	#
	# P3 part 2
	#
	fig, ax = plt.subplots(figsize=(16,9))
	sns.lineplot(x=FDR, y=TPR, ax=ax)
	sns.scatterplot(x=FDR, y=TPR, ax=ax)
	ax.set_xlabel("FDR")
	ax.set_ylabel("TPR")
	sns.despine()
	plt.savefig("P3part2.pdf")

def Problem4load():
	# Load a csv of floats:
	X = np.genfromtxt("upvote_data.csv", delimiter=",")
	# Load a text file of integers:
	Y = np.sqrt( np.loadtxt("upvote_labels.txt", dtype=np.int) )

	X_train = X[0:4000,:]
	X_val = X[4000:5000,:]
	X_test = X[5000:,:]

	Y_train = Y[0:4000]
	Y_val = Y[4000:5000]
	Y_test = Y[5000:]

	lmax = lambdaMax(X_train, Y_train)
	print("Lambda max is:{}".format(lmax))
	return(X_train, Y_train, X_val, Y_val, X_test, Y_test, lmax)


def Problem4partA(X_train, Y_train, X_val, Y_val, X_test, Y_test, lmax):
	d = X_train.shape[1]
	L = lmax
	nonZeros = []
	mses_val = []
	mses_train = []
	Ls = []
	w = np.zeros(d)
	while L > 0.05:
		w,b = fitLasso(X_train, Y_train, L, w=w, delta=0.25)
		notzero = (w>0).sum()
		nonZeros.append( notzero )
		mse_val = MSE(Y_val, X_val, w, b)
		mse_train = MSE(Y_train, X_train, w, b)
		print(mse_val)
		mses_val.append(mse_val)
		mses_train.append(mse_train)
		Ls.append(L)
		L = L/(2.0)
	

	#
	# P4 part a.1
	#
	fig, ax = plt.subplots(figsize=(16,9))
	sns.lineplot(Ls, nonZeros)
	sns.scatterplot(Ls, nonZeros)
	plt.xscale('log')
	ax.set_xlabel("Log Labmda")
	ax.set_ylabel("Number of non zeros")
	sns.despine()
	plt.savefig("P4partA.1.pdf")

	#
	# P4 part a.2
	#
	fig, ax = plt.subplots(figsize=(16,9))
	sns.lineplot(x=Ls, y=mses_val, ax=ax, label="Validation MSE")
	sns.lineplot(x=Ls, y=mses_train, ax=ax, label="Train MSE")
	plt.xscale('log')
	plt.gca().invert_xaxis()
	ax.set_xlabel("Log Lambda (inverted)")
	ax.set_ylabel("MSE")
	plt.legend()
	sns.despine()
	plt.savefig("P4partA.2.pdf")



	


#
# Problem 3
#
#Problem3()

#
# Problem 4
#
# Load a text file of strings:
featureNames = np.array(open("upvote_features.txt").read().splitlines())

X_train, Y_train, X_val, Y_val, X_test, Y_test, lmax = Problem4load()
#Problem4partA( X_train, Y_train, X_val, Y_val, X_test, Y_test, lmax )

#prioblem4partb
# from part A I found that lambda = 1.4618 was the best value I tested

L=0.7309
w,b = fitLasso(X_train, Y_train, L, delta=0.25)
notzero = (w>0).sum()
mse_train = MSE(Y_train, X_train, w, b)
mse_val = MSE(Y_val, X_val, w, b)
mse_test = MSE(Y_test, X_test, w, b)
print("Train error:{}\tValidation error:{}\tTest error:{}".format(mse_train, mse_val, mse_test))


#
# Problem 4 part c
#
idxs = list(reversed(sorted(range(len(w)), key=lambda i: w[i])[-10:]))
for weight, feat in zip(w[idxs], featureNames[idxs]):
	print("{}:{} \\\\".format(weight, feat))




