#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import scipy
import scipy.linalg as la
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
#from mnist import MNIST
import pickle
import os
#from cvxpy import *
import cvxpy

# getting py torch
#import torch 
#import torch.nn as nn
#import torch.nn.functional as F
#import torchvision
#import torchvision.transforms as transforms
#import torch.optim as optim


sns.set()
sns.set_style("ticks")
np.random.seed(2)
print("modules loaded")



def rbf(x, z, gamma):
	mid = np.linalg.norm(x-z, ord=2)**2
	rtn = np.exp(-gamma* mid)
	return(rtn) 


def makeKernel(X, X2=None, hyper=0.1):
	if(X2 is None):
		X2 = X	
	n = X.shape[0]
	m = X2.shape[0]
	K = np.zeros((m,n))
	for i in range(m):
		x = X2[i,:]
		for j in range(n):
			z = X[j,:]
			K[i, j] = rbf(x,z,hyper)
	
	return(K)


def Fx(x):
	ks = np.array( [0.2, 0.4, 0.6, 0.8] )
	xs = np.array( [x,]*4 ).transpose()
	idx =  np.greater(xs, ks) 
	fx = 10 * np.sum(idx, axis=1)
	return(fx)

def makedata(n=50):
	x = np.arange(0,n)/(n-1)
	fx = Fx(x)
	y = fx + np.random.randn(50)


	return(x.reshape((x.shape[0], 1) ), y.reshape((x.shape[0], 1) ), fx)

def makeD(n=49):
	D = np.zeros((n-1,n))
	for i in range(D.shape[0]):
		for j in range(D.shape[1]):
			if(i == j):
				D[i, j] = -1.0
			elif(i == j - 1):
				D[i, j] = 1.0
			else:
				D[i, j] = 0.0
	return(D)

def kfold(K, y, k = 5):
	idx = np.random.permutation(K.shape[0])
	K_trains = []
	K_vals = []
	y_trains = []
	y_vals = []
	for i in range(k):
		start = int(i*X.shape[0]/k)
		end = int((i+1)*X.shape[0]/k)
		idx_val = idx[start:end]
		idx_train = np.concatenate( (idx[0:start], idx[end:]) )
		# select columns and rows not used by validation
		K_trains.append(  K[idx_train, :][:, idx_train]  ) 
		# select only rows used for validation and then reomove columns no longer used
		K_vals.append(K[idx_val, :][:, idx_train])

		y_trains.append(y[idx_train, :])
		y_vals.append(y[idx_val, :])

	return(K_trains, y_trains, K_vals, y_vals)

def predict(K, alpha):
	f = K.dot(alpha)
	return(f)

def MSE(f, y):
	mse = ((f-y)**2).mean()
	return(mse)


def plots(X, y, train, hyper, L):
	K = makeKernel(X, hyper = hyper)
	D = makeD(K.shape[1])
	alpha = train(K, y, D, L=L)
	f = predict(K, alpha)
	mse = MSE(f, y)
	n = X.shape[0]

	name = "{}_{}.pdf".format(n, train.__name__)

	# redefine fx to use more points 
	x = np.arange(0,1,0.001)
	fx = Fx(x)

	sns.scatterplot(X[:,0], y[:,0], color="green", label="y_i")
	sns.lineplot(x, fx, label="f(x)")
	sns.lineplot(X[:,0], f[:,0], label="f_hat")
	plt.title(train.__name__)
	plt.legend()	
	plt.xlabel("x")
	plt.ylabel("f(x)")
	#plt.ylim(-4.5, 6.5)	

	plt.savefig(name)
	
	plt.clf()
	print(name, mse)



def trainA(K, y, D, L=10**-6):
	# alpah = (HH^T + lambda * I )^-1 y 
	# a x + b
	#a = np.inner(H, H.T) + L * np.identity(H.shape[0]) 
	#a = K + L * np.identity(K.shape[0]) 
	#alpha = la.solve( a, y )
	n = K.shape[0]
	alpha = cp.Variable(n)
	loss = cp.pnorm(cp.matmul(K, alpha) - y, p=2)**2
	reg = cp.pnorm(alpha, p=2)**2
	obj = loss  + L * reg
	print(alpha)
	#alpha = Variable(n)
	#
	#  cost = sum(huber(X.T*beta - Y, 1))
    #  Problem(Minimize(cost)).solve()
    #  huber_data[idx] = fit.value
	#

	return(alpha)


def DoCV(X, y, train, k=10):
	hypers = np.float_power( 10, np.arange(-3, 4, .5) )
	Ls = np.float_power( 10, np.arange(-10, 5, .5) )
	
	results = []
	for hyper in hypers:	
		K = makeKernel(X, hyper = hyper)
		# leave one out cross validation
		K_trains, y_trains, K_vals, y_vals =  kfold(K, y, k = k)
		D = makeD(K_trains[0].shape[1])

		for L in Ls:
			mses = []
			for i in range(k):
				K_train = K_trains[i]; y_train = y_trains[i]; K_val = K_vals[i]; y_val = y_vals[i]
				#print(K_train.shape, y_train.shape, K_val.shape, y_val.shape)
				alpha = train(K_train, y_train, D, L=L)
				f = predict(K_val, alpha)
				mse = MSE(f, y_val)
				mses.append(mse)
			results.append( ( np.mean(mses), hyper, L ) )
	
	best = np.inf; bestidx = 0
	for idx, line in enumerate(results):
		if(line[0] < best):
			best = line[0]
			bestidx = idx
	mse, hyper, L = results[bestidx]
	print(mse, hyper, L)
	return(hyper, L)


X, y, fx  = makedata()
gamma, L = DoCV(X, y, trainA, k=X.shape[0])
plots(X, y, trainA, gamma, L)


