#!/usr/bin/env python
import numpy as np
import scipy
import scipy.linalg as la
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
#from mnist import MNIST
import pickle
import os
sns.set()
sns.set_style("ticks")
np.random.seed(0)
np.warnings.filterwarnings('ignore')



def load_data(n=30):
	x = np.random.uniform(0, 1, n).reshape(n, 1)
	fx = 4* np.sin(np.pi * x) * np.cos(6 * np.pi * x**2)
	e = np.random.randn(n).reshape(n, 1)
	y = fx + e
	#print(x.shape, fx.shape, e.shape, y.shape)
	return(x, fx, y)


def train(K, y, L=10**-6):
	# alpah = (HH^T + lambda * I )^-1 y 
	# a x + b
	#a = np.inner(H, H.T) + L * np.identity(H.shape[0]) 
	a = K + L * np.identity(K.shape[0]) 
	alpha = la.solve( a, y )
	return(alpha)


def predict(K, alpha):
	f = K.dot(alpha)
	return(f)

def MSE(f, y):
	mse = ((f-y)**2).mean()
	return(mse)

def poly(X, hyper=1):
	d = hyper
	n = X.shape[0]
	K = np.zeros((n,n))
	for i in range(n):
		x = X[i,:]
		for j in range(n):
			z = X[j,:]
			K[i, j] = (1 + x.T.dot(z) )**d

	#print(K, K.shape)
	return(K)


def rbf(X, hyper=100):
	gamma = hyper
	# ||x-y||^2 = ||x||^2 ||y||^2 - s *x^T*y
	#X_norm = np.sum(X**2, axis = -1)
	#Ktest = np.exp(-gamma * (X_norm[:,None] + X_norm[None,:] -2 *np.dot(X, X.T) ) )

	n = X.shape[0]
	K = np.zeros((n,n))
	for i in range(n):
		x = X[i,:]
		for j in range(n):
			z = X[j,:]
			norm = np.linalg.norm(x-z, ord=2)**2
			#print(norm)
			K[i, j] = np.exp( -gamma * norm)
			#print(K[i,j] , Ktest[i, j])
	#print(K, K.shape)
	return(K)


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


def plots(X, fx, y, kernel, hyper, L, p5, p95):
	K = kernel(X, hyper = hyper)
	alpha = train(K, y, L=L)
	f = predict(K, alpha)
	mse = MSE(f, y)
	n = X.shape[0]

	name = "{}_{}.pdf".format(n, kernel.__name__)


	sns.scatterplot(X[:,0], y[:,0], color="green", label="y_i")
	sns.lineplot(X[:,0], fx[:,0], label="f(x)")
	sns.lineplot(X[:,0], f[:,0], label="f_hat")
	plt.title(kernel.__name__)
	plt.legend()	
	plt.xlabel("x")
	plt.ylabel("f(x)")
	
	# add boostrap 		
	#sns.lineplot(X[:,0], p5, label="5 conf")
	#sns.lineplot(X[:,0], p95, label="95 conf")
	sort = np.argsort(X[:,0])
	plt.fill_between(X[:,0][sort], p95[sort], p5[sort], color='grey', alpha=0.25)
	plt.savefig(name)
	
	plt.clf()
	print(name, mse)

def DoCV(X, fx, y, kernel, k=10):
	n = X.shape[0]	
	
	if(kernel == rbf):
		hypers = np.float_power( 10, np.arange(-3, 4, .25) )
	elif(kernel == poly):
		hypers = np.arange(1, 100, 2)
	#print(hypers)
	Ls = np.float_power( 10, np.arange(-3, 3, .25) )
	
	results = []
	for hyper in hypers:	
		K = kernel(X, hyper = hyper)
		# leave one out cross validation
		K_trains, y_trains, K_vals, y_vals =  kfold(K, y, k = k)
		for L in Ls:
			mses = []
			for i in range(k):
				K_train = K_trains[i]; y_train = y_trains[i]; K_val = K_vals[i]; y_val = y_vals[i]
				#print(K_train.shape, y_train.shape, K_val.shape, y_val.shape)
				alpha = train(K_train, y_train, L=L)
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

def bootstrap(X, y, kernel, hyper, L, B=300):
	n = X.shape[0]	
	fs = np.zeros((B, n))
	for i in range(B):
		Xb = np.random.choice(X[:,0], n).reshape(n, 1)
		K = kernel(Xb, hyper = hyper)
		alpha = train(K, y, L=L)
		f = predict(K, alpha)
		fs[i, :] = f[:,0]
	#print(fs.shape)
	
	p5 = np.percentile(fs, 5, axis=0)
	p95 = np.percentile(fs, 95, axis=0)
	print(p5.shape)
	return(p5, p95)


for n in [30, 300]:
	X, fx, y = load_data(n = n)
	
	
	gamma, L = DoCV(X, fx, y, rbf, k=X.shape[0])
	p5, p95 = bootstrap(X,y,rbf, gamma, L)
	plots(X,fx,y,rbf,gamma,L, p5, p95)
	
	d, L = DoCV(X, fx, y, poly, k=X.shape[0])
	p5, p95 = bootstrap(X,y, poly, d, L)
	plots(X,fx,y,poly,d,L, p5, p95)


