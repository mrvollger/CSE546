#!/usr/bin/env python
import numpy as np
#import scipy
import scipy.linalg as la
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
#from mnist import MNIST
#import pickle
#import os
sns.set()
sns.set_style("ticks")
np.random.seed(0)
#np.warnings.filterwarnings('ignore')
print("Modules Loaded")


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

def poly(x, z, d):
	k = (1 + x.T.dot(z) )**d
	return(k)

def rbf(x,z,gamma):
	norm = np.linalg.norm(x-z, ord=2)**2
	k = np.exp( -gamma * norm)
	return(k)

def makeKernel(X, kernel, X2=None, hyper=100):
	if(X2 is None):
		X2 = X	
	n = X.shape[0]
	m = X2.shape[0]
	K = np.zeros((m,n))
	for i in range(m):
		x = X2[i,:]
		for j in range(n):
			z = X[j,:]
			K[i, j] = kernel(x,z,hyper)
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


def plots(X, fx, y, kernel, hyper, L, x, p5, p95):
	K = makeKernel(X, kernel, hyper = hyper)
	alpha = train(K, y, L=L)
	f = predict(K, alpha)
	mse = MSE(f, y)
	n = X.shape[0]

	name = "{}_{}.pdf".format(n, kernel.__name__)

	# redefine fx to use more points 
	fx = 4* np.sin(np.pi * x) * np.cos(6 * np.pi * x**2)

	sns.scatterplot(X[:,0], y[:,0], color="green", label="y_i")
	sns.lineplot(x[:,0], fx[:,0], label="f(x)")
	sns.lineplot(X[:,0], f[:,0], label="f_hat")
	plt.title(kernel.__name__)
	plt.legend()	
	plt.xlabel("x")
	plt.ylabel("f(x)")
	plt.ylim(-4.5, 6.5)	
	# add boostrap 		
	#sns.lineplot(x[:,0], p5, label="5 conf")
	#sns.lineplot(x[:,0], p95, label="95 conf")
	plt.fill_between(x[:,0], p5, p95, color='grey', alpha=0.5)
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
	Ls = np.float_power( 10, np.arange(-10, 5, .25) )
	
	results = []
	for hyper in hypers:	
		K = makeKernel(X, kernel, hyper = hyper)
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
	# make all xs to predict on
	step = .01
	x = np.arange(0, 1 + step, step)
	x = x.reshape(x.shape[0], 1)
	
	testn = 40
	
	fs = np.zeros((B, x.shape[0]))
	for i in range(B):
		if(i % 100 == 0 ):
			print("bootstrap", i)
		idxs = np.random.choice(n, n)
		Xb = X[idxs,:]
		yb = y[idxs,:]
		K = makeKernel(Xb, kernel, hyper = hyper)
		alpha = train(K, yb, L=L)
		
		kx = makeKernel(Xb, kernel, X2=x, hyper = hyper)
		f = predict(kx, alpha)
		#print(f[testn,0])
		#print(kx.shape, alpha.shape, f.shape)
		fs[i, :] = f[:,0]
	#print(fs.shape)
	
	p5 = np.percentile(fs, 5, axis=0)
	p95 = np.percentile(fs, 95, axis=0)

	#print(fs.shape)
	#print(np.sort(fs[:,testn]))
	#print(p5[testn], p95[testn], x[testn])
	return(x, p5, p95)



# find best hyper parameters 

hypers = [177.82794100389228, 47, 5.6234, 41]
Ls = [0.1, 0.316277, 1.778*(10**-12), 0.017782]
kernels = [rbf, poly, rbf, poly]
#hypers = None

i=0
for n in [30, 300]:
	X, fx, y = load_data(n = n)
	k = X.shape[0]
	if(k > 30):
		k = 10
	for kernel in [rbf, poly]:
		if(hypers is None):
			hyper, L = DoCV(X, fx, y, kernel, k=k)
		else:
			hyper = hypers[i]; L = Ls[i]
		print(hyper, L)
		# run bootstrap and plot 
		x, p5, p95 = bootstrap(X,y, kernel, hyper, L)
		plots(X, fx, y, kernel, hyper, L, x, p5, p95)
		i += 1



