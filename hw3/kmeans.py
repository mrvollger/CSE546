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
print("modules loaded")



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
	
	X = np.concatenate((X_test, X_train))
	y = np.concatenate((labels_test, labels_train))
	print(X.shape, y.shape)
	return(X, y)


def objfun(alldists, clusters, k):
	obj = 0
	for i in range(k):
		idxs = (clusters == i)
		cut = alldists[idxs,i]
		obj += np.sum(cut)
	return(obj)

def dists(X, cent):
	dist = np.linalg.norm(X-cent, axis=1, ord=2)
	return(dist)

def distMat(X, cents):
	n = X.shape[0]
	k = cents.shape[1]
	alldists = np.zeros((n,k))
	for i in range(k):
		alldists[:, i] = dists(X, cents[i, :])	
	return(alldists)

def DxProb(alldists):
	rtn = np.square( np.min(alldists, axis=1) )
	rtn2 = rtn / np.sum(rtn)
	print(rtn2.shape)
	return(rtn2)

def findCluster(X, cents):
	k = cents.shape[0]
	alldists = distMat(X, cents)
	clusters = np.argmin(alldists, axis=1)
	return(clusters, alldists)

def getCents(X, clusters, k):
	cents = np.zeros((k, X.shape[1]))
	for i in range(k):
		#print(clusters, i)
		idxs = (clusters == i)
		#print("idxs", idxs)
		cut = X[idxs,:]
		#print(cut)
		cents[i,:] = np.mean(cut, axis=0)	
	return(cents)

def isDone(old, new):
	n = old.shape[0]
	same = np.sum(old==new)
	return(same == n)

def kppStart(X, k):
	n = X.shape[0]
	tmp = np.arange(n)
	selidx = np.random.choice(tmp, size=1, replace=False)
	cents = X[selidx, :]
	# add aditional centers
	for i in range(k-1):
		alldists = distMat(X, cents)
		probs = DxProb(alldists)
		j = 0 
		while(True):
			randint = np.random.randint(0, n)
			thresh = np.random.uniform()
			val = probs[randint]
			if( val > thresh ):
				newcent = X[randint, :].reshape(1,X.shape[1])
				#print(val, thresh, j, newcent.shape)
				cents = np.concatenate((cents, newcent))
				break			
			j += 1
	print(cents.shape)
	return(cents)

def selStart(X, k):
	n = X.shape[0]
	tmp = np.arange(n)
	selidx = np.random.choice(tmp, size=k, replace=False)
	#print(selidx)
	return(X[selidx,:])

def myplot(cents, name, objs):
	k = cents.shape[0]
	
	#obective fucntion
	fig = plt.figure()
	sns.lineplot(np.arange(len(objs)), objs, markers=True, style=1)	
	plt.xlabel("Iteration")
	plt.ylabel("Objective")
	out = name + ".obj." + str(k) + ".pdf"
	fig.suptitle(out)
	plt.savefig(out)

	# vizualization
	t = int(np.ceil(k/5))
	fig, axs = plt.subplots(ncols = 5, nrows = t, figsize=(5*2, t*2))

	for i in range(k):
		ax = axs.flat[i]
		ax.set_aspect("equal")
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		data = cents[i, :].reshape((28,28))
		sns.heatmap(data, ax = ax, cbar=False)
	out = name + ".heat." + str(k) + ".pdf"

	sns.despine(fig, left=True, bottom=True)
	fig.suptitle(out)
	plt.savefig(out)
	
def runlloyd(X, cents, name = "random"):
	k = cents.shape[0]
	objs = []
	clusters, alldists = findCluster(X, cents)
	obj = objfun(alldists, clusters, k)
	objs.append(obj)

	while(True):
		cents = getCents(X, clusters, k)
		newclusters, alldists = findCluster(X, cents)
		newobj = objfun(alldists, newclusters, k)
		objs.append(newobj)
		print(newobj, newobj < obj)
		if(newobj >= obj):
			break
		else:
			obj = newobj
			clusters = newclusters
	myplot(cents, name, objs)
	return(clusters, cents)
	
#
# load / save
#
X, y = load_data()
#X = X[np.random.randint(1, X.shape[0], size=1000), :]

for k in [5, 10, 20]:
	print("rand")
	cents = selStart(X, k=k)
	runlloyd(X, cents)
	print("kpp")
	cents = kppStart(X, k=k)
	runlloyd(X, cents, name="kpp")


