#!/usr/bin/env python
import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse.linalg as sla
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
#from mnist import MNIST
import pickle
import os
sns.set()
sns.set_style("ticks")
#sns.set_context("paper")
np.random.seed(0)
print("modules loaded")
 
m = 100
n = 24983
ds=[1,2,5,10,20,50]

def load_data(myfile):
	df = np.genfromtxt(myfile, delimiter=',')
	#n = df[0].max()
	#om = df[1].max()
	data = np.full((n,m), np.nan)
	for row in df:
		i = int(row[0])-1; j = int(row[1])-1; s = row[2]
		#print(i,j)
		data[i,j] = s
	#print(data)
	return(data)


def error(predict, test):
	div = np.sum(~np.isnan(test))
	diff = np.abs(predict - test)
	mae = np.nansum(diff)/div
	mse = np.nansum( np.square(diff) )/div
	return(mse, mae)



def partA(train, test):
	colmean = np.nanmean(train, axis=0).reshape(m,1)
	u = np.ones(n).reshape(n,1)
	predict = u.dot(colmean.T)	
	#print(predict)
	mse, mae = error(predict, test)
	print("Part A:\tMSE:{}\tMAE:{}".format(mse, mae))


def doSVD(X, d):
	u,s,v  = sla.svds(X, k = d)
	return(u,s,v)




def partB(train, test):
	ztrain = train.copy()
	ztrain[np.isnan(ztrain)] = 0.0
	mse_l = []
	mae_l = []
	for d in ds:
		u, s, v = doSVD(ztrain, d)
		print(u.shape, s.shape, v.shape)
		predict = u.dot(v)
		mse, mae = error(predict, test)
		mse_l.append(mse)
		mae_l.append(mae)
		print("Part B:\tMSE:{}\tMAE:{}".format(mse, mae))

	fig, axs = plt.subplots(ncols = 2, figsize=(16,9))
	sns.lineplot(ds, mse_l, ax = axs[0]); axs[0].set_xlabel("d"); axs[0].set_ylabel("MSE")
	sns.lineplot(ds, mae_l, ax = axs[1]); axs[1].set_xlabel("d"); axs[1].set_ylabel("MAE")
	plt.savefig("5b.pdf")
	#print(train)
	return()

train = load_data("data/train.txt") 
test = load_data("data/test.txt") 

partA(train, test)
partB(train, test)





