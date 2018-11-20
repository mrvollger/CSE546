#!/usr/bin/env python
import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse.linalg as sla
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
#from mnist import MNIST
import pickle
import os
import pandas as pd
sns.set()
sns.set_style("ticks")
#sns.set_context("paper")
np.random.seed(0)
print("modules loaded")
from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(4)

m = 100
n = 24983
ds=[1,2,5,10,20,50]
Ls = np.float_power( 10, np.arange(-2, 5) )

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
	mse, mae = error(predict, train)
	print("Part A train:\tMSE:{}\tMAE:{}".format(mse, mae))
	mse, mae = error(predict, test)
	print("Part A test:\tMSE:{}\tMAE:{}".format(mse, mae))


def doSVD(X, d):
	u,s,vt  = sla.svds(X, k = d)
	return(u,s,vt)

def partB(train, test):
	ztrain = train.copy()
	ztrain[np.isnan(ztrain)] = 0.0
	ztrain = csr_matrix(ztrain)
	mse_l = [] ; mae_l = []; mse_t = []; mae_t = []
	for d in ds:
		u, s, v = doSVD(ztrain, d)
		predict = (u*s).dot(v)
		
		mse, mae = error(predict, train)
		mse_l.append(mse)
		mae_l.append(mae)
		mse, mae = error(predict, test)
		mse_t.append(mse)
		mae_t.append(mae)

		#print("Part B:\tMSE:{}\tMAE:{}".format(mse, mae))

	fig, axs = plt.subplots(ncols = 2, figsize=(16,9))
	sns.lineplot(ds, mse_t, ax = axs[0], label="test"); axs[0].set_xlabel("d"); axs[0].set_ylabel("MSE")
	sns.lineplot(ds, mae_t, ax = axs[1], label="test"); axs[1].set_xlabel("d"); axs[1].set_ylabel("MAE")
	
	sns.lineplot(ds, mse_l, ax = axs[0], label="train"); axs[0].set_xlabel("d"); axs[0].set_ylabel("MSE")
	sns.lineplot(ds, mae_l, ax = axs[1], label="train"); axs[1].set_xlabel("d"); axs[1].set_ylabel("MAE")
	plt.savefig("5b.pdf")
	#print(train)
	return()


def update_vj(vj, U1, R1, L):
	# vj is d by 1
	# U1 is z by d where z is the number of users at j that have actually rated the movie
	# R1 is z by 1
	z, d = U1.shape

	# (U1T U1 + lambda I ) v_j = U1T R1
	# put in rems of A x = b
	A = U1.T.dot(U1) + L*np.identity(d)
	b = U1.T.dot(R1)
	vj = la.solve(A, b)
	return(vj)

def updateVT(U, VT, R, L):
	d, m = VT.shape
	for j in range(d):
		Rtmp = R[:, j ]
		has_value = Rtmp.indices
		z = has_value.shape[0]
		R1 = Rtmp.data.reshape(z, 1)
		U1 = U[has_value, :]
		vj = VT[:, j].reshape(d, 1)
		#print(vj.shape, U1.shape, R1.shape)
		
		# update step
		vj = update_vj(vj, U1, R1, L) 
		VT[:,j] = vj[:, 0] 
	
	return(VT)

def update_ui(ui, VT1, R1, L):
	# ui is 1 by d
	# VT1 is d by z where z is the up to m in length  (observed jokes in R for that row)
	# R1 is 1 by z 
	d = ui.shape[1]
	z = R1.shape[1]

	# (VT1 V1 + lambda I ) uiT = VT1 RT1 
	# put in terms of A x = b
	A = VT1.dot(VT1.T) + L * np.identity(d) 
	b = VT1.dot(R1.T) 
	uTi = la.solve(A, b)
	ui = uTi.T
	#might need a reshape in here 
	return(ui)

def updateU(U, VT, R, L):
	n, d = U.shape	
	for i in range(n):
		Rtmp = R[i, : ]
		has_value = Rtmp.indices
		z = has_value.shape[0]
		R1 = Rtmp.data.reshape(1, z)
		VT1 = VT[:, has_value]
		
		# solve for ui update 
		ui = U[i, :].reshape(1, d)
		ui = update_ui(ui, VT1, R1, L)
		U[i, :] = ui
	
	return(U)


# Rrow is for slicing along rows, Rcol is for slicing alongs columns 
def makeTrainSmaller(train):
	R = train.copy()
	R[np.isnan(R)] = 0.0

	Rrow = csr_matrix(R)
	Rcol = csc_matrix(R)
	
	#tmp = Rrow[0,:]
	#print(tmp.indices)
	#print(tmp.data)
	
	return(Rrow, Rcol)

def myprint(d, L, mse, mae):
	print("{}_{}:\tMSE:{}\tMAE:{}".format(d, L, mse, mae))

def run_d_l(params):
	Rrow, Rcol, val, train, d, L = params
	U = np.random.rand(n, d)	
	VT = np.random.rand(d, m)	
	predict = U.dot(VT)
	mse, mae = error(val, predict)	
	myprint(d, L, mse, mae)

	i = 0
	thresh = 0.005
	while True:
		U = updateU(U, VT, Rrow, L)
		VT = updateVT(U, VT, Rcol, L)
		predict = U.dot(VT)
		newmse, newmae = error(val, predict)	
		myprint(d, L, newmse, newmae)
	
		# check if I should terminate 
		if( (np.abs(newmae - mae) < thresh) and (np.abs(newmse - mse) < thresh) ):
			break
		else:
			mse = newmse
			mae = newmae
		i += 1 
	return(U, VT, mse, mae)

def makeValidation(train):
	has_value = ~np.isnan(train)
	total = np.sum(has_value)
	valsize = int(total / 5)
	rowidx, colidx = np.where(has_value)
	
	# choose some values for valiation 
	idxs = np.random.choice(total, size=valsize, replace=False)
	valrow = rowidx[idxs]
	valcol = colidx[idxs]
	
	# set the val positions and clear the train positions 
	val = np.full(train.shape, np.nan) 
	val[valrow, valcol]	 = train[valrow, valcol]
	train[valrow, valcol] = np.nan
	
	#print(total, np.sum(~np.isnan(train)), np.sum(~np.isnan(val)))
	return(train, val)

def partC(test, train):
	print("Starting part C")

	# make a validation set 
	train, val = makeValidation(train)
	# show the total error in test and train with jsut guessing zeors acorss the baord 
	print(error(test, np.zeros(test.shape)))
	print(error(val, np.zeros(val.shape)))

	# make spare matrixs
	Rrow, Rcol = makeTrainSmaller(train)
	params = []
	for d in ds:
		for L in Ls:
			param = (Rrow, Rcol, val, train, d, L)
			params.append(param)
	
	results = pool.map(run_d_l, params)
	print(results)

train = load_data("data/train.txt") 
test = load_data("data/test.txt") 

partA(train, test)
#partB(train, test)
partC(train, test)




