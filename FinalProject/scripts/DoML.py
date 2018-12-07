#!/usr/bin/env python
import numpy as np
import scipy
import scipy.linalg as la
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import pandas as pd
#import tensorflow as tf
#from tensorflow.contrib.tensor_forest.python import tensor_forest
from sklearn import svm 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from multiprocessing.dummy import Pool as ThreadPool
import itertools
# TensorFlow and tf.keras
#import tensorflow as tf
#from tensorflow import keras


sns.set()
sns.set_style("ticks")
sns.set_context("paper", font_scale=.4)
np.random.seed(0)

def load_data(validation=False):
	X = np.load("data/data.npy")
	y = np.load("data/labels.npy")
	y = y.ravel()
	# shuffle the data 
	idx = np.random.permutation(X.shape[0])
	X = X[idx,:]
	y = y[idx]
	
	# remove uninformative
	haschange = (X.ptp(axis = 0) > 0.0)
	X = X[:, haschange]

	# normalize the data, important? 
	X = (X - X.min(axis=0)) / X.ptp(axis=0)


	# split into test and train , test will be ~20% of the data
	test = np.arange( int(X.shape[0]/10) )
	train = np.arange( int(X.shape[0]/10) , X.shape[0])
	X_train, X_test = X[train, :], X[test, :]
	
	if(validation):
		val = np.arange( int(X.shape[0]/10) , int(X.shape[0]/5))
		train = np.arange( int(X.shape[0]/5) , X.shape[0])
		X_val, y_val = X[val, :], y[val] 

	X_train, X_test = X[train, :], X[test, :]
	y_train, y_test = y[train], y[test]
	
	
	features = open("data/dlabels.txt").readlines()[0].strip().split()
	if(validation):
		#print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
		return(X_train, y_train, X_val, y_val, X_test, y_test, features)
	else:
		#print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
		return(X_train, y_train, X_test, y_test, features)

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

def accuracy(predict, y):
	types = np.sort(np.unique(y))
	rtn = []
	for val in types:
		idx = (y == val)
		total = np.sum(idx)*1.0
		correct = np.sum(predict[idx] == y[idx])
		rtn.append( correct/total )
	return(rtn)


def draw_heatmap(*args, **kwargs):
	data = kwargs.pop('data')
	d = data.pivot(index=args[1], columns=args[0], values=args[2])
	return(sns.heatmap(d, **kwargs))

def SVM(params):
	X_train, y_train, X_val, y_val, gamma, C, kernel = params 
	
	filename = "SVM_models/{}_{}_{}.pkl".format(gamma, C, kernel)
	if(os.path.exists(filename)):
		model = pickle.load(open(filename, 'rb'))
	else:
		print("Gamma:{}\tC:{}\tKernel:{}\t".format(gamma, C, kernel))
		model = svm.SVC(gamma = gamma, C=C, kernel=kernel, cache_size=2000)
		model.fit(X_train, y_train)
		pickle.dump(model, open(filename, 'wb'))
	
	# get stats of model	
	val = model.predict(X_val)
	acc = accuracy(val, y_val)

	return(model, gamma, C, kernel, acc[0], acc[1])

def RF(params):
	X_train, y_train, X_val, y_val, n_trees, max_features, min_samples_leaf = params
	
	filename = "RF_models/{}_{}_{}.pkl".format(n_trees, max_features, min_samples_leaf)
	if(os.path.exists(filename)):
		model = pickle.load(open(filename, 'rb'))
	else:
		print("nt:{}\tnf:{}\tminleaf:{}\t".format( n_trees, max_features, min_samples_leaf))
		model = RandomForestClassifier(n_estimators=n_trees, max_features=max_features, min_samples_leaf=min_samples_leaf,random_state=0)
		model.fit(X_train, y_train)
		pickle.dump(model, open(filename, 'wb'))

	# get stats of model	
	val = model.predict(X_val)
	acc = accuracy(val, y_val)

	return(model, n_trees,max_features, min_samples_leaf,  acc[0], acc[1])

	

def Gboost(params):
	X_train, y_train, X_val, y_val, n_trees, lr, loss = params
	
	filename = "GB_models/{}_{}_{}.pkl".format(n_trees, lr, loss)
	if(os.path.exists(filename)):
		model = pickle.load(open(filename, 'rb'))
	else:
		print("nt:{}\tlr:{}\tloss:{}\t".format( n_trees, lr, loss))
		model = GradientBoostingClassifier(n_estimators=n_trees, learning_rate = lr, loss=loss)
		model.fit(X_train, y_train)
		pickle.dump(model, open(filename, 'wb'))

	# get stats of model	
	val = model.predict(X_val)
	acc = accuracy(val, y_val)

	return(model, n_trees, lr, loss,  acc[0], acc[1])

	
def plotPCA(X, y):
	#from sklearn.decomposition import PCA as dim_red
	from sklearn.manifold import TSNE as dim_red
	matplotlib.rcParams['lines.markeredgewidth'] = 0
	sklearn_pca = dim_red(n_components=2)
	pca = sklearn_pca.fit_transform(X)
	print(pca, pca.shape, y.shape, X.shape)

	plt.figure(figsize=(6, 4))
	for lab, col, marker in zip(('False Alignment', 'True Alignment'), ('#B22222', 'black'), ('X', "+")):
		label = 0
		if(lab == "True Alignment"):
			label = 1	
		plt.scatter(pca[y==label, 0], pca[y==label, 1], label=lab, color=col, alpha=0.75, marker=marker )
	
	plt.xlabel('Component 1')
	plt.ylabel('Component 2')
	plt.legend()
	plt.savefig("dim_red.pdf")	

def confusion(df, algo):
	sns.set_context("paper", font_scale=1.5)
	
	bestM = df.ix[ df["Average"].idxmax() ]["model"]
	print(bestM)
	y_true = y_test
	y_pred = bestM.predict(X_test)
	class_names = ['False Alignment', 'True Alignment']

	cm = confusion_matrix(y_true, y_pred)
	cm = cm/cm.sum(axis=1)[:,None]
	print(cm)
	df_cm = pd.DataFrame(cm, index=class_names, columns=class_names,)
	fig = plt.figure()
	heatmap = sns.heatmap(df_cm, annot=True)
	heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=90, va='center')#, ha='right')
	heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0) #, ha='right')
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	#plt.title(algo)
	plt.savefig("conf_{}.pdf".format(algo))
	sns.set_context("paper", font_scale=.4)

#
# set up
#
X_train, y_train, X_val, y_val, X_test, y_test, features = load_data(validation=True)
n, d = X_train.shape
#plotPCA(X_train, y_train)

threads = 16
pool = ThreadPool(threads)
ML=[ "SVM", "RF", "GB"]



if("SVM" in ML):
	#
	# SVM
	#
	print("Running SVM")
	Cs = np.float_power( 10, np.arange(-1, 7) )
	gammas = np.float_power( 10, np.arange(-9, 2) )
	kernals = ["rbf", "sigmoid"]
	svmParams = []
	for kernal in kernals:
		for gamma in gammas:
			for C in Cs:
				params = X_train, y_train, X_val, y_val, gamma, C, kernal
				svmParams.append(params)
		

	svmrtn = pool.map(SVM, svmParams)



	svm_df = pd.DataFrame(svmrtn, columns=['model', 'gamma', 'C', "kernel", 'Accuracy_False', 'Accuracy_True'])
	svm_df["Average"] = svm_df[["Accuracy_False", "Accuracy_True"]].mean(axis=1)
	confusion(svm_df, "SVM")
	
	
	svm_df = pd.melt(svm_df, id_vars=['model', 'gamma', 'C', "kernel"], value_vars=['Accuracy_False', 'Accuracy_True', "Average"]  )
	fig, ax = plt.subplots(figsize=(20,20))
	fg = sns.FacetGrid(svm_df, col="kernel", row="variable")
	p = fg.map_dataframe(draw_heatmap, 'gamma', 'C', 'value', vmin=0, vmax=1, annot=True)
	#p.savefig("SVM_heat.pdf")
	#p.show()
	plt.savefig("SVM_heat.pdf")



	
if("RF" in ML):
	#
	# Random Forest
	#
	print("Running Random Forest")
	n_tree_l = np.arange(5,1000,100)
	max_features_l = np.append( np.sqrt(np.array([.1,1,10,100]) * int(X_train.shape[1])), [X_train.shape[1]]).astype(int)
	print(max_features_l)
	min_samples_leaf_l = np.arange(2,5)
	big_l = [n_tree_l, max_features_l, min_samples_leaf_l]
	rfParams = []
	for n_tree, max_feats, min_samples_leaf in list(itertools.product(*big_l)):
		params = X_train, y_train, X_val, y_val, n_tree, max_feats, min_samples_leaf
		#print(params[4:])
		rfParams.append(params)

	rfrtn = pool.map(RF, rfParams)

	cols = ['model', 'num_trees', 'max_features', "min_samples_leaf", 'Accuracy_False', 'Accuracy_True']
	rf_df = pd.DataFrame(rfrtn, columns=cols)
	rf_df["Average"] = rf_df[["Accuracy_False", "Accuracy_True"]].mean(axis=1)
	confusion(rf_df, "Random_Forest")

	rf_df = pd.melt(rf_df, id_vars=cols[:-2], value_vars=['Accuracy_False', 'Accuracy_True', "Average"]  )
	fig, ax = plt.subplots(figsize=(20,20))
	fg = sns.FacetGrid(rf_df, col=cols[3], row="variable")
	p = fg.map_dataframe(draw_heatmap, 'num_trees', 'max_features', 'value', vmin=0, vmax=1, annot=True)
	plt.savefig("RF_heat.pdf")



if("GB" in ML):
	#
	# Gboost
	#
	print("Running Gboost")
	n_tree_l = np.arange(50,5000,400)
	learning_rate_l = np.float_power( 10, np.arange(-4, 1.5, 0.5) )
	loss_l = ["deviance", "exponential"]
	big_l = [n_tree_l, learning_rate_l, loss_l]
	gbParams = []
	for n_tree, lr, loss in list(itertools.product(*big_l)):
		params = X_train, y_train, X_val, y_val, n_tree, lr, loss
		gbParams.append(params)

	gbrtn = pool.map(Gboost, gbParams)

	cols = ['model', 'num_trees', 'learning_rate', "loss_function", 'Accuracy_False', 'Accuracy_True']
	gb_df = pd.DataFrame(gbrtn, columns=cols)
	gb_df["Average"] = gb_df[["Accuracy_False", "Accuracy_True"]].mean(axis=1)
	confusion(gb_df, "Gradient_Boosting_Trees")

	cols.append("Average")
	gb_df = pd.melt(gb_df, id_vars=cols[:-3], value_vars=cols[-3:]  )
	print(gb_df)
	fig, ax = plt.subplots(figsize=(20,20))
	fg = sns.FacetGrid(gb_df, col=cols[3], row="variable")
	p = fg.map_dataframe(draw_heatmap, 'num_trees', 'learning_rate', 'value', vmin=0, vmax=1, annot=True)
	plt.savefig("GB_heat.pdf")









