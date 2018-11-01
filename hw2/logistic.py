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
	
	#for i in range(k):
	#	print(X_tests[i].shape, Y_tests[i].shape, X_trains[i].shape, Y_trains[i].shape)

	#num = X.shape[0]	
	#test_idx = np.random.random_integers(0, high=num-1, size=int(num*.2))		
	#mask = np.zeros(num, dtype=bool)
	#mask[test_idx] = True
	#X_test = X[mask, :]
	#Y_test = Y[mask, :]
	#X_train = X[~mask, :]
	#Y_train = Y[~mask, :]
	
	return(X_tests, Y_tests, X_trains, Y_trains)

def GetGandB(p, d):
	G = np.sqrt(0.1) * np.random.randn(p, d)
	b = np.random.uniform(low=0, high=2*np.pi, size=p)
	return(G, b)

def moveToCos(X_test, X_train, G, b):
	C_test = np.cos( np.transpose(G.dot(np.transpose(X_test))) + b ) 
	C_train = np.cos( np.transpose(G.dot(np.transpose(X_train)) )+ b ) 
	return(C_test, C_train)


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
X_tests, Y_tests, X_trains, Y_trains = kfold(X_train, labels_train)

# read it in if I have already done it
if(os.path.exists("crossval.txt")):
	pass
else:
	ps = list(range(1000, 14001, 1000))
	crossval = open("crossval.txt", "w+")
	
	for p in ps:
		G, b =  GetGandB(p, X_train.shape[1] )
		counter = 1
		for X_test, X_train, Y_test, Y_train in zip(X_tests, X_trains, Y_tests, Y_trains):
			C_test, C_train = moveToCos(X_test, X_train, G, b)
			w_c = train(C_train, Y_train, L)

			p_test = predict(C_test, w_c)
			p_train = predict(C_train, w_c)

			a_test = accuracy(p_test, Y_test)
			a_train = accuracy(p_train, Y_train)

			result = "{}\t{}\t{}\t{}\n".format(a_test, a_train, p, counter)
			print(result[:-1])
			crossval.write(result)
			counter += 1

	crossval.close()

results = np.loadtxt("crossval.txt")
fig, ax = plt.subplots(figsize=(16,9))
for fold in range(1, 5+1):
	idx = results[:,3] == fold
	tbl = results[idx, :]
	#print(tbl)
	ps = tbl[:,2]
	acc_test_p = tbl[:,0]
	acc_train_p = tbl[:,1]
	sns.lineplot(ps, 1-acc_test_p, ax = ax, label="test_cv:{}".format(fold), color="red")
	sns.lineplot(ps, 1-acc_train_p, ax = ax, label = "train_cv:{}".format(fold), color="blue")

ax.set_xlabel("p (complexity)")
ax.set_ylabel("Error")
plt.legend()
sns.despine()
plt.savefig("P3.pdf")




#
# Part e   
#
p_hat = 14000
maxidx = results[:,2] == p_hat
E_test = 1-np.mean(results[maxidx,0])
m = X_test.shape[0]

factor = np.sqrt(np.log(2.0/0.05)/(2*m))
upper = E_test + factor
lower = E_test - factor

print("CI is: {:.04f} < E[E_test] < {:.04f}\tE_test:{:.04f}\tp_hat:{}\tplustminus:{}".format(lower, upper, E_test, p_hat, factor))






