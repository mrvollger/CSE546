#!/usr/bin/env python
import cvxpy as cp
import numpy as np
import scipy
import scipy.linalg as la
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("ticks")
np.random.seed(0)
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
	
	K = K + 1e-14 * np.eye(n)	
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

def Loss(f, y, train):
	#if( train == trainB ):
	if(False):
		z = np.abs(f-y)
		tmp = np.zeros(z.shape)
		g1 = (z >= 1)
		tmp[g1] = 2*z[g1] - 1
		tmp[~g1] = z[~g1]**2
		mse = tmp.mean()
	else:
		mse = (f-y)**2
		mse = np.mean( mse )
	#print(mse)
	return(mse)


def plots(X, y, train, hyper, L, L2=None):
	K = makeKernel(X, hyper = hyper)
	D = makeD(K.shape[0])
	alpha = train(K, y, D, L=L, L2=L2)
	f = predict(K, alpha)
	mse = Loss(f, y, train)
	n = X.shape[0]

	name = "{}_{}.pdf".format(n, train.__name__)

	# redefine fx to use more points 
	x = np.arange(0,1,0.001)
	fx = Fx(x)

	sns.scatterplot(X[:,0], y[:,0], color="green", label="y_i")
	sns.lineplot(x, fx, label="f(x)")
	print(X.shape, f.shape)
	sns.lineplot(X[:,0], f[:,0], label="f_hat")
	plt.title("{}, gamma={}, L={}, L2={}".format(train.__name__, hyper, L, L2))
	plt.legend()	
	plt.xlabel("x")
	plt.ylabel("f(x)")
	#plt.ylim(-4.5, 6.5)	

	plt.savefig(name)
	
	plt.clf()
	print(name, mse)

def trainOrig(K, y, D, L=10**-6, L2=10**-6):
	a = K + L * np.identity(K.shape[0]) 
	alpha = la.solve( a, y )
	return(alpha)

def trainA(K, y, D, L=10**-6, L2=10**-6):
	# make sure K is non negative 
	n = K.shape[0]

	# fix y dim for cvxpy 
	y = y[:,0]
	
	alpha = cp.Variable(n)
	loss = cp.pnorm(cp.matmul(K, alpha) - y, p=2)**2
	reg = L*cp.quad_form(alpha, K)

	obj = loss + reg
	problem = cp.Problem(cp.Minimize(obj))
	try:
		problem.solve()
	except cp.SolverError:
		problem.solve(solver=cp.SCS)
	a = alpha.value

	#print(reg.value)
	#print("a", a)
	if(a is None):
		return(np.zeros((n,1)))

	return(a.reshape((n,1)))


def trainB(K, y, D, L=10**-6, L2=10**-6,):
	# make sure K is non negative 
	n = K.shape[0]

	# fix y dim for cvxpy 
	y = y[:,0]
	
	alpha = cp.Variable(n)
	loss = cp.sum( cp.huber(cp.matmul(K, alpha) -y, 1) )
	reg = L* cp.quad_form(alpha, K)

	obj = loss + reg
	problem = cp.Problem(cp.Minimize(obj))
	try:
		problem.solve()
	except cp.SolverError:
		problem.solve(solver=cp.SCS)
	a = alpha.value
	#print("a", a)
	if(a is None):
		return(np.zeros((n,1)))

	return(a.reshape((n,1)))

def trainC(K, y, D, L=10**-6, L2=10**-6,):
	# make sure K is non negative 
	n = K.shape[0]
	DK = D.dot(K)
	#print(DK.shape)

	# fix y dim for cvxpy 
	y = y[:,0]
	
	alpha = cp.Variable(n)
	#loss = cp.sum( cp.huber(cp.matmul(K, alpha) -y, 1) )
	loss = cp.pnorm(cp.matmul(K, alpha) - y, p=2)**2
	reg1 = L * cp.sum( cp.pnorm(cp.matmul(DK, alpha), p=1) )
	reg2 = L2 * cp.quad_form(alpha, K)

	obj = loss + reg1 + reg2
	problem = cp.Problem(cp.Minimize(obj))
	try:
		problem.solve()
	except cp.SolverError:
		problem.solve(solver=cp.SCS)
	a = alpha.value
	#print("a", a)
	if(a is None):
		return(np.zeros((n,1)))

	return(a.reshape((n,1)))


def trainD(K, y, D, L=10**-6, L2=10**-6):
	# make sure K is non negative 
	n = K.shape[0]
	DK = D.dot(K)

	# fix y dim for cvxpy 
	y = y[:,0]
	
	alpha = cp.Variable(n)
	loss = cp.pnorm(cp.matmul(K, alpha) - y, p=2)**2
	reg = L*cp.quad_form(alpha, K)
	cons = [ cp.matmul(DK, alpha) >= 0 ]

	obj = loss + reg
	problem = cp.Problem(cp.Minimize(obj), cons)
	try:
		problem.solve()
	except cp.SolverError:
		problem.solve(solver=cp.SCS)
	a = alpha.value

	#print(reg.value)
	#print("a", a)
	if(a is None):
		return(np.zeros((n,1)))

	return(a.reshape((n,1)))

def DoCV(X, y, train, k=10):  
	hypers = np.logspace(-1, 3, num = 20)
	Ls = np.logspace( -5, 3, num = 10)
	L2s = [0]
	if(train == trainC):
		L2s = Ls
		Ls = np.logspace(-5, 3, num = 10)

	results = []
	for hyper in hypers:	
		K = makeKernel(X, hyper = hyper)
		# leave one out cross validation
		K_trains, y_trains, K_vals, y_vals =  kfold(K, y, k = k)
		D = makeD(K_trains[0].shape[1])
		for L in Ls:
			for L2 in L2s:
				mses = []
				for i in range(k):
					K_train = K_trains[i]; y_train = y_trains[i]; K_val = K_vals[i]; y_val = y_vals[i]
					alpha = train(K_train, y_train, D, L=L, L2=L2)
					f = predict(K_val, alpha)
					mse = Loss(f, y_val, train)
					mses.append(mse)
				results.append( ( np.mean(mses), hyper, L, L2 ) )
				print("hyper", hyper, "lambda", L, "lambda2", L2, "mse", np.mean(mses))

	best = np.inf; bestidx = 0
	for idx, line in enumerate(results):
		if(line[0] < best):
			best = line[0]
			bestidx = idx
	mse, hyper, L, L2 = results[bestidx]
	print(mse, hyper, L, L2)
	return(hyper, L, L2)



X, y, fx  = makedata()
#tmpmse=10.32008903133354; gamma=1.0; L= 0.03162277660168379
#gamma=100.0; L=0.0001

a=False
b=False
c=False
d=True

if(a): # part a, 
	gamma, L, L2 = DoCV(X, y, trainOrig, k=int(X.shape[0]/1))
	plots(X, y, trainOrig, gamma, L, L2=None)
	
	gamma, L, L2 = DoCV(X, y, trainA, k=int(X.shape[0]/1))
	plots(X, y, trainA, gamma, L)

if(b): # part b
	gamma, L, L2 = DoCV(X, y, trainB, k=X.shape[0])
	plots(X, y, trainB, gamma, L)

if(c):
	#gamma, L, L2 = DoCV(X, y, trainC, k=X.shape[0])
	#gamma, L , L2 = (129.15496650148827, 0.001, 0.0031622776601683794)
	gamma, L, L2 = 379.0, 16.68, 0.0046
	plots(X, y, trainC, gamma, L, L2=L2)

if(d):
	#gamma, L, L2 = DoCV(X, y, trainD, k=X.shape[0])
	gamma, L=379.2690190732246, 0.03593813663804626
	plots(X, y, trainD, gamma, L)


