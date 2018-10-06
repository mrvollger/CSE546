#!/usr/bin/env python
import numpy as np
import scipy
import scipy.linalg as la
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("ticks")
np.random.seed(0)

def makeGuassian(n, u, s):
	Z = np.random.randn(2,n)
	X = np.transpose(la.sqrtm(s).dot(Z))
	X += u
	return(X)

idxs = [0,1,2]
Us = []
Ss = []
Xs = []
Us.append( np.transpose(np.array([1,2])))
Ss.append(np.array([[1,0], [0,2]]))
Us.append(np.transpose(np.array([-1,1])))
Ss.append(np.array([[2,-1.8], [-1.8,2]]))
Us.append(np.transpose(np.array([2,-2])))
Ss.append(np.array([[3,1], [1,2]]))

fig, axs = plt.subplots(ncols=3, sharey=True, figsize=(15,5))



for idx, u, s, ax in zip(idxs, Us, Ss, axs):
	
	### Part A ###
	x = makeGuassian(100, u, s)
	Xs.append(x)
	lim = x.max() + 1 
	sns.scatterplot(x[:,0], x[:,1], ax=ax, marker = "^"	)

	
	### Part B ###
	su = np.mean(x, axis=0)
	ss = np.cov(np.transpose(x))
	#print(ss, su)
	evals, evects = la.eig(ss)
	#print(evects)
	mag = np.diag(np.real(evals)).dot(np.transpose(evects))
	print(mag)
	ax.arrow(su[0], su[1], mag[0,0], mag[0,1], width = .15, color='red')
	ax.arrow(su[0], su[1], mag[1,0], mag[1,1], width = .15, color='red')



	# formatting 
	ax.set_title("u_{0}, s_{0}".format(idx+1) )
	ax.set_xlabel("X_{}: with u={}".format(idx+1, u[0]) )
	ax.set_ylabel("X_{}: with u={}".format(idx+1, u[1]) )
	ax.set(ylim=(-lim, lim), xlim=(-lim,lim))



# remove top bars
sns.despine()
plt.savefig("P1.pdf")


