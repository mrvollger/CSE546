#!/usr/bin/env python

import numpy as np 
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def Yk(n, k):
	yk = np.sum(np.sign(np.random.randn(n,k)) * np.sqrt(1./k), axis=1)
	return(yk)


n=100000
assert np.sqrt(1/(4*n)) < 0.0025
Z=np.random.randn(n)


fig = plt.figure()

plt.step(sorted(Z), np.arange(1,n+1)/float(n) , label = 'Gaussian')


ks = [1,8,64,512]
yks = []
for k in ks:
	yk = Yk(n,k)
	yks.append(yk)
	plt.step(sorted(yk), np.arange(1,n+1)/float(n), label = 'k={}'.format(k))	

plt.legend(ncol=2, loc='upper left')
plt.ylabel('Cumulative probability', fontsize=12)
plt.xlabel('Observations', fontsize=12)
plt.xlim(-3,3)

fig.savefig('problem9.png')



