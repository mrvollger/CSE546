#!/usr/bin/env python
import argparse
import sys

parser = argparse.ArgumentParser(description="")
parser.add_argument("infile", nargs="?", help="input data file",  type=argparse.FileType('r'), default="/net/eichler/vol21/projects/bac_assembly/nobackups/genomeWide/CHM13_V2/LocalAssemblies/PlaceOnt/LHR.tsv")
parser.add_argument("truth", nargs="?", help="input data file",  type=argparse.FileType('r'), default="/net/eichler/vol21/projects/bac_assembly/nobackups/genomeWide/CHM13_V2/LocalAssemblies/PlaceOnt/intersect.bed")
parser.add_argument("-X", help="output data",  default="data/data.npy")
parser.add_argument("-Y", help="output labels", default="data/labels.npy")
parser.add_argument("--dlabels",  help="output labels", type=argparse.FileType('w'), default="data/dlabels.txt")
parser.add_argument('-d', action="store_true", default=False)
args = parser.parse_args()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
colnames = ["correct", "total", "frac", "LHR", "rname", "qname", "dove", "match", "mismatch", "insertion","deletion", "indelsOverThree", "largestIndel", "perID", "rlen", "qlen"]

def readData():
	X = pd.read_table(args.infile, header = None)
	X.columns = colnames
	# remove infinities form LHR 
	mmax = 325.0
	X.loc[X.LHR > mmax, "LHR"] = mmax
	X.loc[X.LHR < -mmax, "LHR"] = -mmax
	#print(X.shape, X.LHR.describe())
	return(X)


def GetLabels(X):
	Y = []
	true_df = pd.read_table(args.truth, header = None)
	rname = true_df[9]
	qname = true_df[3]
	rtoq_true = set(rname + "__" + qname)
	rtoq_data = list( X.rname + "__" + X.qname)
	for pair in rtoq_data:
		if(pair in rtoq_true):
			Y.append(1)
		else:
			Y.append(0)

	# I can now drop X.rname and X.qname and make it only data
	X.drop(["rname", "qname"], axis=1, inplace=True)
	# de mean data 
	X = X.sub(X.mean())
	return(Y)

def mywrite(X, Y):
	mat = X.values
	labels = np.array(Y)
	labels = labels.reshape((len(labels), 1))
	np.save(args.X, mat)
	np.save(args.Y, labels)
	for item in colnames:
		args.dlabels.write(item+"\t")
	args.dlabels.write("\n")
	#print(mat)
	#print(labels)

print("Reading Data")
X = readData()
Y = GetLabels(X)
print("Writing data")
mywrite(X,Y)


