#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import scipy
import scipy.linalg as la
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
#from mnist import MNIST
import pickle
import os
# getting py torch
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim


sns.set()
sns.set_style("ticks")
np.random.seed(1)
print("modules loaded")



batch_size = 4
num_workers = 2
in_channels = 3
out_channels = 6
kernal_d = 5

def load_data():
	transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ] )

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	
	rtn = (transform, trainset, trainloader, testset, testloader, classes)
	print("Data load done")
	return(rtn)

def imshow(img, name = "tmp.pdf"):
	img = (img/2 + 0.5)
	npimg = img.numpy()
	fig = plt.figure()
	plt.imshow(np.transpose(npimg, (1,2,0)))
	plt.savefig(name)
	print("post figure")

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		# 1 input image channel, 6 output channels, 5x5 square convolution
		# kernel
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernal_d)
		self.pool = nn.MaxPool2d(2,2)
		self.conv2 = nn.Conv2d(out_channels, 16, kernal_d)
		# an affine operation: y = Wx + b
		self.fc1 = nn.Linear(16 * kernal_d * kernal_d, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool( F.relu(self.conv1(x)) )
		x = self.pool( F.relu(self.conv2(x)) )
		
		# should be 16 * 5 * 5		
		flat_feats = self.num_flat_features(x)
		print(flat_feats)
		x = x.view(-1, flat_feats)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return(x)

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features



transform, trainset, trainloader, testset, testloader, classes = load_data()




# 
# test data loading
#
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
print("iter")

# show images
imshow(torchvision.utils.make_grid(images))
print("plot made")

# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))



criterion = nn.CrossEntropyLoss()

