#!/usr/bin/env python
from __future__ import print_function
import numpy as np
#import scipy
#import scipy.linalg as la
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
np.random.seed(2)
print("modules loaded")



batch_size = 4
num_workers = 0
in_channels = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print(device)


def load_data():
	transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ] )

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	
	rtn = (transform, trainset, trainloader, testset, testloader, classes)
	#print("Data load done")
	return(rtn)

def imshow(img, name = "tmp.pdf"):
	img = (img/2 + 0.5)
	npimg = img.cpu().numpy()
	fig = plt.figure()
	plt.imshow(np.transpose(npimg, (1,2,0)))
	plt.savefig(name)
	print("post figure")

class Net(nn.Module):
	def __init__(self, part, out_channels, kernal_d, N):
		super(Net, self).__init__()
		self.part=part
		# 3 input image channel, 6 output channels, 5x5 square convolution
		# kernel
		if(self.part == "a"):
			self.a_fc = nn.Linear(32*32*in_channels, 10)
		elif(self.part == "b"):
			self.a_fc = nn.Linear(32*32*in_channels, out_channels)
			self.h1_fc = nn.Linear(out_channels, 10)
		elif(self.part == "c"):
			self.convC = nn.Conv2d(in_channels, out_channels, kernal_d)
			self.poolC = nn.MaxPool2d(N,N)

		else:
			self.conv1 = nn.Conv2d(in_channels, out_channels, kernal_d)
			self.pool = nn.MaxPool2d(2,2)
			self.conv2 = nn.Conv2d(out_channels, 16, kernal_d)
			# an affine operation: y = Wx + b
			self.fc1 = nn.Linear(16 * kernal_d * kernal_d, 120)
			self.fc2 = nn.Linear(120, 84)
			self.fc3 = nn.Linear(84, 10)


	def forward(self, x):
		# set up the input vecotr depending on part
		if(self.part in ["a", "b"]):
			x = x # no manipulation
		else:
			x = self.pool( F.relu(self.conv1(x)) )
			x = self.pool( F.relu(self.conv2(x)) )
		
		# should be 16 * 5 * 5 in the tutorial, and that is ture/works
		flat_feats = self.num_flat_features(x)
		#print(flat_feats)
		x = x.view(-1, flat_feats)

		if(self.part == "a"):
			x = self.a_fc(x)
		elif(self.part == "b"):
			x = F.relu( self.a_fc(x))
			x = self.h1_fc(x)
		else:
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



def testNet(net, testloader):
	total = 0
	correct = 0
	with torch.no_grad():
		for data in testloader:
			inputs, labels = data
			inputs, labels = inputs.to(device), labels.to(device)
			outputs = net(inputs)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
	return( 100.0*correct/ total )

def generateRandomHypers(n=1):
	# set ranges
	lr_range = [-4, -2] # log
	momentum_range = [.1, .95]
	out_channels_range = [1, 3] # log
	kernal_d_range = [2, 100]

	# generate randoms 
	lrtmp = np.logspace(lr_range[0], lr_range[1], num = 100 )
	lr = np.random.choice(lrtmp, size = n)

	momentum = np.random.uniform(low=momentum_range[0], high=momentum_range[1], size = n)	
	
	outtmp = np.logspace(out_channels_range[0], out_channels_range[1], num = 10 )
	out_channels = np.random.choice(outtmp, size = n).astype(int)


	kernal_d = np.random.randint(low=kernal_d_range[0], high=kernal_d_range[1], size = n)	

	return(lr, momentum, out_channels, kernal_d)


transform, trainset, trainloader, testset, testloader, classes = load_data()


def train(net, optimizer, criterion, acc = False, epochs=2):
	loss_l = []; testAcc = []; trainAcc=[]
	for epoch in range(epochs):  # loop over the dataset multiple times
		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# get the inputs
			inputs, labels = data
			inputs, labels = inputs.to(device), labels.to(device)
			
			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# add to loss list, calcualte accuracy if nessisary. 
			loss_l.append(loss.item())
			if(acc and ( i%1000 == 0 ) ):
				testAcc.append(testNet(net, testloader))
				trainAcc.append(testNet(net, trainloader))

			# print statistics
			running_loss += loss.item()
			psize = 4000
			if i % psize == (psize - 1):    # print every psize mini-batches
				ploss = running_loss/psize 
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, ploss) )
				running_loss = 0.0
				if(ploss > 2.5): # stop training if this is a bad initalization
					return(loss_l)
	if(acc):
		return(loss_l, trainAcc, testAcc)
	else:
		return(loss_l)

def loadNet(PATH, part): 
	#model = Net(part, 100, 100)
	#optimizer = optim.SGD(model.parameters(), lr=.01, momentum=.5)
	checkpoint = torch.load(PATH)
	#model.load_state_dict(checkpoint['model_state_dict'])
	#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	#acc = checkpoint['acc']
	#loss_l = checkpoint['loss_l']
	return(checkpoint)

def makeNets(part="a", epochs=2):
	global device
	if(part in ["a"]):
		device="cpu"

	n = 20
	lr_l, momentum_l, out_channels_l, kernal_d_l = generateRandomHypers(n)
	
	# check to see if we have already run this part, and make sure nto to overwrite a better result
	bestNetPath = part + ".best.net"
	if os.path.exists(bestNetPath):
		checkpoint = loadNet(bestNetPath, part)
		bestAcc=checkpoint['acc']
	else:
		bestAcc = 0
	print("Best accuracy so far:", bestAcc)

	for i in range(n):
		lr, momentum, out_channels, kernal_d = lr_l[i], momentum_l[i], out_channels_l[i], kernal_d_l[i]
		print(lr, momentum, out_channels, kernal_d)

		net = Net(part, out_channels, kernal_d).to(device)
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
		loss_l = train(net, optimizer, criterion, epochs=epochs)
		acc = testNet(net, testloader)

		if(acc >= bestAcc):
			bestAcc = acc
			torch.save({
				'params': (lr, momentum, out_channels, kernal_d),
				'model_state_dict': net.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'loss_l': loss_l, 
				'acc': acc },
				 bestNetPath)
			print("Best model updated for part " + part)


def plotNet(part = "a", epochs = 2):
	global device
	if(part in ["a"]):
		device="cpu"

	out = part + ".pdf"
	bestNetPath = part + ".best.net"
	checkpoint = loadNet(bestNetPath, part)
	#lr, momentum, out_channels, kernal_d = checkpoint["params"]
	params = checkpoint["params"]
	print(params)
	print(checkpoint["acc"])

	net = Net(part, *params[2:] ).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=params[0], momentum=params[1] )
	loss_l, trainAcc, testAcc = train(net, optimizer, criterion, acc = True, epochs = epochs)
	iters = np.arange(len(trainAcc))*1000
	print(len(iters), len(trainAcc), len(testAcc))

	plt.figure()
	sns.lineplot(iters, testAcc, label="test")
	sns.lineplot(iters, trainAcc, label="train")
	plt.title(str(params))
	plt.ylabel("Accuracy")
	plt.xlabel("Iteration")
	plt.savefig(out)


#makeNets(part="a")
#plotNet(part="a")

#makeNets(part="b", epochs=16)
#plotNet(part="b", epochs=16)

makeNets(part="", epochs=16)
plotNet(part="c", epochs=16)

