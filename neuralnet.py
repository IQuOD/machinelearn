######################################################################################################
# libraries

import numpy as np
import pandas as pd
import hitbottom as hb
import scipy.optimize as op
import math
import os.path
import matplotlib.pyplot as plt
import random
from netCDF4.utils import ncinfo
from netCDF4 import Dataset


######################################################################################################
# attempt to use classes to write the code for the neural network (lots from mnielsen's code)

# code to reduce the poor data
def reduce_data(X,y):
	'''
	Code to take the various poor data points and removing some so that the data that is fed into
	the network is an even number of good and bad points
	'''
	# counting the number of times there are hits
	m = len(y)
	good_count_indices = []
	good_count = 0
	
	# going through the imported data to find the "good" points
	for i in range(0,m):
		if (y[i] == [1,0]):
			good_count += 1
			good_count_indices.append(i)
		else:
			continue

	# writing new lists that store the "filtered" data points
	X_filt = []
	y_filt = []
	n = len(good_count_indices)
	for i in range(0,n-1):
	
		# finding two indices to add (adding two bad data points)
		rand1 = random.random()
		rand2 = random.random()
		rand3 = random.random()
		if (i == 0):
			top = 0
			floor = good_count_indices[i]
		else:
			top = good_count_indices[i]
			floor = good_count_indices[i+1]
		diff = floor-top
		ind1 = int(floor-diff*rand1)
		ind2 = int(floor-diff*rand2)
		ind3 = int(floor-diff*rand3)

		# making sure there are no repeats in the good data
		if (ind1 == good_count_indices[i]):
			if (random.random() > 0.5):
				ind1 = ind1 + 1
			else:
				ind1 = ind1 - 1

		if (ind2 == good_count_indices[i]):
			if (random.random() > 0.5):
				ind2 = ind2 + 1
			else:
				ind2 = ind2 - 1

		if (ind3 == good_count_indices[i]):
			if (random.random() > 0.5):
				ind3 = ind3 + 1
			else:
				ind3 = ind3 - 1
		
		# adding to new array
		X_filt.append(X[good_count_indices[i]])
		X_filt.append(X[ind1])
		X_filt.append(X[ind2])
		X_filt.append(X[ind3])
		y_filt.append(y[good_count_indices[i]])
		y_filt.append(y[ind1])
		y_filt.append(y[ind2])
		y_filt.append(y[ind3])
		
	print("Data reduction (removing bad data)")
	print("before:",len(X))
	print("after:",len(X_filt))
	return(X_filt,y_filt)


# Miscellaneous functions
def feature_scaling(NNinput):
	'''
	Only need to use feature scaling on the depth relative to bathymetry and the number of standard 
	deviations outside the gradient	
	Choosing to use 1000 to scale the relative depth and 200 to scale the value for number std.dev 
	from mean (chosen through educated guess)
	'''
	[HBpoint, dev, fraction, zdiff] = NNinput
	dev = dev/float(200)
	zdiff = zdiff/float(1000)
	scaledNNinput = [HBpoint, dev, fraction, zdiff]

	return(scaledNNinput)

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

class neuralNet:
	'''
	Used to initialise the neural network (with one hidden layer). Note that you should include the
	bais terms in the dimesions input. Variables:
	size - feed in the number of neurons in each layer. Eg. if I want 5 inputs, 7 hidden neurons and
	3 outputs, I would enter [5,7,3] as the input
	'''

	# creating network architecture and giving values to weights and biases
	def __init__(self, size):
		self.numLayers = len(size)
		self.size = size
		self.weights = [np.random.randn(size[1], size[0]+1), np.random.randn(size[2],size[1]+1)]
		self.weightgrads = [np.zeros((size[1], size[0]+1), dtype=float), np.zeros((size[2],size[1]+1), dtype=float)]

	# ------------------------------------------------------------------------
	# ------------------------------------------------------------------------
	# ------------------------------------------------------------------------

	'''
	Re-writing the code above to have one that calculate the cost function and another the other that computes the gradient of the weights
	These should take X, y and theta as the only inputs

	Regularisation is still to be included in the code below
	'''

	# used for backprop algorithm and cost function
	def forwardProp(self,X,y,lambd):

		# input layer (and adding bias)
		a1 = [1]
		a1.extend(X)
		a1 = np.array(a1)
	
		# hidden layer (and adding bias)
		z2 = np.dot(self.weights[0], a1)		
		a2 = [1]
		a2.extend(list(z2))
		z2 = np.array(a2)
		a2 = np.array(a2)
		a2 = sigmoid(a2)

		# output layer 
		z3 = np.dot(self.weights[1],a2)
		a3 = sigmoid(z3)
		
		return([a1, z2, a2, z3, a3])

	
	# back propogation code
	def backProp(self,X,y,lambd):

		'''
		Need to use a gradient checking code to see that the gradients that are being calculated are close to what we expect them to be
		'''

		# running forward propogation to get parameter values
		[a1, z2, a2, z3, a3] = self.forwardProp(X,y,lambd)

		# output layer
		delta3 = a3 - y
	
		# hidden layer
		theta2 = np.transpose(self.weights[1])
		delta2 = np.dot(theta2, delta3)*sigmoid_prime(z2)
		delta2 = np.delete(delta2,[0])

		# derivative ignoring regularisation for the first set of weights
		delta2 = np.asmatrix(delta2)
		a1 = np.asmatrix(a1)
		self.weightgrads[0] += np.dot(np.transpose(delta2),a1)

		# derivatives ignoring regularisation for the second set of weights
		delta3 = np.asmatrix(delta3)
		a2 = np.asmatrix(sigmoid(z2))
		self.weightgrads[1] += np.dot(np.transpose(delta3),a2)

		return(self.weightgrads)

	
	# computing the cost function
	def costFunc(self,X,y,lambd):

		# running forward propogation to get parameter values
		[a1, z2, a2, z3, a3] = self.forwardProp(X,y,lambd)

		# computing the cost from these
		m = len(a3)
		J = -(1/float(m))*(y*np.log(a3)+(1-y)*np.log(1-a3))

		return(J)
	
	
	# using gradient descent to minimise the cost function
	def gradDescent(self, alpha):
		'''		
		alpha is the learning rate for gradient descent (decides how quickly the function
		converges)
		'''
		# doing the first set of weights
		weight1 = self.weights[0] - alpha*self.weightgrads[0]
		weight2 = self.weights[1] - alpha*self.weightgrads[1]
		self.weights = [weight1, weight2]
		return(self.weights)


######################################################################################################
# reading in neural network input parameters

'''
features: HBpoint, grad deviation from mean (#std.dev), fraction above:below, zdiff
y: expected output based on the 
'''

'''

features = []
hb = []
y = []

with open('nn_complete_training.txt') as f:
	next(f)
	for line in f:
		line = line.split(',')
		hb.append(int(line[0]))
		features.append([int(line[1]),float(line[2]),float(line[3]),float(line[4].replace('\n',''))])

# creating the output by returning an array structured as such: 
# y = [true hit bottom point, not hit bottom point]
tr = [1, 0]
fls = [0, 1]

for i in range(0,len(hb)):
	if (hb[i] == 1):
		y.append(tr)
	else:
		y.append(fls)

# reducting the training data (removing more of the non-hit bottom points)
[X_new, y_new] = reduce_data(features,y)
X_train = X_new
y_train = y_new

# creating neural network architecture and defining the number of iterations wanted
net = neuralNet([4,5,2])
lambd = 0
epoch = 20

# running a loop to compute the cost at each epoch
for j in range(0,epoch):

	costs = [] 
	# computing the average cost
	for i in range(0,len(X_train)):
	
		# check for all functions called in NN
		# net.forwardProp(X_train[i],y_train[i][0],lambd)
		# net.backProp(X_train[i],y_train[i][0],lambd)
		# print(net.costFunc(X_train[i],y_train[i][0],lambd))

		# Here I have implemented a neural net using gradient descent (not the most effective
		# algorithm - need to change the data to have an equal number of good and bad points)

		# running backprop algorithm to 
		net.backProp(X_train[i],y_train[i][0],lambd)
		cost_array = net.costFunc(X_train[i],y_train[i][0],lambd)
		costs.extend(cost_array)
		net.gradDescent(0.005)
	
	# printing the cost (average) for each epoch
	av_cost = sum(costs)/float(len(costs))
	print("Average cost: "+str(av_cost))

'''












######################################################################################################

