######################################################################################################
# libraries

import numpy as np
import pandas as pd
import hitbottom as hb
import scipy.optimize as op
import math
import os.path
import matplotlib.pyplot as plt
from netCDF4.utils import ncinfo
from netCDF4 import Dataset


######################################################################################################
# attempt to use classes to write the code for the neural network (lots from mnielsen's code)

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
	

######################################################################################################
# reading in neural network input parameters

'''
features: HBpoint, grad deviation from mean (#std.dev), fraction above:below, zdiff
y: expected output based on the 
'''

features = []
y = []

with open('nn_training_data.txt') as f:
	next(f)
	for line in f:
		line = line.split(',')
		y.append(int(line[0]))
		features.append([int(line[1]),float(line[2]),float(line[3]),float(line[4].replace('\n',''))])
	
features = np.array(features)
y = np.array(y)

# creating neural network architecture
net = neuralNet([4,5,1])
lambd = 0

for i in range(0,len(features)):
	#print(net.backProp(features[i],y[i],lambd))
	#print(net.costFunc(features[i],y[i],lambd))
	#theta = op.fmin_cg(net.costFunc, fprime=net.backProp, x0=net.weights, args=(features[i],y[i],lambd), maxiter=50)
	#print(theta)
	#res = op.minimize(fun=net.costFunc, x0=net.weights, args=(features[i],y[i],lambd), method='TNC')

















######################################################################################################

