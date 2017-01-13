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
		self.weightgrads = [np.zeros((size[1], size[0]+1), dtype=float), \
							np.zeros((size[2],size[1]+1), dtype=float)]
	
	# forward propogation to return the outputs of the neural network
	# data to be inputted as a list of the features you want to train the network on
	def forwardprop(self, data):
	
		# input layer (and adding bias)
		a1 = [1]
		a1.extend(data)
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
		
		return(a3, z2, a1)

	# backpropagation method for finding the gradients of the weights
	def backprop(self, output, expect, z2, a1):

		'''
		Need to use a gradient checking code to see that the gradients that are being calculated
		are close to what we expect them to be
		'''

		# output layer
		delta3 = output - expect
	
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

	# calculation of the cost function with those gradient values (unregularised)
	def costFunc(self, data, expect):

		# computing the output from the weights
		a1 = [1]
		a1.extend(data)
		a1 = np.array(a1)
		z2 = np.dot(self.weights[0], a1)		
		a2 = [1]
		a2.extend(list(z2))
		z2 = np.array(a2)
		a2 = np.array(a2)
		a2 = sigmoid(a2)
		z3 = np.dot(self.weights[1],a2)
		output = sigmoid(z3) # this is also a3 or h(theta) 

		# m should be the length of both the output and input
		m = len(output)
		
		# computing the cost
		J = -(1/float(m))*(expect*np.log(output)+(1-expect)*np.log(1-output))
		
		return(J)


######################################################################################################
# reading in neural network input parameters






























######################################################################################################

