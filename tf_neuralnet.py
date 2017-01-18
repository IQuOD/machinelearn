######################################################################################################
'''
Attempt to use the tflearn and tensorflow python libraries to teach the machine to identify hit bottom 
profiles. Will also be used to compare output from the hand written neural network
'''

######################################################################################################
# importing libraries

import numpy as np
import tflearn


######################################################################################################
# functions used for preparing or runnning the neural network

# function to process data so that it can be read in directly by tflearn functions
def preprocess(data):
	pass

######################################################################################################
# code to be run

# importing the data
dat1 = []
with open("nn_training_data.txt") as f:
    next(f)
    for line in f:
        dat1.append([float(x.strip()) for x in line.split(",")])

# storing the training data (output can be true hit bottom y[0]=1, or not y[0]=0)
tr = [1, 0]
fls = [0, 1]
y = []
X = []
n1 = len(dat1)
for i in range(0,n1):
	X.append([dat1[i][1],dat1[i][2],dat1[i][3],dat1[i][4]])	
	if (dat1[i][0] == 1):
		y.append(tr)
	else:
		y.append(fls)

# testing the tflearn reading data function
net = tflearn.input_data(shape=[None,4])
net = tflearn.fully_connected(net, 10, regularizer='L2')
net = tflearn.fully_connected(net, 5, activation='sigmoid')
net = tflearn.regression(net)

model = tflearn.DNN(net)



######################################################################################################