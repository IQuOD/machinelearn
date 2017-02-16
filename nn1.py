######################################################################################################
'''
Attempt to use the tflearn and tensorflow python libraries to teach the machine to identify hit bottom 
profiles. Will also be used to compare output from the hand written neural network

Some functions will be collected from a different script "tf_neuralnet.py" which was used as the
script to run the network prior to the creation of this file
'''

######################################################################################################
# importing libraries

from __future__ import division, print_function, absolute_import
import os.path
import matplotlib.pyplot as plt
from netCDF4.utils import ncinfo
from netCDF4 import Dataset
import numpy as np
import tflearn
import random
import hitbottom as hb
import tf_neuralnet as tf


######################################################################################################
# setting up data for feeding into neural network

# checking to see if model already exists
if (os.path.isfile('net1.index') == True):
	print("\nExisting neural network model found")
	print("loading...\n")
	model.load('net1')

else:
	print("\nneural network model not found")
	print("training new neural network...\n")
	# importing the training set data
	dat1 = []
	with open("nn_training_data_2.txt") as f:
		next(f)
		for line in f:
			dat1.append([x.strip() for x in line.split(",")])

	tr = [1, 0]
	fls = [0, 1]
	y_train = []
	X_train = []
	filename_train = []
	n1 = len(dat1)
	for i in range(0,n1):
		X_train.append([float(dat1[i][1]),float(dat1[i][2]),float(dat1[i][3]),float(dat1[i][4])])	
		filename_train.append(dat1[i][5])
		if (int(dat1[i][0]) == 1):
			y_train.append(tr)
		else:
			y_train.append(fls)

	# importing cross validation data
	dat2 = []
	with open("nn_crossvalidation_data_2.txt") as f:
		next(f)
		for line in f:
			dat2.append([x.strip() for x in line.split(",")])

	y_val = []
	X_val = []
	filename_crossval = []
	n2 = len(dat2)
	for i in range(0,n2):
		X_val.append([float(dat2[i][1]),float(dat2[i][2]),float(dat2[i][3]),float(dat2[i][4])])	
		filename_crossval.append(dat2[i][2])
		if (int(dat2[i][0]) == 1):
			y_val.append(tr)
		else:
			y_val.append(fls)

# importing test data
dat3 = []
with open("nn_test_data_2.txt") as f:
	next(f)
	for line in f:
		dat3.append([x.strip() for x in line.split(",")])

filename_test = []
X_test = []
y_test = []
z_test = []
T_test = []
n3 = len(dat3)
for i in range(0,n3):
	filename_test.append(dat3[i][5])
	X_test.append([float(dat3[i][1]),float(dat3[i][2]),float(dat3[i][3]),float(dat3[i][4])])
	y_test.append(int(dat3[i][0]))
	z_test.append(float(dat3[i][6]))
	T_test.append(float(dat3[i][7]))

# filtering through the data to remove majority of poor points
[X_new, y_new] = tf.reduce_data(X_train,y_train)
X_train = X_new
y_train = y_new
[X_new, y_new] = tf.reduce_data(X_val,y_val)
X_val = X_new
y_val = y_new

# printing out dimensions of the training data
print("Data dimensions:")
print("Input:",len(X_train),len(X_train[0]))
print("Outputs:",len(y_train),len(y_train[0]))
print("\n")

######################################################################################################
# running neural network with tflearnS

# checking to see if model already exists
if (os.path.isfile('net2.index') == True):
	pass

else:
	# testing the tflearn reading data function
	net = tflearn.input_data(shape=[None,8])
	net = tflearn.fully_connected(net, 15, activation='sigmoid')
	net = tflearn.fully_connected(net, 2, activation='softmax')
	net = tflearn.regression(net)

	# creating the model
	model = tflearn.DNN(net)
	model.fit(X_train,y_train, n_epoch=25, validation_set = (X_val, y_val), \
				show_metric=True, run_id="XBT hit bottom data")
	#model.save('net1')

# using model to make predictions on test set
pred = model.predict(X_test)

# collecting key statistics on the data
[precision, files_incorrect] = tf.statistics(pred, z_test, T_test, y_test, filename_test, 1, 0, 1)

# evaluation of the neural network
tf.results(filename_test, files_incorrect, pred, z_test, T_test, False)


######################################################################################################
