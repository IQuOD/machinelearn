######################################################################################################
'''
Attempt to use the tflearn and tensorflow python libraries to teach the machine to identify hit bottom 
profiles. Will also be used to compare output from the hand written neural network
'''

######################################################################################################
# importing libraries

from __future__ import division, print_function, absolute_import
import os.path
import numpy as np
import tflearn


######################################################################################################
# setting up data for feeding into neural network

# importing the training set data
dat1 = []
with open("nn_training_data.txt") as f:
	next(f)
	for line in f:
	    dat1.append([float(x.strip()) for x in line.split(",")])

tr = [1, 0]
fls = [0, 1]
y_train = []
X_train = []
n1 = len(dat1)
for i in range(0,n1):
	X_train.append([dat1[i][1],dat1[i][2],dat1[i][3],dat1[i][4]])	
	if (dat1[i][0] == 1):
		y_train.append(tr)
	else:
		y_train.append(fls)

# importing cross validation data
dat2 = []
with open("nn_crossvalidation_data.txt") as f:
	next(f)
	for line in f:
	    dat2.append([float(x.strip()) for x in line.split(",")])

y_val = []
X_val = []
n2 = len(dat2)
for i in range(0,n2):
	X_val.append([dat2[i][1],dat2[i][2],dat2[i][3],dat2[i][4]])	
	if (dat2[i][0] == 1):
		y_val.append(tr)
	else:
		y_val.append(fls)

# importing test data
dat3 = []
with open("nn_test_data.txt") as f:
	next(f)
	for line in f:
	    dat3.append([float(x.strip()) for x in line.split(",")])

X_test = []
y_test = []
n3 = len(dat3)
for i in range(0,n3):
	X_test.append([dat3[i][1],dat3[i][2],dat3[i][3],dat3[i][4]])
	y_test.append(dat3[i][0])

# printing out dimensions of the training data
print("\n")
print("Data dimensions:")
print("Input:",len(X_train),len(X_train[0]))
print("Outputs:",len(y_train),len(y_train[0]))
print("\n")

######################################################################################################
# running neural network with tflearn

# testing the tflearn reading data function
net = tflearn.input_data(shape=[None,4])
net = tflearn.fully_connected(net, 5, activation='sigmoid')
net = tflearn.fully_connected(net, 2, activation='sigmoid')
net = tflearn.regression(net)

# creating the model
model = tflearn.DNN(net)
model.fit(X_train,y_train, n_epoch=20, validation_set = (X_val, y_val), \
			show_metric=True, run_id="XBT hit bottom data")

# using model to predict
pred = model.predict(X_test)
m = len(pred)
count = 0
total = 0

# collecting basic statistics
for i in range(0,m):
	
	# counting the number of hit bottom points
	if (y_test[i] == 1):
		total += 1

	# counting the number detected from code
	if (pred[i][0] > pred[i][1]):
		print(pred[i])
		count += 1
	else:
		continue

print("True detection rate:", float(count)/float(total))


######################################################################################################
