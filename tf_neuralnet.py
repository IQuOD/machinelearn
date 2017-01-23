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
import random


######################################################################################################
# functions to help prepare the data

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


# getting statistics for the network (F-Score)
def statistics(predictions, Xtest, ytest, vb):
	'''
	Uses metrics precision and recall, but can also give the true/false detection rates for
	hit bottoms and non-hit bottom points. Note that all three of the inputs should have the
	same length

	The function returns the F score but will also print the other statistics used to measure
	the success of a learning algorithm

	The good (points that have been identified correctly) have a value of 1 in the array identIndex
	and the points that were not identified correctly will have a zero value

	vb is 1 if you want to print or 0 if you don't want to print stats to screen (will always 
	print the F score)
	'''
	print('\n')
	print("Generating statistics...")
	m = len(predictions)
	identIndex = np.zeros(m)

	# true positive detection rate
	trueHB = 0
	detectHB = 0
	for i in range(0,m):
		# checking if the result is truly a "good" HB point
		if (ytest[i] == 1):
			trueHB += 1
			# counting the number of points identified out of these
			if (predictions[i][0] > predictions[i][1]):
				detectHB += 1
				identIndex[i] = 1
			else:
				continue
		else:
			continue
	true_pos = float(detectHB)/float(trueHB)
	if (vb == 1):
		print("True positive detection rate: "+str(true_pos))

	# true negative detection rate
	trueNeg = 0
	detectNeg = 0
	for i in range(0,m):
		# checking all points that are not hit bottoms
		if (ytest[i] == 0):
			trueNeg += 1
			# counting the number of these points that are identified as not HB points
			if (predictions[i][1] > predictions[i][0]):
				detectNeg += 1
				identIndex[i] = 1
			else:
				continue
		else:
			continue
	true_neg = float(detectNeg)/float(trueNeg)
	if (vb == 1):
		print("True negative detection rate: "+str(true_neg))

	# false positive rate
	falsePos = 0
	detectFalseNotHB = 0 
	for i in range(0,m):
		# checking all points that are not HB points
		if (ytest[i] == 0):
			falsePos += 1
			# taking fraction that are actually detected as HB
			if (predictions[i][0] > predictions[i][1]):
				detectFalseNotHB += 1
			else:
				continue
		else:
			continue
	false_pos = float(detectFalseNotHB)/float(falsePos)
	if (vb == 1):
		print("False positive detection rate: "+str(false_pos))

	# false negative rate
	falseNeg = 0
	detectFalseHB = 0
	for i in range(0,m):
		# checking all of the points that are HB
		if (ytest[i] == 1):
			falseNeg += 1
			# taking fraction of these that are not detected as HB
			if (predictions[i][1] > predictions[i][0]):
				detectFalseHB += 1
			else:
				continue
		else:
			continue
	false_neg = float(detectFalseHB)/float(falseNeg)
	if (vb == 1):
		print("False negative detection rate: "+str(false_neg))

	# pulling out other statistics: precision, recall and F-Score
	precision = float(detectHB)/float(detectHB+detectFalseNotHB)
	recall = float(detectHB)/float(detectFalseHB+detectHB)
	Fscore = float(2*precision*recall)/float(precision+recall)

	# printing to screen the results of this statistics call
	if (vb == 1):
		print("Precison: "+str(precision))
		print("Recall: "+str(recall))
	print("F-score: "+str(Fscore))	
	print("\n")

	return(Fscore, identIndex)


# function to print the remaining 
def files_remaining(namefile_test, identIndex):
	'''
	This function takes the results of the neural network and writes a new file which has the
	filenames of the profiles that were not correctly identified (false positive or negative profiles)
	This is used so that these can be plotted for re-examination
	'''
	print("Writing file of incorrectly identified profile names...")
	n = len(namefile_test)
	filename_array = []
	f = open('nn_incorrect_classification.txt','w+')

	# looping through each example
	for i in range(0,n):
		name = namefile_test[i]
		count = 0

		# checking that there are no repeats
		if (len(filename_array) != 0):
			for j in range(0,len(filename_array)):
				if (filename_array[j] == name):
					count += 1
				else:
					continue
			# adding to the file only if there are no repeats
			if (count == 0):
				if (identIndex[i] == 0):
					f.write(name+"\n")
					filename_array.append(name)
				else:
					continue
			else:
				continue

		# for the first point
		else:
			if (identIndex[i] == 0):
				f.write(name+"\n")
				filename_array.append(name)
			else:
				continue
	

######################################################################################################
# setting up data for feeding into neural network

# checking to see if model already exists
if (os.path.isfile('xbt_nn1.tfl.index') == True):
	model.load('xbt_nn1.tfl')

else:
	# importing the training set data
	dat1 = []
	with open("nn_complete_training.txt") as f:
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
	with open("nn_crossvalidation_data.txt") as f:
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
with open("nn_test_data.txt") as f:
	next(f)
	for line in f:
		dat3.append([x.strip() for x in line.split(",")])

filename_test = []
X_test = []
y_test = []
n3 = len(dat3)
for i in range(0,n3):
	filename_test.append(dat3[i][5])
	X_test.append([float(dat3[i][1]),float(dat3[i][2]),float(dat3[i][3]),float(dat3[i][4])])
	y_test.append(int(dat3[i][0]))

# filtering through the data to remove majority of poor points
[X_new, y_new] = reduce_data(X_train,y_train)
X_train = X_new
y_train = y_new

# printing out dimensions of the training data
print("\n")
print("Data dimensions:")
print("Input:",len(X_train),len(X_train[0]))
print("Outputs:",len(y_train),len(y_train[0]))
print("\n")

######################################################################################################
# running neural network with tflearnS

# checking to see if model already exists
if (os.path.isfile('xbt_nn1.tfl.index') == True):
	pass

else:
	# testing the tflearn reading data function
	net = tflearn.input_data(shape=[None,4])
	net = tflearn.fully_connected(net, 5, activation='sigmoid')
	net = tflearn.fully_connected(net, 10)
	net = tflearn.fully_connected(net, 2, activation='sigmoid')
	net = tflearn.regression(net)

	# creating the model
	model = tflearn.DNN(net)
	model.fit(X_train,y_train, n_epoch=20, validation_set = (X_val, y_val), \
				show_metric=True, run_id="XBT hit bottom data")

# using model to make predictions on test set
pred = model.predict(X_test)

# collecting key statistics on the data
[Fscore, identIndex] = statistics(pred, X_test, y_test, 1)
	
# function to remove the test data correctly identified by the neural network
files_remaining(filename_test, identIndex)


######################################################################################################
