######################################################################################################
'''
Attempt to use the tflearn and tensorflow python libraries to teach the machine to identify hit bottom 
profiles. Will also be used to compare output from the hand written neural network
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


# getting statistics for the network (detection fraction)
def statistics(predictions, ztest, Ttest, ytest, filenames, vb, tf, filewrite):
	'''
	Uses metrics precision and recall, but can also give the true/false detection rates for
	hit bottoms and non-hit bottom points. Note that all three of the data inputs and the 
	namefile array should have the same length
	
	The test for a true positive will be to look to see how many times the highest probability HB 
	point is detecting the point closest to the true hit bottom location as scientifically QC'ed. 

	The function returns the F score but will also print the other statistics used to measure
	the success of a learning algorithm

	The good (points that have been identified correctly) have a value of 1 in the array identIndex
	and the points that were not identified correctly will have a zero value

	vb is 1 if you want to print or 0 if you don't want to print stats to screen
	
	tf - parameter if you want to print the file names of those profiles that were identified 
		 correctly to the screen
	
	file - parameter to indicate whether or not you want to write the undetected profiles to a file
	'''
	print('\n')
	print("Generating statistics...")
	m = len(predictions)

	# filtering through the name array to identify unique profiles
	unique_name = remove_repeats(filenames)
	n = len(unique_name)
	files_incorrect = []
	
	# threshold is the max depth difference between scientific QC and NN
	threshold = 10

	# creating variables for the stats
	truepos = 0
	trueneg = 0
	falsepos = 0
	falseneg = 0
	
	# looping through each profile to find the point of highest probability
	for i in range(0,n):
		# initialisation of parameters used in loop
		name = unique_name[i]
		HBindex = 0
		HBprob = 0
		HBz = 0
		QCz = 0

		# finding QC depth
		for j in range(0,m):
			if (filenames[j] == name):
				if (ytest[j] == 1):
					QCz = ztest[j]
				else:
					continue
			else:
				continue

		# initial loop to find the index of highest HB probability
		for j in range(0,m):
			if (filenames[j] == name):
				# finding the index of the point with the highest probability
				if (predictions[j][0] > HBprob):
					HBindex = j
					HBz = ztest[j]
					HBprob = predictions[j][0]
				else:
					continue
			else:
				continue

		# second loop to count statistics
		for j in range(0,m):
			if (filenames[j] == name):
				# looking at points that are not the hit bottom
				if (j != HBindex):
					# counting true negatives
					if (ytest[j] == 0):
						trueneg += 1
					# counting false negatives 
					else:
						falseneg += 1

				# looking at the detected hit bottom point
				else:
					# counting true positives (if the difference in depths is less than a threshold)
					if (abs(QCz - HBz) < threshold):
						truepos += 1
						# printing as a test if tf = True is met
						if (tf == True):
							print(name)
							print(HBz,QCz)
							print("\n")
					# counting false positives
					else:
						falsepos += 1
			else:
				continue
	
		# recording if the depth of the detected HB is within a threshold of the scientific QC
		if (abs(QCz-HBz)>threshold):
			files_incorrect.append(name)
		else:
			continue
	
	# pulling out statistics
	trueposRate = truepos/float(m)
	truenegRate = trueneg/float(m)
	falseposRate = falsepos/float(m)
	falsenegRate = falseneg/float(m)
	precision = trueposRate/float(trueposRate+falsenegRate)
	recall = trueposRate/float(truenegRate+trueposRate)
	Fscore = float(2*precision*recall)/float(precision+recall)

	# printing out statistics to terminal
	if (vb == True):
		print("total number of points: "+str(m))
		print("true positives: "+str(truepos))
		print("true negatives: "+str(trueneg))
		print("false positives: "+str(falsepos))
		print("false negatives: "+str(falseneg))
		print("precision: "+str(precision))
		print("recall: "+str(recall))
		print("F-score: "+str(Fscore))
		print("Statistics generated\n")
	else:
		print("Statistics generated (not printed)\n")

	# writing file if filewrite == True
	if (filewrite == True):
		f = open('nn_filecheck.txt','w')
		for i in range(0,len(files_incorrect)):
			f.write(files_incorrect[i]+"\n")
		f.close
	else:
		pass

	# returning key metric and list of the profiles that have been identified correctly
	return(precision, files_incorrect)


# function to take an array and remove the repeated elements from that array
def remove_repeats(array):

	# initialise output
	output_array = []
	n = len(array)
	
	# looping through the input array
	for i in range(0,n):
		count = 0
		if (len(output_array) != 0):
			# checking the output array to see if repeated (count = 1)
			m = len(output_array)
			for j in range(0,m):
				if (array[i] == output_array[j]):
					count = 1
				else:
					continue
			# action if there is no repeat
			if (count == 0):
				output_array.append(array[i])
			else:
				continue			
		else:
			output_array.append(array[i])
	
	# returning filtered array
	return(output_array)

# function for printing and outputting the statistics
def results(filename_test, undetected_profiles, tf):
	'''
	This can potentially be moved into a function evaluation of the performance with 
	another metric - that is visual inspection of all of the test profiles to see if 
	the machine identified hit bottom location can be determined to within some threshold.

	Counting to gather statistics on the success rate of the model

	tf - parameter with True or False input - plots profile if tf = True
	'''

	# initialise file name list
	filearray = undetected_profiles
	n = len(undetected_profiles)

	# setting up statistics for counting the success rate of the test set
	correct = 0

	print('Plotting profiles and collecting statistics')
	# looping for printing
	for i in range(0,n):
		filename = filearray[i]
		source = "../HBfiles/"+filename
		[data, gradient, flags, hb_depth, latitude, longitude, date] = hb.read_data(source)
		# plotting temperature
		if (tf == True):
			plt.figure(figsize=(11.5,9))
			plt.subplot(1,2,1)
			plt.plot(data[:,1],data[:,0])
			plt.ylabel("Depth [m]")
			plt.xlabel("Temperature [degrees C]")
			plt.gca().invert_yaxis()
			plt.title("High probability HB region")
		# plotting HB depth (as flagged by QC)
		if (tf == True):
			for i in range(0,len(flags.flag)):
				if (flags.flag[i] == "HB"):
					ref = flags.depth[i]
					plt.axhline(y=ref, hold=None, color='r')
				else:
					continue
		# plotting points that have high probability of being HB (as computed by NN)
		m = len(filename_test)
		points = []
		for j in range(0,m):
			if (filename == filename_test[j]):
				points.append([pred[j][0],pred[j][1],z_test[j],T_test[j]])
			else:
				continue
		# collecting high probability points and plotting
		deeznuts = 0
		deeznuts_prob = 0
		plot_z = []
		plot_T = []
		high_z = []
		high_T = []
		max_T = []
		max_z = []
		for j in range(0,len(points)):

			# finding the point in profile with greatest probability of being a hit bottom
			if (points[j][0]>points[j][1]):
	
				# also worth checking for the point in the profile with the greatest difference
				if ((points[j][0]) > deeznuts_prob):
					deeznuts_prob = points[j][0]
					deeznuts = j

				# adding all points with greater probability to plot arrays
				plot_z.append(points[j][2])
				plot_T.append(points[j][3])

			# adding all points with a probability of being a hit bottom greater than 75%
			if (points[j][0] >= 0.75):
				high_z.append(points[j][2])
				high_T.append(points[j][3])

		max_z = points[deeznuts][2]
		max_T = points[deeznuts][3]
		if (tf == True):
			plt.plot(plot_T,plot_z,'go')
			plt.plot(high_T,high_z,'bo')
			plt.plot(max_T,max_z,'ro')

		# plotting gradient
		if (tf == True):
			plt.subplot(1,2,2)
			plt.plot(gradient[:,1], gradient[:,0])
			plt.ylabel("Depth [m]")
			plt.xlabel("Temperature Gradient [degrees C/m]")
			plt.gca().invert_yaxis()
			plt.title("dTdz")	
		# plotting HB depth (as flagged by QC)
		if (tf == True):
			for i in range(0,len(flags.flag)):
				if (flags.flag[i] == "HB"):
					ref = flags.depth[i]
					plt.axhline(y=ref, hold=None, color='r')
				else:
					continue
		# plotting detected points onto gradient plots
		if (tf == True):
			plt.axhline(y=max_z, hold=None, color='g')

		# collecting statistics
		for i in range(0,len(flags.flag)):
			if (flags.flag[i] == "HB"):
				ref = flags.depth[i]
				# choose threshold for detection
				if (abs(max_z-ref) < 10):
					correct += 1

		# printing the difference in the depth of the flagged and identified points
		print(filename)
		print("flagged HB depth: "+str(ref))
		print("detected HB depth: "+str(max_z))
		print("difference: "+str(abs(max_z-ref))) 
		print("\n")

		# plotting if tf == True
		if (tf == True):
			plt.show()

	# returning statistics
	correct_rate = correct/float(n)
	print("Correct detection rate: "+str(correct_rate))
	return(correct_rate)


######################################################################################################
# setting up data for feeding into neural network

# checking to see if model already exists
if (os.path.isfile('net.tflearn.index') == True):
	print("\nExisting neural network model found")
	print("loading...\n")
	model.load('net.tflearn')

else:
	print("\nneural network model not found")
	print("training new neural network...\n")
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
[X_new, y_new] = reduce_data(X_train,y_train)
X_train = X_new
y_train = y_new
[X_new, y_new] = reduce_data(X_val,y_val)
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
if (os.path.isfile('net.tflearn.index') == True):
	pass

else:
	# testing the tflearn reading data function
	net = tflearn.input_data(shape=[None,4])
	net = tflearn.fully_connected(net, 5, activation='sigmoid')
	net = tflearn.fully_connected(net, 2, activation='softmax')
	net = tflearn.regression(net)

	# creating the model
	model = tflearn.DNN(net)
	model.fit(X_train,y_train, n_epoch=25, validation_set = (X_val, y_val), \
				show_metric=True, run_id="XBT hit bottom data")
	#model.save('net.tflearn')

# using model to make predictions on test set
pred = model.predict(X_test)

# collecting key statistics on the data
[precision, files_incorrect] = statistics(pred, z_test, T_test, y_test, filename_test, 1, 0, 1)

# evaluation of the neural network
results(filename_test, files_incorrect, False)


######################################################################################################
