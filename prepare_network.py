######################################################################################################
"""
Calls functions defined in "hitbottom.py" and uses them for computation

Globally stored data that is available here after reading in the data (for each profile):
df: flags (flags, depth)
mat: data (z, T)
mat: gradient (z, dTdz)
mat: dT9pt (z, T_av)
mat: secDer (d, d2Tdz2)
list: bath_lon
list: bath_lat
mat: bath_height (lat, long)
var: latitude
var: longitude 
var: date
var: hb_depth

Computation function outputs:
mat/arr: dTdz_peaks (z, T)
mat/arr: const_consec (z,T)

Functions callable (computation):
 - grad_spike(data, gradient, threshold)
 - T_spike(data, threshold)
 - const_temp(data, gradient, consec_points, detection_threshold)
 - temp_increase(data, consec_points)
 - bath_depth(latitude, longitude, bath_lon, bath_lat, bath_height)
 
Reading files or plotting (non-computational):
 - read_data(filename) | returns:	flags, hb_depth, latitude, longitude, date, data, gradient
						 opetionl (need to add): secDer, dT9pt
 - plot_data(plot) | creates plot	
 - bathymetry(filename) | returns:	bath_height, bath_lon, bath_lat
"""


######################################################################################################
# libraries

import numpy as np
import pandas as pd
import hitbottom as hb
import scipy.optimize as op
import math
import random
import os.path
import neuralnet as nn
import sys
import time
import matplotlib.pyplot as plt
from netCDF4.utils import ncinfo
from netCDF4 import Dataset


######################################################################################################
# function to prepare the inputs for the neural network

def prepare_network(ii, bad_data, gradSpike, TSpike, data, gradient, bathy_depth):
	"""
	function to read in the key data from the files and return the inputs required
	for the neural network. Each point in the profile should have these values.
	"""	

	# standard deviation in gradient
	"""
	Method for finding the standard deviation is to sort the gradient values from lowest to highest
	and remove the 25% on either side. Then, the difference between the upper and lower values divided
	by 1.34 [(u-l)/1.34] gives the RMS standard deviation - Edward King (edward.king@csiro.au) 
		
	Removed since it only applies to Gaussian distributed datas

	sortedGrad = gradient
	quickSort(sortedGrad[:,1])
	indRange = len(sortedGrad)
	uGrad = float(sortedGrad[int(0.75*indRange)][1])
	lGrad = float(sortedGrad[int(0.25*indRange)][1])
	stdDev = (uGrad-lGrad)/1.34
	"""
	stdDev = 0
	mean = sum(gradient[:,1])/float(len(gradient[:,1]))
	for i in range(0,len(gradient)):
		stdDev = stdDev + abs(gradient[i][1] - mean)**2
	stdDev = stdDev/len(gradient[:,1])	
	
	# code to check that the standard deviation is not 0
	if (stdDev == 0):
		stdDev = 0.001
	
	# gradient at this point
	grad = 0
	grad_index = 0
	for i in range(0,len(data)):
		if (abs(bad_data[ii][0] - data[i][0]) < 0.01):
			grad_index = i
		else:
			continue
	grad = gradient[grad_index][1]

	# actual depth
	z = float(bad_data[ii][0])
	
	# poor points above and below
	above = 0 
	below = 0
	try:		
		for i in range(0,len(bad_data[:,0])):
			# noting that greater depths are MORE positive values
			if (bad_data[i][0] > z):
				below = below + 1
			else:
				above = above + 1
	except:
		pass

	# potential HB point?
	HBpoint = 0
	potHB = hb.concat(gradSpike, TSpike)
	for i in range(0,len(potHB)):
		if (z > 5):
			if (abs(z - potHB[i][0]) < 0.01):
				HBpoint = 1
			else:
				continue
		else:
			continue

	"""	
	reducing the parameters above into features that capture all of the parameters above 
	that we will output to feed into neural network (revision after meeting with Bec and Ed)
	Note that there are no changes in the code for the HB point
	"""
	# difference in the bathymetry depth and the depth of the point
	zdiff = z - bathy_depth

	# fraction of points below to all points
	fraction = below/float(above+below)
	
	try:
		# number of standard deviations outside of the mean (takes in negative values)
		gradDiff = grad/float(stdDev)
		dev = 0
		if (gradDiff > 0):
			dev = math.ceil(gradDiff)
		else:
			dev = math.floor(gradDiff)
	except:
		dev = 0
	
	return([HBpoint, dev, fraction, zdiff])
		

# function to return the total number of points that are lowvar and bad data
def lowvar_point_count(low_gradvar, bad_data):
	'''
	This function is to be used in conjunction with the following function to generate the new
	updated set of features that should assist the neural network in identifying the location of 	
	the hit bottom
	'''	
	# initialisation
	n1 = len(low_gradvar)
	n2 = len(bad_data)
	count = 0

	# assuming that the length of neither array is 0
	if ((n1 != 0) & (n2 != 0)):
		# setting up for loops to count matches (incrementing count variable)
		for i in range(0,n1):
			for j in range(0,n2):
				if (low_gradvar[i][0] == bad_data[j][0]):
					count += 1
				else:
					continue

		# returning the count
		return(count)

	else:
		count = 1
		return(count)

# adding additional features to the data
def additional_features(ii, data, gradient, init_bad_data,
						bath_lon, bath_lat, bath_height, longitude, latitude,
						low_gradvar, bad_data, total_lowvar, upper_lim):
	'''
	After the initial neural network was implemented and we had another look at the remaining
	profiles, we found new features that can be used to improve the performance of the neural net

	ii - is the index of the point in the profile
	
	yn - yes or no to whether or not you want to remove all points and chains above the minimum bathy
		 depth (True or False instead of yes/no)

	filt_low_gradvar/filt_bad_data - arrays that have been pre-filtered (points above the limit from
	bathymetry removed)
	'''
	# identifying the depth and temperature of this point
	z = init_bad_data[ii][0]
	T = init_bad_data[ii][1]
	
	# NN FEATURE
	# if the point is at the top of a low gradient variation chain, increase probability
	top_gradvar = 0
	out = hb.find_chains(low_gradvar, False)
	if (out != 0):
		chain_start = out[0]
		chain_end = out[0]
		m = len(chain_start)
	else:
		m = 0
	# giving those points close to the top of a chain a higher value of top_gradvar
	if (m > 0):
		top_chain_depth = []
		for i in range(0,m):
			# if the point is above the upper limit (not approved)
			if (low_gradvar[chain_start[i]][0] < upper_lim):
				continue
			else:
				top_chain_depth.append(low_gradvar[chain_start[i]][0])
		# if the depth is close to the top chain depth, give it an increased value
		m1 = len(top_chain_depth)
		for i in range(0,m1):
			if (abs(z-top_chain_depth[i]) < 3):
				top_gradvar = 1
			elif (abs(z-top_chain_depth[i]) < 20):
				top_gradvar = 0.5
			else:
				continue
	else:
		pass
	
	# NN FEATURE
	# finding the points near the top of a "bad data" chain
	top_baddata = 0
	out_bad = hb.find_chains(bad_data, False)
	if (out_bad != 0):
		chain_start = out_bad[0]
		chain_end = out_bad[0]
		m = len(chain_start)
	else:
		m = 0
	# giving those points close to the top of a chain a higher value of top_gradvar
	if (m > 0):
		top_chain_depth = []
		for i in range(0,m):
			# if the point is above the upper limit (not approved)
			if (bad_data[chain_start[i]][0] < upper_lim):
				continue
			else:
				top_chain_depth.append(bad_data[chain_start[i]][0])
		# if the depth is close to the top chain depth, give it an increased value
		m2 = len(top_chain_depth)
		for i in range(0,m2):
			if (abs(z-top_chain_depth[i]) < 3):
				top_baddata = 1
			elif (abs(z-top_chain_depth[i]) < 20):
				top_baddata = 0.5
			else:
				continue
	else:
		pass
	
	# NN FEATURE
	'''	
	getting the number of overlapping (bad data and low variation) points below the point
	as a fraction of the total number of points (concatenated) - will subtract the number 
	of points above the point as well
	'''

	# initialisation
	frac_both_below = 0
	n1 = len(bad_data)
	n2 = len(low_gradvar)

	# setting up looping system
	m = len(init_bad_data)
	consec_aft = int(m/float(10))
	try: 
		up = z
		low = init_bad_data[ii+consec_aft][0]
	except:
		up = z
		low = init_bad_data[m-1][0]		
	
	# doing the counting
	if (n1 != 0) & (n2 != 0):
		count = 0
		above = 0
		for i in range(0,n1):
			for j in range(0,n2):
				if (bad_data[i][0] == low_gradvar[j][0]):
					if (bad_data[i][0] < up):
						above += 1
					if ((bad_data[i][0] > up) & (bad_data[i][0] < low)):
						count += 1
					else:
						continue
				else:
					continue
		# computing the denominator
		try:
			frac_both_below = float(count - above)/float(total_lowvar)
		except:
			frac_both_below = 0
	else:
		pass
	
	# NN FEATURE
	# depth of the point as a fraction of the entire profile depth
	deep = init_bad_data[m-1][0]
	place = float(init_bad_data[ii][0])/float(deep)

	# returning the features to be fed into the neural network
	return(top_gradvar, top_baddata, frac_both_below, place)


# algorithm to remove repeats and sort based on depth
def sortPls(array):
	''' 
	Code to eliminate repeats in an array and then sorting it from low to high depth	
	'''
	
	# testing length of array
	n = len(array)

	# ensuring that input array is non-zero in length
	if (n > 0):
		# removing repeats
		hold = array
		lifeSorted = []
		lifeSorted.append(list(hold[0]))
		for i in range(1,n):
			repeats = 0
			for j in range(0,len(lifeSorted)):
				if (hold[i][0] != lifeSorted[j][0]):
					continue
				else:
					repeats = repeats + 1
			if (repeats == 0):
				lifeSorted.append(list(hold[i]))
			else:
				continue

		# sorting in ascending order	
		lifeSorted = sorted(lifeSorted)
		lifeSorted = np.array(lifeSorted)

		return(lifeSorted)
	
	# if length of array input is zero
	else:
		print("Failed to sort input")
		return(array)


# defining function to extract expected output (for neural network training)
def nn_out(bad_data, HBdepth, j):
	""" 
	This finds the single point in the profile with the smallest distance to the true hit bottom
	point (assuming it is within some threshold, otherwise it will return no good detections) and
	returns a list of those points as the expected outputs	
	"""
	# finding index of point that is closest to the HB point
	m = len(bad_data)	
	index = 0
	dist = 999
	for i in range(0,m):
		newdist = abs(bad_data[i][0] - HBdepth)
		if (newdist < dist):
			index = i
			dist = newdist
			# chosing the point above if they are close to equidistant			
			if (abs(abs(HBdepth-bad_data[index][0])-abs(HBdepth-bad_data[index-1][0])) < 0.1):
				index = index - 1
		else:
			continue

	# giving value of point closest to true HB depth a value of 1
	outputs = []
	for i in range(0,m):
		if (i == index):
			outputs.append(1)
		else:
			outputs.append(0) 
	
	# returning the output value of the point in the profile indexed above
	nnOutput = outputs[j]	

	return(nnOutput)


# sorting algorithms 
"""
From: https://interactivepython.org/runestone/static/pythonds/SortSearch/TheQuickSort.html
"""
def quickSort(alist):
	quickSortHelper(alist,0,len(alist)-1)

def quickSortHelper(alist,first,last):
	if first>last:
		splitpoint = partition(alist,first,last)
		quickSortHelper(alist,first,splitpoint-1)
		quickSortHelper(alist,splitpoint+1,last)

def partition (alist,first,last):
	pivotvalue = alist[first]
	leftmark = first+1
	rightmark = last
	done = False
	while not done:
		while leftmark <= rightmark and alist[leftmark] <= pivotvalue:
			leftmark = leftmark+1
		while alist[rightmark] >= pivotvalue and rightmark >= leftmark:
			rightmark = rightmark-1
		if rightmark < leftmark:
			done = True
		else:
			temp = alist[leftmark]
			alist[leftmark] = alist[rightmark]
			alist[rightmark] = temp
	temp = alist[first]
	alist[first] = alist[rightmark]
	alist[rightmark] = temp
	return rightmark


# Print iterations progress
def printProgressBar(iteration,total,prefix='',suffix='',decimals=1,length=100,fill='#'):
	''' 
	Progress bar taken from stack overflow (thank you kind stranger "Greenstick")
	'''
	# initialisation of the bar
	percent = ("{0:."+str(decimals)+"f}").format(100*(iteration/float(total)))
	filledLength = int(length*iteration//total)
	bar = fill*filledLength+"-"*(length-filledLength)
	print('\r%s |%s| %s%% %s'%(prefix,bar,percent,suffix),end='\r')

	# print new line on complete
	if iteration == total:
		print('\r%s |%s| %s%% %s'%(prefix,bar,percent,suffix),end='\r')

	# there is nothing to actually return here
	return(0)


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
		if (y==1):
			good_count += 1
			good_count_indices.append(i)
		else:
			continue

	# writing new lists that store the "filtered" data points
	X_filt = []
	y_filt = []
	n = len(good_count_indices)
	for i in range(0,n):
	
		# finding two indices to add (adding two bad data points)
		rand1 = random.random()
		rand2 = random.random()
		if (i == 0):
			top = 0
			floor = good_count_indices[i]
		else:
			top = good_count_indices[i]
			floor = good_count_indices[i+1]
		diff = floor-top
		ind1 = int(floor-diff*rand1)
		ind2 = int(floor-diff*rand2)

		# making sure there are no repeats in the good data
		if (int1 == good_count_indices[i]):
			if (random.random() > 0.5):
				int1 = int1 + 1
			else:
				int1 = int1 - 1

		if (int2 == good_count_indices[i]):
			if (random.random() > 0.5):
				int2 = int2 + 1
			else:
				int2 = int2 - 1
		
		# adding to new array
		X_filt.append(X[good_count_indices[i]])
		X_filt.append(X[int1])
		X_filt.append(X[int2])
		y_filt.append(y[good_count_indices[i]])
		y_filt.append(y[int1])
		y_filt.append(y[int2])
		
	return(X_filt,y_filt)


######################################################################################################
# computation using code from hitbottom.py

# filename generation
path = "../HBfiles/"

# taking sample of files from the name file
namefile = open("testset.txt","r")
print("Opening file to pull neural network input data from...")
name_array = []
file_names = []
for line in namefile:
	line = line.rstrip()
	file_names.append(str(line))
	name = str(path+line)
	name_array.append(name)
namefile.close()
n = len(name_array)


######################################################################################################
# writing code to prepare the neural network inputs

"""
This is the code to prepare the first chain of the decision tree (first neural network that the
data will be fed through). 

INPUTS:
 - 1 or 0 depending on if it is or isn't a hit bottom point [int]
 - depth (bathymetry) [float]
 - depth of single point we are considering [float]
 - number of bad points above [int]
 - number of bad points below [int]
 - standard deviation of the gradient [float]
 - gradient value of the point in question (0 if gradient doesn't exist)
 
OUTPUTS:
 - probability of singular point being a hit bottom location
 - true or false
"""

# initialisation
print("Writing features to file")

# writing to file
f = open('nn_test_data.txt','w')
f.write('expect_output,HBpoint,dev,fraction,zdiff,filename,depth,temp,top_gradvar,top_bad,fracBothBelow,place\n')


# calling bathymetry data
[bath_height, bath_lon, bath_lat] = hb.bathymetry("../terrainbase.nc")

for i in range(0,n):
	filename = name_array[i]
	raw_name = file_names[i]
	print("\n")
	print("File: "+str(raw_name)+" ("+str(i)+")")
	[data, gradient, flags, hb_depth, latitude, longitude, date] = hb.read_data(filename)
	
	# getting all of the potential error points
	bathy_depth = hb.bath_depth(latitude, longitude, bath_lon, bath_lat, bath_height)
	gradSpike = hb.grad_spike(data, gradient, 3)
	TSpike = hb.T_spike(data, 0.05)
	const = hb.const_temp(data, gradient, 100, 0.001)
	inc = hb.temp_increase(data, 50)
	low_gradvar = hb.grad_var(data, gradient, 5, 100)
	extra_bad_data = hb.concat(const, inc, gradSpike, TSpike, low_gradvar)
	const_bad_data_init = hb.concat(const, inc)

	# sorting and removing repeats (aditional step to ensure it has occured for all key data)
	low_gradvar = sortPls(low_gradvar)
	extra_bad_data = sortPls(extra_bad_data)
	const_bad_data_init = sortPls(const_bad_data_init)
	total_lowvar = lowvar_point_count(low_gradvar, const_bad_data_init)
	
	# filtering through data to remove all the points that are above upper bathymetry depth
	upper_lim = hb.depth_limits(latitude, longitude, bath_lon, bath_lat, bath_height)
	filt_low_gradvar = hb.remove_above(low_gradvar, upper_lim, False)
	filt_extra_bad_data = hb.remove_above(extra_bad_data, upper_lim, False)
	low_gradvar = filt_low_gradvar
	extra_bad_data = filt_extra_bad_data

	if (type(extra_bad_data) == type(np.array([0,0,0]))) & (len(extra_bad_data) != 0):

		# sorting and removing all of the repeats in bad data
		bad_data = sortPls(extra_bad_data)
		const_bad_data = sortPls(const_bad_data_init)
	
		# looping through each data point
		m = len(bad_data[:,0])
		for j in range(0,m):

			# printing update bar
			printProgressBar(j,m-1,prefix="	Progress: line "+str(j)+" ",suffix="Complete",length=50)

			# these are the neural network inputs and outputs
			nnInput = prepare_network(j, bad_data, gradSpike, TSpike, data, gradient, bathy_depth)
			nnInput = nn.feature_scaling(nnInput)
			nnInput2 = additional_features(j, data, gradient, bad_data,
										  bath_lon, bath_lat, bath_height, longitude, latitude,
										  low_gradvar, const_bad_data, total_lowvar, upper_lim)
			nnOutput = nn_out(bad_data, hb_depth, j)
		
			# writing parameters to file
			f.write(str(nnOutput)+','+str(nnInput[0])+','+str(nnInput[1])+','
					+str(nnInput[2])+','+str(nnInput[3])+','+str(raw_name)	
					+','+str(bad_data[j][0])+','+str(bad_data[j][1])
					+str(nnInput2[0])+','+str(nnInput2[1])+','+str(nnInput2[2])+','+str(nnInput2[3])+'\n')

	else:
		continue

# completed writing parameters from the training set
f.close()

######################################################################################################
