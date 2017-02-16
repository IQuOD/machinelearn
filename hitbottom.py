######################################################################################################
"""
This is a script with functions to check the the profiles for hit bottoms based on:
 - large spikes in the temperature gradient (over 5 sigma)
 - consecutive constant temperatures (for after the hit bottom event)
 - steady increases in temperature
 - small temperature increase spikes
 - bathymetry tests
"""
######################################################################################################
# libraries

import numpy as np
import pandas as pd
import math
import time
import os.path
import matplotlib.pyplot as plt
from netCDF4.utils import ncinfo
from netCDF4 import Dataset


######################################################################################################
# initialising - plotting and reading in data

# function to read in data from cdf files
def read_data(filename):
	"""
	Plot profiles if the "plot" input is equal to 1. For anything else, it will 
	not plot the profile of temperature and depth		
	Data will be stored in a python list (row major so the first index is the row)
	"""
	# read in files
	dat = Dataset(filename, 'r')
	init_depth = dat.variables['Depthpress']
	init_temp = dat.variables['Profparm']
	Act_Code = np.squeeze(dat.variables["Act_Code"])
	Aux_ID = np.squeeze(dat.variables["Aux_ID"])

	# joining the arrays for the Act codes
	QCflags = []
	QCdepths = []
	for i in range(0,len(Act_Code)):
		if (str(Act_Code[i]).replace("b","").replace(" ","").replace("'","").replace("[","").replace("]","") != ""):		
			code = str(Act_Code[i]).replace("b","").replace(" ","").replace("'","").replace("[","").replace("]","")
			QCflags.append(code)
			QCdepths.append(float(Aux_ID[i]))
	flags = pd.DataFrame({"flag":QCflags,"depth":QCdepths})

	# pulling the hit bottom depth out of the flags
	hb_depth = 0
	for i in range(0,len(flags.flag)):
		if (flags.flag[i] == "HB"):
			hb_depth = flags.depth[i]

	# choose variables that you want to import (other than T and z) here:
	latitude = dat.variables['latitude']
	longitude = dat.variables['longitude']
	date = dat.variables['woce_date']
	# also need the ID of the profile

	# create arrays (make use-able in python)
	init_depth = np.squeeze(init_depth)
	init_temp = np.squeeze(init_temp)
	latitude = np.squeeze(latitude)
	longitude = np.squeeze(longitude)
	date = np.squeeze(date)

	# filtering through invalid values
	"""
	This section of code takes temperature values that are non-physical (negative
	or above 50 degrees) and removes them from the data, storing it into a new array
	"""

	# taking temp and depth data
	depth = []
	temp = []
	m = len(init_depth)
	if (m != 0):
		try:
			for i in range(0,m):
				if (init_temp[i] < 0)|(init_temp[i]>50):
					continue
				else:
					depth.append(init_depth[i])
					temp.append(init_temp[i])
		except:
			print("depth filter failed, passed")
			pass
	n = len(depth)
	# writing all data to data frame
	data = np.column_stack((depth,temp))

	# taking temperature gradient and depth data
	dTdz = []
	depth_grad = []
	for jj in range(0,n-1):
		depth_grad.append(data[jj][0])
		dT = data[jj+1][1]-data[jj][1]
		dz = data[jj+1][0]-data[jj][0]
		dTdz.append(dT/dz)
	gradient = np.column_stack((depth_grad,dTdz))
	
	# close file
	dat.close()
	return(data, gradient, flags, hb_depth, latitude, longitude, date)

# defining a function for plotting the data
def plot_data(plot, data, gradient, flags, bathydepth, error_pts, pot_hb, 
			  filename, low_gradvar, bathlim):
	"""
	Input of the function for "plot" takes into account whether or not you want to plot 
	the points potential hit bottom and bad points
	"""

	# plotting temperature
	plt.figure(figsize=(11.5,9))
	plt.subplot(1,2,1)
	plt.plot(data[:,1],data[:,0])

	# plotting "bad data" if true
	if (plot == True):
		plt.plot(error_pts[:,1],error_pts[:,0],'bo')	
		pass

	# plotting the points of potential HB event if true
	if (plot == True):
		plt.plot(pot_hb[:,1],pot_hb[:,0],'ro')
		pass
	
	# plotting points of low gradient variation
	if (plot == True):
		plt.plot(low_gradvar[:,1],low_gradvar[:,0],'yo')
		pass

	plt.ylabel("Depth [m]")
	plt.xlabel("Temperature [degrees C]")
	plt.gca().invert_yaxis()
	plt.title("T")
	plt.axhline(y=bathydepth, hold=None, color='g')	
	for i in range(0,len(flags.flag)):
		if (flags.flag[i] == "HB"):
			ref = flags.depth[i]
			plt.axhline(y=ref, hold=None, color='r')
		else:
			continue
	plt.axhline(y=bathlim, hold=None, color='y')
	# plotting temperature gradient
	plt.subplot(1,2,2)
	plt.plot(gradient[:,1], gradient[:,0])
	plt.ylabel("Depth [m]")
	plt.xlabel("Temperature Gradient [degrees C/m]")
	plt.gca().invert_yaxis()
	plt.title("dTdz")	
	plt.axhline(y=bathydepth, hold=None, color='g')	
	for i in range(0,len(flags.flag)):
		if (flags.flag[i] == "HB"):
			ref = flags.depth[i]
			plt.axhline(y=ref, hold=None, color='r')
		else:
			continue
	plt.axhline(y=bathlim, hold=None, color='y')
	plt.suptitle("Temperature data for "+str(filename))
	plt.show()
	
	
# importing bathymetry data and using it as a test
def bathymetry(filename):
	"""
	filename needs to be the directory and filename of the file you want to read (.nc file)
	Note that the bathymetry data here also considers heights above the water too
	This information is stored in a global data
	"""
	dat = Dataset(filename, 'r')
	
	# extracting key information
	lon = dat.variables["lon"]
	lat = dat.variables["lat"]
	height = dat.variables["height"]
	lon = np.squeeze(lon)		
	lat = np.squeeze(lat)
	height = np.squeeze(height)	
	dat.close()

	"""
	height information is indexed such that each of the first indices is the lat
	and the second index gives the longitude
	"""
	
	# returning global variables
	bath_height = height
	bath_lon = lon
	bath_lat = lat

	return(bath_height, bath_lon, bath_lat)

# function to join the arrays of data available
def concat(*args):
	try:
		if (type(args[0]) == type(np.array([0,0]))):
			array = args[0]
			for i in range(1,len(args)):
				if (type(args[i]) == type(np.array([0,0]))):
					array = np.concatenate((array,args[i]),axis=0)
				else:
					continue
		else:
			array = args[1]
			for i in range(1,len(args)):
				if (type(args[i]) == type(np.array([0,0]))):
					array = np.concatenate((array,args[i]),axis=0)
				else:
					continue
	except:
		array = np.array([[0,0]])
		pass

	return(array)


######################################################################################################
# Computation functions
"""
This section of code is a logic checking implementation to attempt 
to detect hit bottoms for XBT drops.
These should all return a list of points (and associated temperature values)s
"""

# function to check for consecutive constant temperatures
def const_temp(data, gradient, consec_points, detection_threshold):
	"""
	consec_points: the number of points to count after the ith point (int)
	detection_threshold: absolute detection threshold for gradient, percentage of mean for 
						 temperature (float)

	Going to check for constant temperatures by looking for points with consecutive
	constant temperature and zero gradient. This will be done by averaging the ith point
	and some number of points after, and consider constant if they are within a chosen
	threshold (function input). Note that this will only detect the first instance the temperatures
	become constant

	returns the point where both the temperature and gradient stop fluctuating (both relatively
	constant)
	"""
	# importing key data from nested lists
	n = len(data[:,1])
	m = len(gradient[:,1])

	# global data
	global const_consec
	
	# finding constant profiles in temperature list
	const_temp = []
	const_depth_init = []
	for i in range(0,n-consec_points):
		Tav_consec = 0
		T_init = data[i][1]
		z_init = data[i][0]
		# counting the consecutive temperature values
		for j in range(0,consec_points):
			Tav_consec = Tav_consec + data[i+j][1]
		# finding the average and seeing if it is within threshold of initial value
		Tav_consec = Tav_consec/(float(consec_points))
		var = detection_threshold*T_init
		if (abs(Tav_consec - T_init) < var):
			const_temp.append(T_init)
			const_depth_init.append(z_init)
		else: 
			continue

	# finding constant profiles in gradient list
	const_dTdz = []
	const_z_init = []
	for i in range(0,m-consec_points):
		dTav_consec = 0
		dT_init = gradient[i][1]
		z_init = gradient[i][0]	
		# counting consecutive gradient values
		for j in range(0,consec_points):
			dTav_consec = dTav_consec + gradient[i+j][1]
		# finding the average and seeing if it's within threshold
		dTav_consec = dTav_consec/(float(consec_points))
		if (abs(dTav_consec-dT_init) < detection_threshold):
			const_dTdz.append(dT_init)
			const_z_init.append(z_init)
		else:
			continue
	
	# looking for matching values in T and dTdz profiles (identifies only the first)
	n1 = len(const_temp)
	n2 = len(const_dTdz)
	zvals = []
	Tvals = [] 
	for i in range(0,n1):	
		for j in range(0,n2):
			if (const_z_init[j] == const_depth_init[i]):
				zvals.append(const_depth_init[i])
				Tvals.append(const_temp[i])
			else:
				continue
	const_consec = np.column_stack((zvals,Tvals))

	return(const_consec)

# function to check for steady temperature increases
def temp_increase(data, consec_points):
	"""
	consec_points: number of points from the ith data point that need an 
	increasing or equal temperature	
	
	Looking at temperature profiles for regions where the temperature increases steadily
	(indicative of a hit bottom)
	"""	
	# length of data set 
	n = len(data[:,0])
	
	# arrays for storing all increasing T and z values
	global temp_grow
	T_grow = []
	z_grow = []

	# for loop to iterate over depth values
	for i in range(0,(n-consec_points)):
		count = 0
		for j in range(0, consec_points):
			if (data[i][1] <= data[i+j][1]):
				continue
			else:
				count = count + 1
		if (count != 0):
			continue
		else:
			zstart = data[i][0]
			Tstart = data[i][1]
		T_grow.append(Tstart)
		z_grow.append(zstart)

	# storing all together in nested list
	temp_grow = np.column_stack((z_grow,T_grow))

	return(temp_grow)

# function to pull the height data from bathymetry to compare
def bath_depth(latitude, longitude, bath_lon, bath_lat, bath_height):
	"""
	Find the depth in the bathymetry data from the lat and long provided
	Note that the longitude needs to be converted to 0-360 from -180-180
	Fuction returns the depth

	latitude and longitude refer to the values of lat and long of the profile being used
	the bath_lon, bath_lat and bath_height data is that from the bathymetry database
	"""
	# lengths of the two arrays
	n1 = len(bath_lat)
	n2 = len(bath_lon)
	
	# two for loops to find the appropriate lat and long
	for i in range(0,n1-1):
		if (latitude >= bath_lat[i]) & (latitude < bath_lat[i+1]):
			i_cor = i
		else:
			continue
	for j in range(0,n2-1):
		# converting longitude to appropriate scale
		if (longitude < 0):
			longitude = -longitude
		if (longitude >= bath_lon[j]) & (longitude < bath_lon[j+1]):
			j_cor = j
		else:
			continue
	
	# correct bathymetry value is returned heres
	corr_depth = abs(bath_height[i_cor][j_cor])

	return(corr_depth)

# writing function to find small temperature spikes that may be indicative of a hit bottom
def T_spike(data, threshold):
	"""
	The function takes in the data of temperature and depth and looks for a small
	increase and immediate decrease in the temperature. Threshold is a measure of how 
	much the temperature has to increase (and decrease) to count as a small spike
	"""
	# initialising arrays and length of loop
	temp = data[:,1]
	depths = data[:,0]
	Tspike_z = []
	Tspike_T = []
	spikes = []
	n = len(temp)

	# for loop to identify points with spike
	for i in range(0,n-2):
		init = temp[i]
		zinit = depths[i]
		next = temp[i+1]
		after = temp[i+2]
		
		# logic to find increase and decrease values
		if (init < (next-threshold)):
			if ((next-threshold) > after):
				"""
				These points meet the requirement that the initial value is lower than 
				the next and that the value after is smaller again
				"""
				Tspike_T.append(init)
				Tspike_z.append(zinit)
			else:
				continue
		else:
			continue
	spikes = np.column_stack((Tspike_z,Tspike_T))
	
	return(spikes)
	

# finding characteristic gradient increase spike for hit bottoms
def grad_spike(data, gradient, threshold):
	"""
	threshold: is the number of standard deviations away from the mean gradient value you 

	want to use for identifying a spike (IDENTIFIED 3 IS MOST IDEAL)
	
	Using large positive gradient spikes to identify whether or not there is a peak
	Wire breaks tend to have high temperature after gradient peak, so remove those

	with higher temp after peak
	Note that this is only to identify clear, large spikes in grad (one property of HB data)
	"""
	# importing key data from nested lists
	n = len(data[:,1])
	m = len(gradient[:,1])
	
	# calculate average gradient value
	if (m != 0):
		dt_av = sum(gradient[:,1])/float(m)
	else:
		dt_av = 999

	# calculate standard deviation in the gradient (average)
	"""	
	Note that this method of finding the standard deviation is not truly good for profiles
	observable trends in dT (overall trend needs to be subtracted)

	"""
	sq_dev = 0
	for i in range(0,m):
		sq_dev = sq_dev + (dt_av - gradient[i][1])**2
	if (m != 0):	
		std_dev = np.sqrt(sq_dev/float(m))
	
		"""
		Finding large peaks (3 std.dev away from average chosen arbitrarily, T must decrease)

		"""	
		global dTdz_peaks
		z_peak = []
		T_peak = []
		for i in range(0,m):
			if (abs(gradient[i][1]-dt_av) > threshold*std_dev) & (gradient[i][1] > 0):
				z_peak.append(data[i][0])
				T_peak.append(data[i][1])
		if (len(z_peak) != 0):
			dTdz_peaks = np.column_stack((z_peak,T_peak))
			return(dTdz_peaks)

		else:
			return(999)
	else:
		return(999)


# code to find the appropriate range of depths from the surrounding bathymetry data
def depth_limits(latitude, longitude, bath_lon, bath_lat, bath_height):
	'''
	latitude and longitude - parameters to give information about the position of the drop
	
	bath_lon, bath_lat, bath_height - give the information about the depth in the ocean for a
									  given depth and position
	'''
	# lengths of the two arrays
	n1 = len(bath_lat)
	n2 = len(bath_lon)
	
	# identify the index for the profile location in the bathymetry data
	i_cor = 0
	j_cor = 0
	for i in range(0,n1-1):
		if (latitude >= bath_lat[i]) & (latitude < bath_lat[i+1]):
			i_cor = i
		else:
			continue
	for j in range(0,n2-1):
		# converting longitude to appropriate scale
		if (longitude < 0):
			longitude = -longitude
		if (longitude >= bath_lon[j]) & (longitude < bath_lon[j+1]):
			j_cor = j
		else:
			continue
	
	# find the max and min depths in a given range of lat and long
	diff = 3
	depth_array = []
	for i in range((i_cor-diff),(i_cor+diff)):
		for j in range((j_cor-diff),(j_cor+diff)):
			# try method to eliminate errors at boundaries	
			try:
				# storing all of the depth values
				depth = abs(bath_height[i][j])
				depth_array.append(depth)
			except:
				continue
	
	# finding the smallest and largest values of the depths from bathymetry
	min_depth = min(depth_array)
	max_depth = max(depth_array)	
	return(min_depth)


# code that will attempt to identify regions where the gradient variation decreases 
def grad_var(data, gradient, threshold, consec):
	'''
	Function that identifies the lowest gradient and looks in the profile for points that
	have gradients similiar enough to that value (up to some threshold)
	
	data and gradient are lists of data that are being fed into the function (to be used)

	threshold - the detection threshold. This value should always be greater than 1

	consec - the number of points after a given point you want to look at to calculate
			 the variation (and mean) from
	'''
	
	# finding region (with +- consec points around) with lowest gradient variation
	gradvar_low = 999
	n = len(gradient[:,1])

	# looping through all of the points to find the one with the lowest gradient variation
	for i in range(0,n):
		# computing the mean 
		mean = 0
		length = 0
		dev = 0
		for j in range(i,i+consec):
			try: 
				mean = mean + gradient[j][1]
				length += 1
			except:
				continue
		mean = float(mean)/float(length)
		# computing the standard deviation
		for j in range(i,i+consec):
			try:
				dev = dev + (gradient[j][1]-mean)**2
			except:
				continue
		dev = float(dev)/float(length)
		# finding the point with the lowest standard deviation
		if (dev < gradvar_low):
			if (dev == 0):
				continue
			else:
				gradvar_low = dev
		else:
			continue
	
	# looking for all of the points within some threshold of the lowest gradient variation
	pointz = []
	pointT = []
	for i in range(0,n):
		# compute the gradient variation in the consec range
		mean_sample = 0
		stddev_sample = 0
		len_sample = 0
		for j in range(i,i+consec):
			try:
				mean_sample = mean_sample + gradient[j][1]
				len_sample += 1
			except:
				continue
		mean_sample = float(mean_sample)/float(len_sample)
		for j in range(i,i+consec):
			try:
 				stddev_sample = stddev_sample + (gradient[j][1]-mean_sample)**2
			except:
				continue
		stddev_sample = stddev_sample/float(len_sample)
		# checking it is within some threshold, record temperature and depth
		if ((stddev_sample/gradvar_low) < threshold):
			if (i > (n-20)):
				continue
			else:
				pointz.append(data[i][0])
				pointT.append(data[i][1])
		else:
			continue
		
	# returning the points that fall within the threshold
	grad_drop = np.column_stack((pointz,pointT))
	return(grad_drop)


# function to identify chains
def find_chains(array, yn):
	'''
	Finds all of the 'chains' in an array (considered part of the same chain if the two consecutive
	points are less than 10m apart

	Prints information to user if yn == True, doesn't if yn == False
	'''
	n = len(array)
	chain_start_index = []
	chain_end_index = []

	# finding the indices of the upper limit and lower limit
	if (len(array) > 0):	
		z_init = array[0][0]
		onoff = 0
		for i in range(0,n-1):
			if (abs(z_init - array[i+1][0]) < 10):
				if (onoff == 0):	
					chain_start_index.append(i)		
					onoff = 1
			else:
				if (onoff == 1):
					chain_end_index.append(i)
					onoff = 0
			z_init = array[i+1][0] 
	
			# if it is the end of the array
			if (i == (n-2)):
				if (onoff == 1):
					chain_end_index.append(n)
					onoff = 0
				else:
					continue		

	else:
		if (yn == True):
			print("No points in the array to filter")
		return(array)

	# checking lengths of the arrays are equal (if contintion met)	
	if (yn == True):
		print("Arrays are equal length: ", len(chain_start_index)==len(chain_end_index))
		print(len(chain_start_index),len(chain_end_index))
		print("length of array: ", n)
		print('starting indices: ', chain_start_index)
		print('ending indices: ', chain_end_index)
	
	# returning the starting and ending indices of the chains
	if (len(chain_start_index) != 0):
		return(chain_start_index, chain_end_index)	
	else:
		return(0)


# function to remove points above limits set by bathymetry
def remove_above(array, upper_lim, yn):
	'''
	Writing a code to remove all of the points for a given array that are above (or connected to a 
	chain starting above) the lower limit of depth set by the bathymetry
	
	array - the array of data that you want to check (use with low_gradvar is ideal, and possibly 
	the bad data array as well).

	Other parameters are to locate the minimum depth set from bathymetry
	'''
	# pulling the function to find the number of chains
	n = len(array)
	filtered_array = []
	y = find_chains(array, yn)
	if (y != 0):
		chain_start_index = y[0]
		chain_end_index = y[1]
		m = len(chain_start_index)
	else:
		m = 0

	# removing points that are above the chain (not including them in the new array)
	'''
	Check the starting indices of each of the chains. If it is above the 'restricted' region, 
	ignore and don't use the data (don't append to the filtered array). Otherwise, add the 
	data to filtered_array. First loop is to find the index that should be defined as the 'cutoff'
	'''
	# if there is more than one chain
	if (m > 1):
		cutoff_index = chain_end_index[m-1]
		# identifying the cutoff index
		for i in range(0,m):
			if (array[chain_start_index[i]][0] < upper_lim):
				continue
			else:
				cutoff_index = chain_start_index[i]
				break
		# adding the range of points
		for i in range(cutoff_index, n):
			filtered_array.append([array[i][0],array[i][1]])
		filtered_array = np.array(filtered_array)
	# if there is exactly one chain
	elif (m == 1):
		if (array[0][0] > upper_lim):
			for i in range(0,len(array)):
				filtered_array.append([array[i][0],array[i][1]])
		else:	
			pass
		filtered_array = np.array(filtered_array)
	# no chains:
	else:
		filtered_array = np.array(array)
		
	# returning and printing a length change if the user wants this
	if (yn == True):
		print("length change: ",len(filtered_array),len(array))
	return(filtered_array)
			

######################################################################################################

# code for finding the depth of the hit bottom given the predicted hit bottom data
def hit_predict(data, error_pts, pot_hb, bad_points, fraction, hb_depth):
	"""
	Writing a function to collect all of the points and determine the values of 3 key parameters
	that optimise the detection efficiency (true positive rate)
	
	function inputs:
	bad_points - number of points after a given depth that we will look for bad points
	fraction - fraction of points required (in the bad_points range) to consider the initial 
			   point to be the "detected" hit bottom
	fracAbove - if we're using the blue points (pot_hb points not meeting criteria for a hit 
				bottom) then look at the fraction of points above and below (given as below/total)
				and use as a metric to determine whether or not a given "blue" point can be
				flagged as the hit bottom
	(other function inputs are defined earlier)
	
	function returns:
	predicted depth of hit bottom

	function edit - only working on the points with potential hit bottom points at the top
	"""
	n = len(data)
	all_data = concat(error_pts, pot_hb)
	predHB = 0
	
	if (len(pot_hb) >= 1):
	# trying this code only on those with potential hit bottom points

		try:
			# if we have an array of values for the potential hit bottom
			if (type(pot_hb) == type(np.array([[0,0]]))):
				#print("array of HB points found")
				# first look through the potential hit bottom points
				if (len(pot_hb) >= 1):	
					#print("long array of pot_hb")
					for j in range(0,len(pot_hb)):	
	
						# find correct index
						eq_index = 0
						for i in range(0,n):
							if (pot_hb[j][0]==data[i][0]):
								eq_index = i
							else:
								continue
				
						# count the number of bad points within range
						count = 0
						for i in range(eq_index,eq_index+bad_points):
							for k in range(0,len(error_pts)):	
								if (data[i][0] == error_pts[k][0]):
									count = count +1 
								else:	
									continue
				
						# computing fraction of points detected required
						frac = count/float(len(error_pts))
						if (frac >= fraction):
							predHB = pot_hb[j][0]
							break

				else:
					#print("pot_hb only one entry")
					# find correct index
					eq_index = 0
					for i in range(0,n):
						if (pot_hb[0][0]==data[i][0]):
							eq_index = i

					# count the number of bad points within
					count = 0
					for i in range(eq_index,eq_index+bad_points):
						for k in range(0,len(error_pts)):	
							if (data[i][0] == error_pts[k][0]):
								count = count +1 
							else:	
								continue

					# computing fraction of points detected required
					frac = count/float(len(error_pts))
					if (frac > fraction):
						predHB = pot_hb[0][0]

		except:
			predHB = 0
			pass

	return(predHB)



######################################################################################################
