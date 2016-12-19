######################################################################################################
"""
This is a script to check the the profiles for hit bottoms based on:
 - large spikes in the temperature gradient (over 5 sigma)
 - consecutive constant temperatures (for after the hit bottom event)
 - steady increases in temperature
 - bathymetry tests
"""
######################################################################################################
# libraries

import numpy as np
import pandas as pd
import math
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
	global flags
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

	# choose variables that you want to import (other than T and z) here:
	global latitude, longitude, date
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
	# User can change the global variables that are outputted if they wish
	global data, gradient

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

	# looking at the second derivative
	d2Tdz2 = []
	depth_secgrad = []
	global secDer
	for jj in range(0,n-2):
		depth_secgrad.append(gradient[jj][0])
		der = float((gradient[jj+1][1]-gradient[jj][1])/(gradient[jj+1][1]-gradient[jj][1]))
		d2Tdz2.append(der)
	secDer = np.column_stack((depth_secgrad,d2Tdz2))

	# taking a moving average for the temperature values
	global dT9pt
	depth_9pt = []
	temp9pt = []
	n = len(gradient[:,0])
	for jj in range(4,n-4):
		depth_9pt.append(gradient[jj][0])
		Tav = (gradient[jj-4][1]+gradient[jj-3][1]+gradient[jj-2][1]+gradient[jj-1][1]+gradient[jj][1]+gradient[jj+1][1]+gradient[jj+2][1]+gradient[jj+3][1]+gradient[jj+4][1])/float(9.0)
		temp9pt.append(Tav)
	dT9pt = np.column_stack((depth_9pt,temp9pt))
	
	# close file
	dat.close()	

# defining a function for plotting the data
"""
Input to the plot function is either True or False, where True will show the plots
and False will not show the plots
"""
def plot_data(plot):
	
	# conditional subplot of temperature and gradient
	if (plot == True):
		# plotting temperature
		plt.figure(figsize=(11.5,9))
		plt.subplot(1,3,1)
		plt.plot(data[:,1],data[:,0])
	
		# plotting additonal features
		spikes = spike(data,gradient, 3)
		if (type(spikes) != int):
			plt.plot(spikes[:,1], spikes[:,0],'ro')
		consecpts = const_temp(data,gradient,100,0.001)
		plt.plot(consecpts[:,1],consecpts[:,0],'go')
		grow = temp_increase(data,100)
		plt.plot(grow[:,1],grow[:,0],'bo')	
		bath_z = bath_depth(latitude, longitude, bath_lon, bath_lat, bath_height)
		plt.axhline(y=bath_z, hold=None, color='g')	

		plt.ylabel("Depth [m]")
		plt.xlabel("Temperature [degrees C]")
		plt.gca().invert_yaxis()
		plt.title("T")
		for i in range(0,len(flags.flag)):
			if (flags.flag[i] == "HB"):
				ref = flags.depth[i]
				plt.axhline(y=ref, hold=None, color='r')
			else:
 				continue
		# plotting temperature gradient
		plt.subplot(1,3,2)
		plt.plot(gradient[:,1], gradient[:,0])
		plt.ylabel("Depth [m]")
		plt.xlabel("Temperature Gradient [degrees C/m]")
		plt.gca().invert_yaxis()
		plt.title("dTdz")
		for i in range(0,len(flags.flag)):
			if (flags.flag[i] == "HB"):
				ref = flags.depth[i]
				plt.axhline(y=ref, hold=None, color='r')
			else:
 				continue
		# plotting temperature 9 point moving average
		plt.subplot(1,3,3)
		plt.plot(dT9pt[:,1], dT9pt[:,0])
		plt.ylabel("Depth [m]")
		plt.xlabel("T - 9pt moving av [degrees C]")
		plt.gca().invert_yaxis()
		plt.title("T 9pt MA")
		for i in range(0,len(flags.flag)):
			if (flags.flag[i] == "HB"):
				ref = flags.depth[i]
				plt.axhline(y=ref, hold=None, color='r')
			else:
 				continue
		plt.suptitle("Temperature data for "+str(filename))
		plt.show()
	else:
		pass
	
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
	global bath_height, bath_lon, bath_lat
	bath_height = height
	bath_lon = lon
	bath_lat = lat

	return(0)


######################################################################################################
# Computation functions
"""
This section of code is a logic checking implementation to attempt 
to detect hit bottoms for XBT drops.
"""

# finding characteristic gradient increase spike for hit bottoms
def spike(data, gradient, threshold):
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
	

######################################################################################################
# computation (WHERE TO MANIPULATE TO USE CODES ABOVE)
######################################################################################################

# filename generation
path = "../HBfiles/"

# taking sample of files from the name file
namefile = open("HB_content.txt","r")
name_array = []
for line in namefile:
	line = line.rstrip()
	name = str(path+line)
	name_array.append(name)
namefile.close()

"""
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

Computation function outputs:
mat/arr: dTdz_peaks (z, T)
mat/arr: const_consec (z,T)
"""

# reading files
for i in range(0,len(name_array)):
	
	# reading in file here
	filename = name_array[i]
	print(i,filename)
	read_data(filename)
	bathymetry("../terrainbase.nc")
	plot_data(True)


######################################################################################################
