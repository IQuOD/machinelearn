######################################################################################################
"""
This is a script to check the the profiles for hit bottoms based on:
 - large spikes in the temperature gradient (over 5 sigma)
 - TBA
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
	f = open(filename,'r')
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
		spikes = spike(data,gradient, 3)
		if (type(spikes) != int):
			plt.plot(spikes[:,1], spikes[:,0],'ro')
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
	

######################################################################################################
# Computation functions
"""
This section of code is a logic checking implementation to attempt 
to detect hit bottoms for XBT drops.
"""

# finding characteristic gradient increase spike for hit bottoms
def spike(data, gradient, threshold):
	"""
	Threshold is the number of standard deviations away from the mean gradient value you 
	want to use for identifying a spike
	Using large positive gradient spikes to identify whether or not there is a peak
	Wire breaks tend to have high temperature after gradient peak, so remove those
	with higher temp after peak
	Note that this is only to identify clear, large spikes in grad (one property of HB data)
	"""
	# importing key data from dataframes
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
			return(0)
	else:
		return(0)

# statistics for the spikes (independent)
"""
collect information about what is the best standard deviation threshold (function input) 
to use to get the fewest false detections
The match precision for a "good match" will be set to +- 5m in depth (chosen arbitrarily)
Bad is if no match is detected or if difference is more than 5m
"""
def get_stats(data, gradient, threshold):
	
	# determining the depth of actual HB flag:
	for i in range(0,len(flags.flag)):
			if (flags.flag[i] == "HB"):
				HB_depth = flags.depth[i]
			else:
 				continue
	
	# calculate peak accuracy (spike function returns int 0 if no peak found)
	spikes = spike(data, gradient, threshold) 
	if (type(spikes) != int):
		spike_depth = spikes[0][0]
		difference = HB_depth - spike_depth
	else:
		difference = 999
	
	# categorising (1=good, 0=bad)
	if (abs(difference) < 5):
		spike_stats = 1
	else:
		spike_stats = 0

	return(spike_stats)


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
Data that is available here after reading in the data (for each profile):
df: flags (flags, depth)
mat: data (z, T)
mat: gradient (z, dTdz)
mat: dT9pt (z, T_av)
mat: secDer (d, d2Tdz2)
var: latitude
var: longitude 
var: date

Computation function outputs:
mat/arr: dTdz_peaks (z, T)
"""

# writing statistics data to file
f = open('stats.txt','w')
f.write("sigma,good,bad\n")
good_spike = []
bad_spike = []
domain = []
	
# reading files
# collecting stats
for j in range(2,6):
	good = 0
	bad = 0
	print("Main loop: "+str(j))
	
	for i in range(0,len(name_array)):
		
		# reading in file here
		filename = name_array[i]
		print(i,filename)
		read_data(filename)
		#plot_data(False)
		#spike(data, gradient, 3)
	
		# collecting statistics about spikes
		if (get_stats(data, gradient, j) == 1):
			good = good + 1
		else: 
			bad = bad + 1
	
	good_spike.append(good)
	bad_spike.append(bad)
	domain.append(j)

	# printing header data
	"""
	print(str(filename)+"data:")
	print("latitude: "+str(latitude))
	print("longitude: "+str(longitude))
	print("date: "+str(date))
	print("flags:",flags.flag)
	print("\n")
	"""
	f.write(str(j)+","+str(good)+","+str(bad)+"\n")

width = 0.8
plt.bar(domain,good_spike,width,alpha=0.8,label="good",color='g')
plt.bar(domain,bad_spike,width,alpha=0.3,label="bad")
plt.xlabel("sigma")
plt.ylabel("Number of profiles")
plt.title("Good and bad profiles with threshold chosen")
plt.show()

f.close()


	
######################################################################################################
