##########################################################################
# libraries

import numpy as np
import pandas as pd
import math
import os.path
import matplotlib.pyplot as plt
from netCDF4.utils import ncinfo
from netCDF4 import Dataset


##########################################################################
# defined functions

# function to read in data from cdf files
def read_data(filename, plot):
	"""
	Plot profiles if the "plot" input is equal to 1. For anything else, it will 
	not plot the profile of temperature and depth		
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
	global data, gradient, n

	# taking temp and depth data
	depth = []
	temp = []
	m = len(init_depth)
	for i in range(0,m):
		if (init_temp[i] < 0)|(init_temp[i]>50):
			continue
		else:
			depth.append(init_depth[i])
			temp.append(init_temp[i])
	n = len(depth)
	# writing all data to data frame
	data = pd.DataFrame({"Depth":depth,"Temperature":temp})

	# taking temperature gradient and depth data
	dTdz = []
	depth_grad = []
	for jj in range(0,n-1):
		depth_grad.append(data.Depth[jj])
		dT = data.Temperature[jj+1]-data.Temperature[jj]
		dz = data.Depth[jj+1]-data.Depth[jj]
		dTdz.append(dT/dz)
	gradient = pd.DataFrame({"z":depth_grad,"dTdz":dTdz})

	"""
	# looking at the second derivative
	d2Tdz2 = []
	depth_secgrad = []
	global secDer
	for jj in range(0,n-2):
		depth_secgrad.append(gradient.z[jj])
		der = float((gradient.dTdz[jj+1]-gradient.dTdz[jj])/(gradient.z[jj+1]-gradient.z[jj]))
		d2Tdz2.append(der)
	secDer = pd.DataFrame({"z":depth_secgrad,"d2Tdz2":d2Tdz2})
	
	# close file
	dat.close()	
	"""

	# taking a moving average for the temperature values
	global dT9pt
	depth_9pt = []
	temp9pt = []
	n = len(gradient.z)
	for jj in range(4,n-4):
		depth_9pt.append(gradient.z[jj])
		Tav = (gradient.dTdz[jj-4]+gradient.dTdz[jj-3]+gradient.dTdz[jj-2]+gradient.dTdz[jj-1]+gradient.dTdz[jj]+gradient.dTdz[jj+1]+gradient.dTdz[jj+2]+gradient.dTdz[jj+3]+gradient.dTdz[jj+4])/float(9.0)
		temp9pt.append(Tav)
	T9pt = pd.DataFrame({"z":depth_9pt,"T_av":temp9pt})

	# conditional subplot of temperature and gradient
	if (plot == True):
		# plotting temperature
		plt.figure(figsize=(11.5,9))
		plt.subplot(1,3,1)
		plt.plot(data.Temperature,data.Depth)
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
		plt.plot(gradient.dTdz, gradient.z)
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
		# plotting temperature gradient
		plt.subplot(1,3,3)
		plt.plot(T9pt.T_av, T9pt.z)
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

############################################
# Functions to identify features of the data

# finding positive temperature increase spike at hit bottom
#def spike(temperature, depth):
	

##########################################################################
# computation

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

# reading files
for i in range(0,len(name_array)):
	
	# reading in file here
	filename = name_array[i]
	read_data(filename, 1)

	print(str(filename))
	print("latitude: "+str(latitude))
	print("longitude: "+str(longitude))
	print("flags:",flags.flag)
	print("\n")

	
##########################################################################
