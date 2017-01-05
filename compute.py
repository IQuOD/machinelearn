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
import math
import os.path
import matplotlib.pyplot as plt
from netCDF4.utils import ncinfo
from netCDF4 import Dataset
import hitbottom as hb


######################################################################################################
# computation using code from hitbottom.py

# filename generation
path = "../HBfiles/"

# taking sample of files from the name file
namefile = open("subsetHB.txt","r")
name_array = []
for line in namefile:
	line = line.rstrip()
	name = str(path+line)
	name_array.append(name)
namefile.close()

"""
code to take in the data and find the optimal values for the inputs to the functions

Most ideal cases:
grad_spike: threshold = 3
T_spike: threshold = ?
temp_increase: consec_points = ?
const_temp: consec_points = ? threshold = ?
"""

######################################################################################################
# code for collecting statistics on the data (optimisation)	

"""
# grad_spike optimisation 
f = open('stats_grad_spike.txt','w')
f.write('sigma,first_detect,total_close\n')
print("Optimisation process for grad_spike function parameters")
range_vals = np.arange(1,9,1)
n = len(range_vals)

# changing through detection threshold values
for j in range(0,n):
	detect_threshold = range_vals[j]
	print("outer, detection threshold="+str(detect_threshold))
	count_first = 0
	count_overall = 0
	
	# reading files
	for i in range(0,len(name_array)):
	
		# reading in file here
		filename = name_array[i]

		print(i,filename)
		[data, gradient, flags, hb_depth, latitude, longitude, date] = hb.read_data(filename)
		[bath_height, bath_lon, bath_lat] = hb.bathymetry("../terrainbase.nc")
		
		# computing the points that will be used to compare
		points = hb.grad_spike(data,gradient,detect_threshold)	
		if (type(points) == int):
			continue
		else:
			if (len(points) > 1):
				for k in range(0,len(points)): 
					if (abs(points[k][0]-hb_depth) < 5):
						count_overall = count_overall + 1
						if (k == 0):
							count_first = count_first + 1
					else: 
						continue
			else:
				try:
					if (abs(points[0][0]-hb_depth) < 5):
						count_overall = count_overall + 1
						count_first = count_first + 1
					else:
						continue
				except:
					pass
		
	# recording information
	f.write(str(detect_threshold)+","+str(count_first)+","+str(count_overall)+"\n")

f.close()


# T_spike optimisation 
f = open('stats_T_spike.txt','w')
f.write('threshold,first_detect,total_close\n')
print("Optimisation process for T_spike function parameters")
range_vals = np.logspace(-4,-1,4)
n = len(range_vals)

# changing through detection threshold values
for j in range(0,n):
	detect_threshold = range_vals[j]
	print("outer, detection threshold="+str(detect_threshold))
	count_first = 0
	count_overall = 0
	
	# reading files
	for i in range(0,len(name_array)):
	
		# reading in file here
		filename = name_array[i]
		print(i,filename)
		[data, gradient, flags, hb_depth, latitude, longitude, date] = hb.read_data(filename)
		[bath_height, bath_lon, bath_lat] = hb.bathymetry("../terrainbase.nc")
		
		# computing the points that will be used to compare
		points = hb.T_spike(data,detect_threshold)	
		if (len(points) > 1):
			for k in range(0,len(points)): 
				if (abs(points[k][0]-hb_depth) < 5):
					count_overall = count_overall + 1
					if (k == 0):
						count_first = count_first + 1
				else: 
					continue
		else:
			try:
				if (abs(points[0][0]-hb_depth) < 5):
					count_overall = count_overall + 1
					count_first = count_first + 1
				else:
					continue
			except:
				pass
		
	# recording information
	f.write(str(detect_threshold)+","+str(count_first)+","+str(count_overall)+"\n")

f.close()


# temp_increase optimisation
f = open('stats_temp_increase.txt','w')
f.write('num_consec,above,below\n')
print("Optimisation process for temp_increase function parameters")
consec_range = np.arange(50,270,20)
m = len(consec_range)

# changing through detection threshold values
for j in range(0,m):
	consec_points = consec_range[j]
	print("outer, consecutive points="+str(consec_points))
	above = 0
	below = 0
	
	# reading files
	for i in range(0,len(name_array)):
	
		# reading in file here
		filename = name_array[i]
		print(i,filename)
		[data, gradient, flags, hb_depth, latitude, longitude, date] = hb.read_data(filename)
		[bath_height, bath_lon, bath_lat] = hb.bathymetry("../terrainbase.nc")
		
		# computing the points that will be used to compare
		points = hb.temp_increase(data,consec_points)	
		for k in range(0,len(points)):
			if (points[k][0] < hb_depth):
				above = above + 1
			else:
				below = below + 1
		
	# recording information
	f.write(str(consec_points)+","+str(above)+","+str(below)+"\n")

f.close()


# const_temp optimisation
f = open('stats_const_temp.txt','w')
f.write("consec_pts,threshold,above,below\n")
print("Optimisation process for const_temp function parameters")
consec_range = np.arange(50,270,20)
threshold_range = np.logspace(-5,-1,5)
m1 = len(consec_range)
m2 = len(threshold_range)

# changing through detection threshold values
for ii in range(0,m1):
	consec_points = consec_range[ii]
	print("outer, consecutive points="+str(consec_points))
	for j in range(0,m2):
		threshold = threshold_range[j]
		print("more outer, threshold="+str(threshold))
		above = 0
		below = 0
	
		# reading files
		for i in range(0,len(name_array)):
	
			# reading in file here
			filename = name_array[i]
			print(i,filename)

			[data, gradient, flags, hb_depth, latitude, longitude, date] = hb.read_data(filename)
			[bath_height, bath_lon, bath_lat] = hb.bathymetry("../terrainbase.nc")
		
			# computing the points that will be used to compare
			points = hb.const_temp(data, gradient, consec_points, threshold)	
			for k in range(0,len(points)):
				if (points[k][0] < hb_depth):
					above = above + 1
				else:
					below = below + 1
		
		# recording information
		f.write(str(consec_points)+","+str(threshold)+","+str(above)+","+str(below)+"\n")	

f.close()
"""

######################################################################################################
# code for assessing success of program

# writing to file
f = open("hb_wpothb_success.txt","w")
f.write("bad_points,fraction,fracAbove,num_detected,frac_detected\n")

bad_points = np.arange(25,205,5)
fraction = np.arange(0.02,1,0.02)
fracAbove = np.arange(0.02,0.5,0.02)

m1 = len(bad_points)
m2 = len(fraction)
m3 = len(fracAbove)

for j1 in range(0,m1):
	print("points ="+str(bad_points[j1]))
	for j2 in range(0,m2):
		print("fraction="+str(fraction[j2]))
		for j3 in range(0,m3):
			print("fracAbove="+str(fracAbove[j3]))
			# reading files
			n = len(name_array)
			for i in range(0,n):

				# reading in file here
				filename = name_array[i]
				count = 0
				predict_correctly = 0

				print(i,filename)
				[data, gradient, flags, hb_depth, latitude, longitude, date] = hb.read_data(filename)
				[bath_height, bath_lon, bath_lat] = hb.bathymetry("../terrainbase.nc")
	
				# function inputs
				const = hb.const_temp(data, gradient, 100, 0.001)
				inc = hb.temp_increase(data, 50)
				error_pts = hb.concat(const, inc)
				Tspike = hb.T_spike(data, 0.05)
				dTspike = hb.grad_spike(data, gradient, 3)
				pot_hb = hb.concat(Tspike,dTspike)

				# counting script and printing to check
				predict_depth = hb.hit_predict(data, error_pts, pot_hb, bad_points[j1], fraction[j2], fracAbove[j3], hb_depth)

				if (predict_depth == hb_depth):
					continue

				count = count + 1
				print(abs(predict_depth - hb_depth))
				if (abs(predict_depth - hb_depth) < 10):
					predict_correctly = predict_correctly + 1

			# collecting key stats
			num_detected = predict_correctly
			frac_detected = num_detected/count
			f.write(str(bad_points[j1])+","+str(fraction[j2])+","+str(fracAbove[j3])+","+str(num_detected)+","+str(frac_detected))

f.close()

######################################################################################################
