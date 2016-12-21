######################################################################################################
# function to plot the text files with statistics

import numpy as np
import pandas as pd
import math
import os.path
import matplotlib.pyplot as plt
from netCDF4.utils import ncinfo
from mpl_toolkits.mplot3d import Axes3D


######################################################################################################
# reading in files and creating plots

# constant temperature stats
dat = []
with open('stats_const_temp.txt') as f:
	next(f)
	for line in f:
		dat.append([float(x.strip()) for x in line.split(',')])
		
above = []
below = []
total = []
frac = []
domain1 = [] # consecutive points
domain2 = [] # detection threshold values
for i in range(0,len(dat)):
	if ((dat[i][2]+dat[i][3]) != 0):
		fraction = dat[i][3]/(dat[i][2]+dat[i][3])
	else: 
		fraction = 0
	frac.append(fraction)
	domain1.append(dat[i][0])
	domain2.append(dat[i][1])
	above.append(dat[i][2])
	below.append(dat[i][3])
	total.append(dat[i][2]+dat[i][3])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(domain1,np.log(domain2),frac)
plt.title("Constant Temperature - good detection percentage")
plt.xlabel("Consecutive points")
plt.ylabel("Detection threshold")
plt.show()

f.close()


######################################################################################################

# temperature increase stats
dat = []
with open('stats_temp_increase.txt') as f:
	next(f)
	for line in f:
		dat.append([float(x.strip()) for x in line.split(',')])
		
above = []
below = []
total = []
frac = []
domain = []
for i in range(0,len(dat)):
	fraction = dat[i][2]/(dat[i][1]+dat[i][2])
	frac.append(fraction)
	domain.append(dat[i][0])
	above.append(dat[i][1])
	below.append(dat[i][2])
	total.append(dat[i][1]+dat[i][2])

# plotting bar graph of the number of detections against total number
plt.bar(domain, total, width=10, alpha=0.5)
plt.bar(domain, below, width=10, alpha=0.5)
plt.title("Temperature increase - Number of points below HB")
plt.ylabel("Fraction of points below expected HB")
plt.xlabel("Number of consecutive points required (for detection)")
plt.show()

plt.plot(domain,frac)
plt.title("Temperature Increase - Percentage of bad points below HB detected")
plt.xlabel("Consecutive points required (for detection)")
plt.ylabel("Percentage (below HB points/total points)")
plt.show()

f.close()


######################################################################################################
