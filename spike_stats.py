######################################################################################################
"""
This is a section of code that is removed from the hitbottom.py file after being used
"""

# statistics for the spikes (independent)
"""
collect information about what is the best standard deviation threshold (function input) 
to use to get the fewest false detections
The match precision for a "good match" will be set to +- 5m in depth (chosen arbitrarily)
Bad is if no match is detected or if difference is more than 5m
"""
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
"""

######################################################################################################
# importing libraries
import numpy as np


######################################################################################################

""" 
Opening file for reading the fraction of good detections found with each threshold
"""

threshold = []
good = []
bad = []

# reading file
with open("stats.txt") as f:
	next(f) #skipping header
	for line in f:
		line = line.split(',')
		threshold.append(int(line[0]))
		good.append(float(line[1]))
		bad.append(float(line[2].rstrip('\n')))

f.close()

threshold = np.array(threshold)
good = np.array(good)
bad = np.array(bad)

# calculation of the fraction of the hit bottoms that are being identified accurately with the
# temperature spikes method
frac = good/(good+bad)

# writing to terminal
for i in range(0,len(threshold)):
	print("for "+str(threshold[i])+"*sigma threshold, HB identified = "+str(frac[i])+" accuracy.")

"""
Find that the 3 sigma threshold for detections is the most ideal for determining HB
"""


######################################################################################################
