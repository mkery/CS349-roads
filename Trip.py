import matplotlib.pyplot as pyplot
import numpy as np
import os
import sys
import rdp_trip
import math


def distance(x0, y0, x1, y1):
	return math.sqrt((x1-x0)**2 + (y1-y0)**2)

def findSpeed_Dist(trip):
	v = []
	a = []
	dist = 0
	for i in range(1, trip.shape[0]):
		d = distance(trip[i-1][0], trip[i-1][1], trip[i][0], trip[i][1])
		dist += d
		v.append(d)

	return v,dist

def findStops(speeds):
	stops = [] #stops are a start and end time pair
	start = -1
	end = -1
	for i in range(1, len(speeds)):
		advS = (speeds[i] + speeds[i-1])/2 #smooth out noise in stop duration
		if speeds[i] == 0: #start of stop
			end = i
			if start == -1:
				start = i
		elif start > -1 and advS > 1: 
			stops.append([start,end])
			start = -1
			end = -1
	if start > -1:
		stops.append([start, len(speeds)])
	return stops


class Trip(object):

	def __init__(self, filename):

		#read in trip from file
	 	tripPath = np.genfromtxt(filename, delimiter=',', skip_header=1)
	 	#add a column for time in seconds (so if we chop data, still have timepoints)
	 	self.tripPath = np.append(tripPath, np.arange(tripPath.shape[0]).reshape(tripPath.shape[0],1),1)
	 	
	 	self.v, self.tripDist = findSpeed_Dist(tripPath)

	 	self.tripTime = tripPath.shape[0] #length of trip in seconds
	 	self.advSpeed = self.tripDist/self.tripTime #meters per second
	 	self.maxSpeed = max(self.v)

	 	self.stops = len(findStops(self.v))

	def printFeatures(self):
		features = ""
		features += str(self.advSpeed)+","
		features += str(self.tripDist)+"\n"
		return features

	def plotTrip(self):
		#first figure is the xy path
	 	pyplot.figure(1)
		pyplot.subplot(211)
		startPoint = (self.tripPath[0]) 
		pyplot.plot(self.tripPath[:,0], self.tripPath[:,1], 'r-', startPoint[0], startPoint[1], 'bs')
		for st,end in self.stops:
			pyplot.plot(self.tripPath[st][0], self.tripPath[st][1], 'rs')
		#second figure is velocity over time
		pyplot.subplot(212)
		pyplot.plot(self.v, 'g-')
		for st,end in self.stops:
			pyplot.plot(st,self.v[st], 'bs', end, self.v[st], 'rs')
			print end - st
		pyplot.show()


"""doc = sys.argv[1]
trip_test = Trip(doc)
trip_test.plotTrip()"""

#print trip_test.advSpeed


