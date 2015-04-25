import matplotlib.pyplot as pyplot
import numpy as np
import os
import sys
import rdp_trip
import math
import Pmf


def distance(x0, y0, x1, y1):
	return math.sqrt((x1-x0)**2 + (y1-y0)**2)

"""def distance_point(pt1, pt2):
	return math.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt2[0])**2)

def dist_point_to_line(pt, ln_a, ln_b):
	l2 = distance_point(ln_a, ln_b)
	if l2 == 0:
		return distance_point(pt, ln_a)

	t = ((pt[0] - ln_a[0]) * (ln_b[0] - ln_a[0]) + (pt[1] - ln_a[1]) * (ln_b[1] - ln_a[1])) / l2
	if t < 0:
		return distance_point(pt, ln_a)
	if t > 1:
		return distance_point(pt, ln_b)
	return distance_point(pt, (ln_a[0] + t * (ln_b[0] - ln_a[0]),   ln_a[1] + t * (ln_b[1] - ln_a[1])))
"""

def pldist(x0, x1, x2):
    """
    Calculates the distance from the point ``x0`` to the line given
    by the points ``x1`` and ``x2``.
    :param x0: a point
    :type x0: a 2x1 numpy array
    :param x1: a point of the line
    :type x1: 2x1 numpy array
    :param x2: another point of the line
    :type x2: 2x1 numpy array
    """

    if x1[0] == x2[0]:
        return np.abs(x0[0] - x1[0])

    return np.divide(np.linalg.norm(np.linalg.det([x2 - x1, x1 - x0])),
                     np.linalg.norm(x2 - x1))


def rdp_simplify(trip, epsilon):
	# find the point with the maximum distance
	dmax = 0
	index = 0
	for i in range(1, trip.shape[0]): #every point but first and last
		d = pldist( np.array([trip[i][0], trip[i][1]]),
					np.array([trip[0][0], trip[0][1]]),
					np.array([trip[-1][0], trip[-1][1]]) )
		if (d > dmax):
			index = i
			dmax = d
	# If max distance is greater than epsilon, recursively simplify
	if (dmax > epsilon):
		#build the result list
		res1 = rdp_simplify(trip[:index+1], epsilon)
		res2 = rdp_simplify(trip[index:], epsilon)
		return np.vstack((res1, res2))
	else:
		return np.vstack((trip[0],trip[-1]))


def findSpeed_Dist(trip):
	v = []
	dist = 0
	for i in range(1, trip.shape[0]):
		d = distance(trip[i-1][0], trip[i-1][1], trip[i][0], trip[i][1])
		dist += d
		v.append(3.6*d)

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

def printHist_Feature(hist):
	h = ""
	for i in range(len(hist)-1):
		h += str(hist[i])+","
	#to avoid final comma (will mess up input)
	h += str(hist[len(hist)-1])	
	return h

class Trip(object):

	def __init__(self, filename):

		#read in trip from file
	 	self.tripPath = np.genfromtxt(filename, delimiter=',', skip_header=1)
	 	#add a column for time in seconds (so if we chop data, still have timepoints)
	 	#self.tripPath = np.append(tripPath, np.arange(tripPath.shape[0]).reshape(tripPath.shape[0],1),1)
	 	
	 	self.rdp = rdp_simplify(self.tripPath, epsilon = 0.75)
	 	print "DONE rdp_simplify!"

	 	#self.v, self.tripDist = findSpeed_Dist(self.tripPath)
	 	self.findSpeed_Hist(self.tripPath)

		self.tripTime = self.tripPath.shape[0] #length of trip in seconds
	 	self.advSpeed = self.tripDist/self.tripTime #meters per second
	 	self.maxSpeed = max(self.v)

	 	self.stops = findStops(self.v)#len(findStops(self.v))

	 	#self.speed_hist, self.acc = findSpeed_Hist(self.tripPath)



	#changed the implementation of this method, which brought the metrics up a bit
	#I used km/h, but we can easily change that
	def findSpeed_Hist(self, trip):
		
		speedList = []
		speedList.append(0)
		accList = []
		accList.append(0)

		for i in range (1,len(self.tripPath)):

			speedList.append (round(3.6*distance(self.tripPath[i-1][0], self.tripPath[i-1][1], self.tripPath[i][0], self.tripPath[i][1])))
			accList.append(speedList[i]-speedList[i-1])

		mypmf = Pmf.MakePmfFromList(speedList)
		self.speed_hist = []
		MAX = 220
		vals, freqs = mypmf.Render()
		val = 0
		for i in range(MAX):

			try:
				val = freqs[vals.index(i)]
			except ValueError:
				val = 0

			self.speed_hist.append (val)

		mypmf = Pmf.MakePmfFromList(accList)
		self.acc_hist = []
		MAX = 50
		vals, freqs = mypmf.Render()
		val = 0
		for i in range(MAX):

			try:
				val = freqs[vals.index(i)]
			except ValueError:
				val = 0

			self.acc_hist.append (val)
		#print self.speed_hist

		#mypmf.Items()
		#sys.exit()
		
		
		vel =  np.diff(trip, axis = 0) #x1-x0 and y1-y0
		self.v = (vel[:,0]**2 + vel[:,1]**2)**0.5 #take distance
		self.tripDist = np.sum(self.v)
		"""
		self.acc = np.diff(self.v, axis = 0)
		self.speed_hist = [np.percentile(self.v, i*5) for i in range(1,20)]
		self.acc_hist = [np.percentile(self.acc, i*10) for i in range(1,10)]
		"""
		

	def printFeatures(self):
		features = ""
		features += printHist_Feature(self.speed_hist)+","
		features += str(self.tripDist)+","
		features += printHist_Feature(self.acc_hist)

		return features + "\n"

	def plotTrip(self):
		#first figure is the xy path
	 	pyplot.figure(1)
		#pyplot.subplot(211)
		startPoint = (self.tripPath[0]) 
		endPoint = (self.tripPath[self.tripPath.shape[0]-1])
		pyplot.plot(self.tripPath[:,0], self.tripPath[:,1], 'rx', startPoint[0], startPoint[1], 'gD', endPoint[0], endPoint[1], 'bD')
		pyplot.plot(self.rdp[:,0], self.rdp[:,1], 'bo')
		for st,end in self.stops:
			pyplot.plot(self.tripPath[st][0], self.tripPath[st][1], 'rs')
		#second figure is velocity over time
		"""pyplot.subplot(212)
		pyplot.plot(self.v, 'g-')
		for st,end in self.stops:
			pyplot.plot(st,self.v[st], 'bs', end, self.v[st], 'rs')
			#print end - st
		pyplot.plot(self.acc, 'b-')"""
		pyplot.show()


trip_test = Trip(sys.argv[1])
trip_test.plotTrip()

#print trip_test.advSpeed


