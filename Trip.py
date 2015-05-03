import matplotlib.pyplot as pyplot
import numpy as np
import os
import sys
import math
import Pmf


def computeNorm(x, y):
	return distance(x, y, 0, 0)

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

def computeAngle (p1, p2):
	dot = 0
	if computeNorm(p2[0], p2[1]) == 0 or computeNorm(p1[0], p1[1])==0: #may be incorrect
		dot = 0
	else:
		dot = (p2[0]*p1[0]+p2[1]*p1[1])/float(computeNorm(p1[0], p1[1])*computeNorm(p2[0], p2[1])) 

	if dot > 1:
		dot = 1
	elif dot < -1:
		dot = -1

	return math.acos(dot)*180/math.pi



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
	 	
	 	#self.v, self.tripDist = findSpeed_Dist(self.tripPath)

	 	#features computed so far
	 	self.v = [] #speed at each second
		self.acc = [] #acceleration at each second
		self.v_a = [] #velocity*acceleration = horsepower/mass
		self.jerk = [] #change of acceleration over a second
		self.ang = [] #change of angle over 3 seconds
		self.ang_sp = [] #turning agression = speed*angle
		self.ang_or = [] #angle from the initial vector
		self.low_sp_count = [] #time at low speed (below 0.25)
		self.dist = [] #distance driven up to a particular point
		self.bee_dist = [] #bee flight distance


	 	self.computeSpeedAcc()
	 	self.computeTurningAngles()
	 	self.computeTimeLowSpeeds()


	 	self.findSpeed_Hist()
	 	self.findAngle_Hist()

		self.tripTime = self.tripPath.shape[0] #length of trip in hours
	 	self.advSpeed = self.tripDist/self.tripTime #meters per second
	 	self.maxSpeed = max(self.v)

	 	self.stops = findStops(self.v)#len(findStops(self.v))

	 	#self.speed_hist, self.acc = findSpeed_Hist(self.tripPath)

	def computeTurningAngles (self):
		dV = []
		for ind in range (1, len(self.tripPath)):
			dV.append(((self.tripPath[ind][0]-self.tripPath[ind-1][0]),(self.tripPath[ind][1]-self.tripPath[ind-1][1])))
			
			#not the implementation, subject to change
			try:
				self.ang_or.append(computeAngle((1,0), dV[ind-1]))
			except ZeroDivisionError:
				self.ang_or.append(0)
			#print ind
		#t.append([])
		#t.append([])
		#t.append([])

		for ind in range (2, len(dV)):
			angle = computeAngle(dV[ind-2], dV[ind])
			self.ang.append(angle)
			self.ang_sp.append(angle*((self.v[ind-2]+self.v[ind-1]+self.v[ind]))/3)
			#t[0].append(ind)
			#t[1].append(angle)
			#t[2].append((v[ind-2]+v[ind-1]+v[ind])/3)


	def computeSpeedAcc(self):
		
		self.tripDist = 0
		self.v.append(0)
		self.acc.append(0)
		self.v_a.append(0)
		self.jerk.append(0)
		self.dist.append(0)
		self.bee_dist.append(0)

		for i in range (1,len(self.tripPath)):
			curr = distance(self.tripPath[i-1][0], self.tripPath[i-1][1], self.tripPath[i][0], self.tripPath[i][1])
			self.tripDist += curr
			self.v.append(curr)
			self.acc.append(self.v[i]-self.v[i-1])
			self.v_a.append(self.v[i]*self.acc[i])
			self.jerk.append(self.acc[i]-self.acc[i-1])
			self.dist.append(self.tripDist)
			self.bee_dist.append(distance(self.tripPath[i][0], self.tripPath[i][1], 0, 0))


	def computeTimeLowSpeeds (self):
		#0.05, 0.1, 0.15, 0.2, 0.25
		self.low_sp_count = [0 for i in range(6)]
		perc = [np.percentile (self.v, j*0.05) for j in range (6)]
		for i in range(len(self.v)):
			for j in range (6):
				if (self.v[i]<perc[j]):
					self.low_sp_count[j]+=1
					break


	def findAngle_Hist(self):
		ba = 5
		bas = 5
		bao = 5
		self.ang_hist = [np.percentile(self.ang, i*ba) for i in range(1,100/ba+1)]
		self.ang_sp_hist = [np.percentile(self.ang_sp, i*bas) for i in range(1,100/bas+1)]
		self.ang_or_hist = [np.percentile(self.ang_or, i*bao) for i in range(1,100/bao+1)]

	def findSpeed_Hist(self):
		bv = 5
		ba = 5
		bva = 5
		bj = 5
		bd = 5
		bdb = 5 
		self.speed_hist = [np.percentile(self.v, i*bv) for i in range(1,100/bv+1)]
		self.acc_hist = [np.percentile(self.acc, i*ba) for i in range(1,100/ba+1)]
		self.v_a_hist = [np.percentile(self.v_a, i*bva) for i in range(1,100/bva+1)]
		self.jerk_hist = [np.percentile(self.jerk, i*bj) for i in range(1,100/bj+1)]
		self.dist_hist = [np.percentile(self.dist, i*bd) for i in range(1,100/bd+1)]
		self.bee_dist_hist = [np.percentile(self.bee_dist, i*bd) for i in range(1,100/bd+1)]

	def printFeatures(self):
		features = ""
		features += str(self.tripDist)+","
		features += str (self.advSpeed) + ","
		features += str(self.maxSpeed) + ","
		#-features += printHist_Feature(self.speed_hist)+","
		#-features += printHist_Feature(self.acc_hist) + ","
		#-features += printHist_Feature(self.ang_hist) + ","
		#-features += printHist_Feature(self.ang_sp_hist) + ","
		#-features += printHist_Feature(self.v_a_hist) + ","
		#features += printHist_Feature(self.ang_or_hist) +","
		#-features += printHist_Feature(self.low_sp_count) + ","
		#-features += printHist_Feature(self.jerk_hist) + ","
		#features += printHist_Feature(self.dist_hist) + ","
		#features += printHist_Feature(self.bee_dist_hist) + "," 

		return features[:-1] + "\n"

	def plotTrip(self):
		#first figure is the xy path
	 	pyplot.figure(1)
		pyplot.subplot(211)
		startPoint = (self.tripPath[0]) 
		endPoint = (self.tripPath[self.tripPath.shape[0]-1])
		pyplot.plot(self.tripPath[:,0], self.tripPath[:,1], 'r-', startPoint[0], startPoint[1], 'gD', endPoint[0], endPoint[1], 'bD')
		for st,end in self.stops:
			pyplot.plot(self.tripPath[st][0], self.tripPath[st][1], 'rs')
		#second figure is velocity over time
		pyplot.subplot(212)
		pyplot.plot(self.v, 'g-')
		for st,end in self.stops:
			pyplot.plot(st,self.v[st], 'bs', end, self.v[st], 'rs')
			#print end - st
		pyplot.plot(self.acc, 'b-')
		pyplot.show()


"""trip_test = Trip(sys.argv[1])
trip_test.plotTrip()

print trip_test.advSpeed"""


