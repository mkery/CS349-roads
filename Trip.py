import matplotlib.pyplot as pyplot
import numpy as np
import os
import sys
import math
import Pmf
from scipy.ndimage import gaussian_filter1d


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
	 	self.tripPathRaw = np.genfromtxt(filename, delimiter=',', skip_header=1)
	 	#add a column for time in seconds (so if we chop data, still have timepoints)
	 	#self.tripPath = np.append(tripPath, np.arange(tripPath.shape[0]).reshape(tripPath.shape[0],1),1)
	 	self.tripPath=self.tripPathRaw
	 	#self.v, self.tripDist = findSpeed_Dist(self.tripPath)

		#self.smooth_data()

	 	#self.plotTrip()
	 	self.dV = []
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
		self.turn_dist = [] #turning distance
		self.turn_ang = [] #turning angle
		self.sharp_turn_sp = []


	 	self.computeSpeedAcc()
	 	self.computeTurningAngles()
	 	self.computeTimeLowSpeeds()

	 	self.findTurns()
	 	#self.plotTrip()

	 	self.computeHistograms()

		self.tripTime = self.tripPath.shape[0] #length of trip in hours
	 	self.advSpeed = self.tripDist/self.tripTime #meters per second
	 	self.maxSpeed = max(self.v)

	 	self.stops = findStops(self.v)#len(findStops(self.v))

	 	#self.speed_hist, self.acc = findSpeed_Hist(self.tripPath)

	#code taken from stackexchage http://stackoverflow.com/questions/15178146/line-smoothing-algorithm-in-python
	def smooth_data(self, data):
		#print data
		a = np.array(data)
		x, y = a.T
		t = np.linspace(0, 1, len(x))
		t2 = np.linspace(0, 1, 100)

		x2 = np.interp(t2, t, x)
		y2 = np.interp(t2, t, y)
		sigma = 0.5
		x3 = gaussian_filter1d(x2, sigma)
		y3 = gaussian_filter1d(y2, sigma)

		temp = [(x3[i],y3[i]) for i in range(len(x3))]
		temp = np.array(self.tripPath)
		
		return data


	def computeTurningAngles (self):
		self.dV = []
		for ind in range (1, len(self.tripPath)):
			self.dV.append(((self.tripPath[ind][0]-self.tripPath[ind-1][0]),(self.tripPath[ind][1]-self.tripPath[ind-1][1])))
			
			#not the implementation, subject to change
			try:
				self.ang_or.append(computeAngle((1,0), self.dV[ind-1]))
			except ZeroDivisionError:
				self.ang_or.append(0)

		for ind in range (2, len(self.dV)):
			angle = computeAngle(self.dV[ind-2], self.dV[ind])
			self.ang.append(angle)
			self.ang_sp.append(angle*((self.v[ind-2]+self.v[ind-1]+self.v[ind]))/3)

		
	
	def findTurns (self):
		self.t1 = []
		self.t2 = []
		
		ind = 0
		curr = 1
		th = 50
		tol = 5
		while(ind<len(self.dV) and curr < len(self.dV)):
			#print ind
			#print "-", curr
			#print computeAngle(self.dV[curr], self.dV[ind])
			prev = 0
			while(curr <len(self.dV)):
				ang = computeAngle(self.dV[curr], self.dV[ind])
				if ang > prev:
					prev = ang
					curr += 1
				else:
					if prev > th:
						self.turn_ang.append(prev)
						self.turn_dist.append(self.dist[curr] - self.dist[ind])
						self.sharp_turn_sp.append((self.dist[curr] - self.dist[ind])/(curr-ind+1))
						#print str(ind) + " " + str(curr-1) + " " + str(prev) + " " +str(self.dist[curr]) +" " + str(self.dist[ind]) +" "+ str(self.dist[curr] - self.dist[ind])
						self.t1.append(self.tripPath[ind][0])
						self.t2.append(self.tripPath[ind][1])
						self.t1.append(self.tripPath[curr][0])
						self.t2.append(self.tripPath[curr][1])

					break
				#print "*", prev
			if curr == len(self.dV):
				break
			if ind == curr-1:
				ind = ind+1
				curr = ind+1
			else:
				ind = curr-1
		#print self.turn_ang
		#print self.turn_dist

		#print self.sharp_turn_sp
	

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
			self.dist.append(self.tripDist)
			self.bee_dist.append(distance(self.tripPath[i][0], self.tripPath[i][1], 0, 0))
		
		temp = [(z, self.acc[z]) for z in range(len(self.acc))]
		temp = self.smooth_data(temp)
		temp = [temp[i][1] for i in range(len(temp))]
		self.jerk = np.diff(temp)

		#print self.dist
	

	def computeTimeLowSpeeds (self):
		#0.05, 0.1, 0.15, 0.2, 0.25
		self.low_sp_count = [0 for i in range(6)]
		perc = [np.percentile (self.v, j*0.05) for j in range (6)]
		for i in range(len(self.v)):
			for j in range (6):
				if (self.v[i]<perc[j]):
					self.low_sp_count[j]+=1
					break


	def computeHistograms(self):
		b = 5

		self.ang_hist = self.computeHist(b, self.ang)
		self.ang_sp_hist = self.computeHist(b, self.ang_sp)
		self.ang_or_hist = self.computeHist(b, self.ang_or)
		self.speed_hist = self.computeHist(b, self.v)
		self.acc_hist = self.computeHist(b, self.acc)
		self.v_a_hist = self.computeHist(b, self.v_a)
		self.jerk_hist = self.computeHist(b,self.jerk)
		self.dist_hist = self.computeHist(b, self.dist)
		self.bee_dist_hist = self.computeHist(b, self.bee_dist)
		self.turn_ang_hist = self.computeHist(b, self.turn_ang)
		self.turn_dist_hist = self.computeHist(b, self.turn_dist)
		self.sharp_turn_sp_hist = self.computeHist(b, self.sharp_turn_sp)
		#print self.sharp_turn_sp_hist

		

	def computeHist(self, b, data):
		if np.array(data).shape[0] == 0:
			return [0 for i in range(0,100/b+3)]
		hist = [np.percentile(data, i*b) for i in range(0,100/b+1)]
		mean = np.mean(data)
		stdev = np.std(data)
		hist.append(mean)
		hist.append(stdev)
		return hist


	def printFeatures(self):
		features = ""
		#features += str(self.tripDist)+","
		#features += str (self.advSpeed) + ","
		#features += str(self.maxSpeed) + ","
		features += printHist_Feature(self.speed_hist)+"," #1
		features += printHist_Feature(self.acc_hist) + "," #2
		features += printHist_Feature(self.ang_hist) + "," #3
		features += printHist_Feature(self.ang_sp_hist) + "," #4
		features += printHist_Feature(self.v_a_hist) + "," #5
		features += printHist_Feature(self.ang_or_hist) +"," #6
		features += printHist_Feature(self.low_sp_count) + "," #7
		features += printHist_Feature(self.jerk_hist) + "," #8
		#features += printHist_Feature(self.dist_hist) + "," #9
		features += printHist_Feature(self.bee_dist_hist) + "," #10 
		features += printHist_Feature(self.turn_ang_hist) + "," #11
		features += printHist_Feature(self.turn_dist_hist) +"," #12
		features += printHist_Feature(self.sharp_turn_sp_hist) + "," #13

		return features[:-1] + "\n"


	def plotTrip(self):
		#first figure is the xy path
	 	pyplot.figure(1)
		pyplot.subplot(211)
		startPoint = (self.tripPathRaw[0]) 
		endPoint = (self.tripPathRaw[self.tripPathRaw.shape[0]-1])
		pyplot.plot(self.tripPathRaw[:,0], self.tripPathRaw[:,1], 'r-', startPoint[0], startPoint[1], 'gD', endPoint[0], endPoint[1], 'bD', self.t1, self.t2, "kx")
		

		pyplot.subplot(212)
		startPoint = (self.tripPath[0]) 
		endPoint = (self.tripPath[self.tripPath.shape[0]-1])
		pyplot.plot(self.tripPath[:,0], self.tripPath[:,1], 'r-', startPoint[0], startPoint[1], 'gD', endPoint[0], endPoint[1], 'bD')
		

		"""
		for st,end in self.stops:
			pyplot.plot(self.tripPath[st][0], self.tripPath[st][1], 'rs')
		#second figure is velocity over time
		
		
		pyplot.plot(self.v, 'g-')
		for st,end in self.stops:
			pyplot.plot(st,self.v[st], 'bs', end, self.v[st], 'rs')
			#print end - st
		pyplot.plot(self.acc, 'b-')
		"""
		pyplot.show()


"""trip_test = Trip(sys.argv[1])
trip_test.plotTrip()

print trip_test.advSpeed"""

t = Trip("../drivers/2/100.csv")

