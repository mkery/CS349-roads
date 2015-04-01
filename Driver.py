import matplotlib.pyplot as pyplot
import numpy as np
import sys
import math
from Trip import Trip
import os
import random


training_self = 120
training_others = 200

class Driver(object):

	def __init__(self, driverName):
		self.name = driverName
		
		#get training trips from self driver
		f = open("driver_stats/"+str(self.name)+"_trips.csv")
		f.readline() #skip header labels
		traintrips = np.genfromtxt(f, delimiter=',')
		f.close()

		#add training trips randomly sampled from other drivers
		f = open("driver_stats/"+str(self.name)+"_NOTtrips.csv")
		traintrips = np.append(traintrips, np.genfromtxt(f, delimiter=','), axis=0)

		#make a list of labels for the trips in traintrips
		target = np.empty(training_self)
		target.fill(1) #1 is self driver
		ot = np.empty(training_others)
		ot.fill(0) #0 is not from this driver
		target = np.append(target, ot)

		print traintrips.shape, target


	def writeCSV(self, numTrips=training_self):
		g = open ("driver_stats/"+str(self.name)+"_trips.csv", "w")
		#a header and then the features for each trip
		g.write("advSpeed,tripDist\n")
		for i in range (1, numTrips+1):
			t = Trip("../drivers/"+str(self.name)+"/"+str(i)+".csv")
			g.write(t.printFeatures())
		g.close()

	def writeCSV_notDriver(self, numTrips = training_others):
		#list other drivers in directory, since their numbers skip around
		notDrivers = os.listdir("../drivers/")

		g = open ("driver_stats/"+str(self.name)+"_NOTtrips.csv", "w")
		for i in range(numTrips):
			dnum = notDrivers[random.randint(1, len(notDrivers))] #sample a random driver
			while dnum == self.name: #don't sample from self
				dnum = notDrivers[random.randint(1, len(notDrivers))]
			tnum = random.randint(1,201)#sample a random trip
			t = Trip("../drivers/"+str(dnum)+"/"+str(tnum)+".csv")
			g.write(t.printFeatures())
		g.close()





d1 = Driver(sys.argv[1])
#d1.writeCSV()

"""d2 = Driver(sys.argv[2])

pyplot.hist(d1.advSpeed, 10, color='blue')
pyplot.hist(d2.advSpeed, 10, color='red')
pyplot.show()"""

"""
fig = pyplot.figure()
f_dist = fig.add_subplot(111)
f_dist.scatter(d1.distance, d1.advSpeed, c='b')
f_dist.scatter(d2.distance, d2.advSpeed, c='r')
pyplot.show()"""