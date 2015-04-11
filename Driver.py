import matplotlib.pyplot as pyplot
import numpy as np
import sys
import math
from Trip import Trip
import os
import random
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.svm import SVC

num_selfTrips = 120
num_testTrips = 40
num_NOTselfTrips = 200

class Driver(object):

	def __init__(self, driverName):
		self.name = driverName


	def classify(self):
		#get training trips for this driver
		f = open("driver_stats/"+str(self.name)+"_training.csv")
		#f.readline() #skip header labels
		traintrips = np.genfromtxt(f, delimiter=',')
		f.close()

		#get list of labels for the trips in traintrips
		g = open("driver_stats/trainingLabels.csv")
		target = np.genfromtxt(g, delimiter=',')
		g.close()

		#get test trips for this driver
		h = open("driver_stats/"+str(self.name)+"_test.csv")
		testtrips = np.genfromtxt(h, delimiter=',')
		h.close()
		k = open("driver_stats/testLabels.csv")
		test_target = np.genfromtxt(k, delimiter=',')
		k.close() 


		print traintrips.shape, target.shape
		print traintrips[1]
		clf = SVC()
		print clf.fit(traintrips, target)
		print clf.score(testtrips, test_target)
		#joblib.dump(clf, "driver_stats/"+str(self.name)+"_clf.pkl")


	def writeCSV(self):
		g = open ("driver_stats/"+str(self.name)+"_trips.csv", "w")
		#a header and then the features for each trip
		g.write("advSpeed,tripDist\n")
		for i in range (1,num_selfTrips+1):
			t = Trip("../drivers/"+str(self.name)+"/"+str(i)+".csv")
			g.write(t.printFeatures())
		g.close()

	def getRandomDriverTrips(self, numtrips = num_NOTselfTrips):
		notDrivers = os.listdir("../drivers/")
		tripList = []
		for i in range(numtrips):
			dnum = notDrivers[random.randint(1, len(notDrivers) - 1)] #sample a random driver
			while dnum == self.name: #don't sample from self
				dnum = notDrivers[random.randint(1, len(notDrivers) - 1)]
			tnum = random.randint(1,200)#sample a random trip
			t = Trip("../drivers/"+str(dnum)+"/"+str(tnum)+".csv")
			tripList.append(t.printFeatures())
		return tripList

	def writeCSV_notDriver(self):
		#list other drivers in directory, since their numbers skip around
		notDrivers = os.listdir("../drivers/")

		g = open ("driver_stats/"+str(self.name)+"_NOTtrips.csv", "w")
		tripList = self.getRandomDriverTrips()
		for other in tripList:
			g.write(other)
		g.close()

	def writeCSV_training(self):
		g = open ("driver_stats/"+str(self.name)+"_training.csv", "w")
		#a header and then the features for each trip
			#g.write("advSpeed,tripDist\n")
		#first trips from this driver
		for i in range (1,num_selfTrips+1):
			t = Trip("../drivers/"+str(self.name)+"/"+str(i)+".csv")
			g.write(t.printFeatures())
		#trips from other drivers
		tripList = self.getRandomDriverTrips()
		for other in tripList:
			g.write(other)
		g.close()

	def writeCSV_labels(self):
		#file containing training labels, same for any driver
		h = open ("driver_stats/"+"trainingLabels.csv", "w")
		for i in range(num_selfTrips):
			h.write(str(1)+"\n")
		for i in range(num_NOTselfTrips):
			h.write(str(0)+"\n")
		h.close()

	def writeCSV_testlabels(self):
		#file containing test labels, same for any driver
		h = open ("driver_stats/"+"testLabels.csv", "w")
		for i in range(num_testTrips):
			h.write(str(1)+"\n")
		for i in range(num_NOTselfTrips):
			h.write(str(0)+"\n")
		h.close()

	def writeCSV_test(self):
		g = open ("driver_stats/"+str(self.name)+"_test.csv", "w")
		#first trips from this driver
		for i in range (num_selfTrips+1, num_selfTrips+num_testTrips+1):
			t = Trip("../drivers/"+str(self.name)+"/"+str(i)+".csv")
			g.write(t.printFeatures())
		#trips from other drivers
		tripList = self.getRandomDriverTrips()
		for other in tripList:
			g.write(other)
		g.close()



d1 = Driver(sys.argv[1])
d1.writeCSV_test()
d1.writeCSV_training()
d1.classify()


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