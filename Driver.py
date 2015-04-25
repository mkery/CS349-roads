import matplotlib.pyplot as pyplot
import numpy as np
import sys
import math
from Trip import Trip
import os
import random
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.svm import SVC
import random

num_selfTrips = 120
num_testTrips = 80
num_NOTselfTrips = 200


class Driver(object):

	def __init__(self, driverName):
		self.name = driverName


	#we might have to change the rounding at some point, but it's a good way to start
	def calculateResults(self,predicted, true):
		tp = 0
		tn = 0
		fp = 0
		fn = 0

		for i in range (len(true)):
			if (true[i] == 1 and round(predicted[i]) == 1):
				tp+=1
			if (true[i] == 1 and round(predicted[i]) == 0):
				fn+=1
			if (true[i] == 0 and round(predicted[i]) == 1):
				fp+=1
			if (true[i] == 0 and round(predicted[i]) == 0):
				tn+=1

		#print tp, tn, fp, fn
		prec = float(tp)/(tp+fp)
		recall = float(tp)/(tp+fn)
		acc = float (tp+tn)/(tp+tn+fp+fn)
		#print 'Precision: ', prec
		#print 'Recall: ', recall

		return (prec, recall, acc)

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



		print "driver ", self.name
		clf = RandomForestClassifier(n_estimators=30, criterion='entropy', max_depth=15,  n_jobs=1, min_samples_leaf=5,verbose=1)
		print clf.fit(traintrips, target)
		print clf.score(traintrips, target), "train"
		print clf.score(testtrips, test_target), "test"

		#print traintrips.shape, target.shape
		#print traintrips[1]
		#clf = RandomForestRegressor() #LogisticRegression()
		#clf.fit(traintrips, target)
		clf = RandomForestClassifier() #LogisticRegression()
		clf.fit(traintrips, target)
		#print clf.score(traintrips, target)


		predLabels = clf.predict (testtrips)

		#print predLabels
		#print test_target

		return self.calculateResults(predLabels, test_target)
		
		#print clf.score(testtrips, test_target)

		#joblib.dump(clf, "driver_stats/"+str(self.name)+"_clf.pkl")

	def writeCSV(self):
		g = open ("driver_stats/"+str(self.name)+"_trips.csv", "w")
		#a header and then the features for each trip
		g.write("advSpeed,tripDist\n")
		for i in range (1,num_selfTrips+1):
			t = Trip("../drivers/"+str(self.name)+"/"+str(i)+".csv")
			g.write(t.printFeatures())
		g.close()

	def getRandomDriverTrips(self, numtrips):
		notDrivers = os.listdir("../drivers/")
		random.shuffle(notDrivers[1:])
		numNotDrivers = len(notDrivers) #change this parameter to consider a different number
		tripList = []
		for i in range(numtrips):
			dnum = notDrivers[random.randint(1, len(notDrivers) - 1)] #sample a random driver
			while dnum == self.name: #don't sample from self
				dnum = notDrivers[random.randint(1, numNotDrivers - 1)]
			tnum = random.randint(1,200)#sample a random trip
			t = Trip("../drivers/"+str(dnum)+"/"+str(tnum)+".csv")
			tripList.append(t.printFeatures())
		return tripList

	def writeCSV_notDriver(self):
		#list other drivers in directory, since their numbers skip around
		notDrivers = os.listdir("../drivers/")

		g = open ("driver_stats/"+str(self.name)+"_NOTtrips.csv", "w")
		tripList = self.getRandomDriverTrips(num_NOTselfTrips)
		for other in tripList:
			g.write(other)
		g.close()

	def writeCSV_training(self, order):
		g = open ("driver_stats/"+str(self.name)+"_training.csv", "w")
		#a header and then the features for each trip
			#g.write("advSpeed,tripDist\n")
		#first trips from this driver
		for i in range (1,num_selfTrips+1):
			t = Trip("../drivers/"+str(self.name)+"/"+str(order[i])+".csv")
			g.write(t.printFeatures())
		#trips from other drivers
		tripList = self.getRandomDriverTrips(num_NOTselfTrips)
		for other in tripList:
			g.write(other)
		g.close()

	#TODO: extend to both training and testing labels 
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
		for i in range(num_testTrips):
			h.write(str(0)+"\n")
		h.close()

	def writeCSV_test(self, order):
		g = open ("driver_stats/"+str(self.name)+"_test.csv", "w")
		#first trips from this driver
		for i in range (num_selfTrips, num_selfTrips+num_testTrips):
			t = Trip("../drivers/"+str(self.name)+"/"+str(order[i])+".csv")
			g.write(t.printFeatures())
		#trips from other drivers
		tripList = self.getRandomDriverTrips(num_testTrips)
		for other in tripList:
			g.write(other)
		g.close()

	def createDataSets(self):
		order = [i for i in range(1, 201)]
		random.shuffle(order)
		self.writeCSV_training(order)
		self.writeCSV_labels()
		self.writeCSV_test(order)
		self.writeCSV_testlabels()



d1 = Driver(sys.argv[1])
d1.createDataSets()
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