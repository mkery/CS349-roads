import matplotlib.pyplot as pyplot
import numpy as np
import sys
import math
from Trip import Trip

class Driver(object):

	def __init__(self, driverName):
		self.name = driverName
		
		f = open("driver_stats/"+str(self.name)+"_trips.csv")
		f.readline() #skip header labels
		myTrips = np.genfromtxt(f, delimiter=',')
		f.close()

		n_samples, n_features = myTrips.shape

	def writeCSV(self, numTrips=60):
		g = open ("driver_stats/"+str(self.name)+"_trips.csv", "w")
		#a header and then the features for each trip
		g.write("advSpeed,tripDist\n")
		for i in range (1, numTrips+1):
			t = Trip("../drivers/"+str(self.name)+"/"+str(i)+".csv")
			g.write(t.printFeatures())
		g.close()




d1 = Driver(sys.argv[1])
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