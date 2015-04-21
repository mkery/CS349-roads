import os
import sys
from Driver import Driver

drivers = os.listdir("../drivers/")
#print drivers
n = int(sys.argv[1])
g = open (sys.argv[2], "w")
g.close()

for i in range(1, len(drivers)):
	g = open (sys.argv[2], "a")
	#print drivers[i]
	d = Driver(drivers[i])
	g.write(drivers[i] + ":")
	for j in range(n):
		d.createDataSets()
		res = d.classify()
		g.write (str(res[0]) + " " + str(res[1]) + " ")

	g.write("\n")
	#sys.exit()
	g.close()



