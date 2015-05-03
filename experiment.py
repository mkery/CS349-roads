import os
import sys
from Driver import Driver
import random

drivers = os.listdir("../drivers/")
copy = drivers[1:]
random.shuffle(copy)
drivers[1:]=copy
#print drivers[1:20]
#print drivers
m = int(sys.argv[1])
n = int(sys.argv[2])
g = open (sys.argv[3], "w")
g.write("driver,precision,recall,f1,auc\n")
g.close()

for i in range(1, m+1):
	g = open (sys.argv[3], "a")
	#print drivers[i]
	d = Driver(drivers[i])
	for j in range(n):
		d.createDataSets()
		results = d.classify()
		for res in results:
			f1 = 2*(res[0]*res[1])/(res[0]+res[1])
			g.write (drivers[i] + ","+ str(res[0]) + "," + str(res[1]) + "," +str(f1) +"," + str(res[2])+'\n')
	#sys.exit()
	g.close()



