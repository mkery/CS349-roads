import numpy as np
import sys
import matplotlib.pyplot as pyplot
import rdp_trip as rdp

driver = sys.argv[1]

for i in range(1,201):
	print "generating rdp for "+str(driver)+" trip "+str(i)
	rdp.generateRDP(str(driver)+"_"+str(i), str(driver), str(i))