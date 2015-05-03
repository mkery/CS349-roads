import sys


g = open ("combined_results.csv", "r")
g.readline()

drivers = [line.strip().split(",") for line in g]


dnum = 1
d = open ("ans_d1.csv", "w")

for i in range(0,len(drivers)):
	prob = 0
	if float(drivers[i][1]) > 100:
		prob = 1
	#d.write(str(drivers[i][0])+","+str(prob)+"\n")
	d.write(str(prob)+"\n")
	if (i+1)%200 == 0 :
		print(str(i))
		d.close()
		dnum = int(drivers[i+1][0].split("_")[0])
		d = open ("ans_d"+str(dnum)+".csv", "w")

g.close()