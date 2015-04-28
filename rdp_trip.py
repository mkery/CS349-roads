import numpy as np
import sys
import matplotlib.pyplot as pyplot

"""
edited 4/25 to fit trip default numpy format
If you import a trip and then add a 3rd column to the trip that is time,
when this runs the time field is kept... a bit hacky but works.

"""

def dist_point_to_line(x0, x1, x2):
    """ Calculates the distance from the point ``x0`` to the line given
    by the points ``x1`` and ``x2``, all numpy arrays """

    if x1[0] == x2[0]:
        return np.abs(x0[0] - x1[0])

    return np.divide(np.linalg.norm(np.linalg.det([x2 - x1, x1 - x0])),
                     np.linalg.norm(x2 - x1))


def rdp_simplify(trip, epsilon):
    # find the point with the maximum distance
    dmax = 0
    index = 0
    for i in range(1, trip.shape[0]): #every point but first and last
        d = dist_point_to_line( np.array([trip[i][0], trip[i][1]]),
                    np.array([trip[0][0], trip[0][1]]),
                    np.array([trip[-1][0], trip[-1][1]]) )
        if (d > dmax):
            index = i
            dmax = d
    # If max distance is greater than epsilon, recursively simplify
    if (dmax > epsilon):
        #build the result list
        res1 = rdp_simplify(trip[:index+1], epsilon)
        res2 = rdp_simplify(trip[index:], epsilon)
        return np.vstack((res1[:-1], res2)) #not sure why [:-1] works, but it keeps duplicates from happening
    else:
        return np.vstack((trip[0],trip[-1]))

def rdp_expand(trip, triplen):
    xs = trip[:,0]
    ys = trip[:,1]
    times = trip[:,2]

    xs_interp = np.interp(range(0,triplen), times, xs)
    ys_interp = np.interp(range(0,triplen), times, ys)
    return np.append(xs_interp.reshape(xs_interp.shape[0],1), ys_interp.reshape(ys_interp.shape[0],1), 1)


"""filename = sys.argv[1]
tripPath = np.genfromtxt(filename, delimiter=',', skip_header=1)
#add a column for time in seconds (so if we chop data, still have timepoints)
tripPath = np.append(tripPath, np.arange(tripPath.shape[0]).reshape(tripPath.shape[0],1),1)
rdp = rdp_simplify(tripPath, epsilon = 0.75)
rdp_ex = rdp_expand(rdp, tripPath.shape[0])
print "original: " + str(tripPath.shape) + " rdp expanded: " + str(rdp_ex.shape)


pyplot.figure(1)
pyplot.plot(tripPath[:,0], tripPath[:,1], 'rx')
pyplot.plot(rdp[:,0], rdp[:,1], 'bo')
pyplot.plot(rdp_ex[:,0], rdp_ex[:,1], 'g-')
pyplot.plot(rdp_ex[:,0], rdp_ex[:,1], 'go')

np.savetxt("rdp_test.csv", rdp_ex, delimiter=",")

#pyplot.show()"""