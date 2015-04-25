import numpy as np
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