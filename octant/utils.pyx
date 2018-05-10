# TODO: docstrings!
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport pi, sin, cos, acos


cdef double EARTH_RADIUS = 6371009.  # in metres


cdef double _great_circle(double lon1,
                          double lon2,
                          double lat1,
                          double lat2,
                          double r=EARTH_RADIUS):
    """
    See the docstring for great_circle()
    """

    cdef double deg2rad = pi / 180.
    cdef double ang
    cdef double dist
    cdef double eps = 1e-12

    if abs(lon1-lon2) < eps and abs(lat1-lat2) < eps:
        dist = 0.
    else:
        ang = (sin(deg2rad * lat1) * sin(deg2rad * lat2) +
               cos(deg2rad * lat1) * cos(deg2rad * lat2)
               * cos(deg2rad * (lon1 - lon2)))
        dist = acos(ang) * r
    return dist


cpdef double great_circle(double lon1,
                          double lon2,
                          double lat1,
                          double lat2,
                          double r=EARTH_RADIUS):
    """
    Calculate great circle distance between two points on a sphere

    Arguments
    ---------
    lon1: double
        Longitude of the first point
    lat1: double
        Latitude of the first point
    lon2: double
        Longitude of the second point
    lat2: double
        Latitude of the second point
    r: double, optional (default: EARTH_RADIUS)
        Radius of the sphere in metres
    Returns
    -------
    dist: double
        Distance in metres
    """

    return _great_circle(lon1, lon2, lat1, lat2, r=r)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double total_dist(double[:, ::1] lonlat):
    """
    Calculate the total distance given an array of longitudes and latitudes

    Arguments
    ---------
    lonlat: double, shape(N, 2)
        Array of longitudes and latitudes

    Returns
    -------
    dist: double
        Total distance in metres
    """
    cdef int p
    cdef int pmax = lonlat.shape[0]
    cdef double dist

    dist = 0.
    for p in range(pmax-1):
        dist = dist + _great_circle(lonlat[p, 0], lonlat[p+1, 0],
                                    lonlat[p, 1], lonlat[p+1, 1])
    return dist


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:, ::1] density_grid_rad(double[:, ::1] lon2d,
                                      double[:, ::1] lat2d,
                                      double[:, ::1] lonlat,
                                      double[:, ::1] count,
                                      double rad):
    cdef int i, j, p
    cdef int jmax = lat2d.shape[0]
    cdef int imax = lon2d.shape[1]
    cdef int pmax = lonlat.shape[0]
    for p in range(pmax):
        for j in range(jmax):
            for i in range(imax):
                if _great_circle(lonlat[p, 0], lon2d[j, i],
                                 lonlat[p, 1], lat2d[j, i]) <= rad:
                    count[j, i] = count[j, i] + 1
    return count


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:, ::1] density_grid_each(double[:, ::1] lon2d,
                                       double[:, ::1] lat2d,
                                       double[:, ::1] lonlat,
                                       double[:, ::1] count):
    cdef int i, j, p
    cdef int jmax = lat2d.shape[0]-1
    cdef int imax = lon2d.shape[1]-1
    cdef int pmax = lonlat.shape[0]

    for p in range(pmax):
        for j in range(jmax):
            for i in range(imax):
                if ((lon2d[j, i  ] <= lonlat[p, 0])
                and (lon2d[j, i+1] >  lonlat[p, 0])
                and (lat2d[j, i  ] >  lonlat[p, 1])
                and (lat2d[j+1, i] <= lonlat[p, 1])):
                    count[j, i] = count[j, i] + 1
    return count


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _masking_loop_func(double[:, ::1] mask,
                               double[:, ::1] lon2d,
                               double[:, ::1] lat2d,
                               double lon,
                               double lat,
                               double rad):
    cdef int i, j
    cdef int jmax = lat2d.shape[0]
    cdef int imax = lon2d.shape[1]

    for j in range(jmax):
        for i in range(imax):
            if _great_circle(lon, lon2d[j, i],
                             lat, lat2d[j, i]) <= rad:
                if mask[j, i] == 1:
                    return 1.
    return 0.


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double mask_tracks(double[:, ::1] mask,
                         double[:, ::1] lon2d,
                         double[:, ::1] lat2d,
                         double[:, ::1] lonlat,
                         double rad):
    """
    Count how many points of a cyclone track should be masked by their
    proximity to masked values in a 2D array.

    Arguments
    ---------
    mask: double, shape(M, N)
        Mask array with 1 for masked values
    lon2d: double, shape(M, N)
        Array of longitudes corresponding to the mask
    lat2d: double, shape(M, N)
        Array of latitudes corresponding to the mask
    lonlat: double, shape(P, 2)
        Array of track's longitudes and latitudes
    rad: double
        Radius to check proximity

    Returns
    -------
        Fraction of masked points of the track
    """

    cdef int p
    cdef int pmax = lonlat.shape[0]
    cdef double points_near_coast

    points_near_coast = 0.
    for p in range(pmax):
        points_near_coast += _masking_loop_func(mask, lon2d, lat2d,
                                                lonlat[p, 0], lonlat[p, 1],
                                                rad)
    return points_near_coast / pmax


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double _traj_variance(double[:] x1,
                           double[:] y1,
                           double[:] t1,
                           double[:] x2,
                           double[:] y2,
                           double[:] t2,
                           double alpha=1.,
                           double beta=100):

    cdef int imax1 = x1.shape[0]
    cdef int imax2 = x2.shape[0]
    cdef int i1, i2
    cdef double variance_sum

    variance_sum = 0
    for i1 in range(imax1):
        for i2 in range(imax2):
            variance_sum += ( alpha * (_great_circle(x1[i1], x2[i2],
                                                     y1[i1], y2[i2]) ** 2)
                             + beta * ((t1[i1] - t2[i2])) ** 2 )
    return variance_sum


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
cpdef double distance_metric(double[:] x1,
                             double[:] y1,
                             long[:] t1,
                             double[:] x2,
                             double[:] y2,
                             long[:] t2,
                             double alpha=1.,
                             double beta=100.):

    cdef double dm
    cdef double sigma11
    cdef double sigma12
    cdef double sigma22
    cdef double A1
    cdef double A2
    cdef int imax1 = x1.shape[0]
    cdef int imax2 = x2.shape[0]
    cdef double nano_s = 1e-9
    h_np1 = np.zeros([imax1], dtype=np.double)
    h_np2 = np.zeros([imax2], dtype=np.double)
    cdef double[:] t1_s = h_np1
    cdef double[:] t2_s = h_np2

    for i1 in range(imax1):
        t1_s[i1] = <double>t1[i1] * nano_s
    for i2 in range(imax2):
        t2_s[i2] = <double>t2[i2] * nano_s

    A1 = t1_s[imax1-1] - t1_s[0]
    A2 = t2_s[imax2-1] - t2_s[0]

    sigma12 = _traj_variance(x1, y1, t1_s, x2, y2, t2_s, alpha=alpha, beta=beta) # / (A1 * A2)
    sigma11 = _traj_variance(x1, y1, t1_s, x1, y1, t1_s, alpha=alpha, beta=beta) # / (A1 * A1)
    sigma22 = _traj_variance(x2, y2, t2_s, x2, y2, t2_s, alpha=alpha, beta=beta) # / (A2 * A2)

    dm = ((sigma12 - 0.5 * (sigma11 + sigma22)) / (A1 * A2)) ** 0.5

    return dm
