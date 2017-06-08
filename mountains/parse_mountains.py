#!/usr/bin/env ipython

import gpxpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from geopy.distance import vincenty

def gps_distance_elevation(fname):
    segment = gpxpy.parse(open(fname + '.gpx', 'r')).tracks[0].segments[0]
    elevation = []
    loc = []
    for p in segment.points:
        elevation.append(p.elevation)
        lat, lon = p.latitude, p.longitude
        loc.append((lat, lon))

    distance = np.array([0] + [vincenty(loc[i], loc[i-1]).meters for i in range(len(loc)-1)]).cumsum()
    plt.plot(distance, elevation, label=fname)
    plt.savefig(fname + '.png')
    plt.clf()
    return distance, elevation

def downsample_mountain(fname, length=30):
    """ Downsample trace to specified length """
    distance, elevation = gps_distance_elevation(fname)
    d = np.linspace(distance[0], distance[-1], length)
    e = np.interp(d, distance, elevation)
    plt.plot(d, e, label=fname)
    plt.savefig(fname + '_downsampled.png')
    plt.clf()
    assert len(d) == length
    assert len(e) == length
    return d, e

def get_mts():
    d_m, e_m = downsample_mountain('tour-mont-blanc-anti-clockwise')
    d_c, e_c = downsample_mountain('carrauntoohil')
    # scale both
    e_m -= np.mean(e_m)
    e_m /= np.max(e_m)
    e_c -= np.mean(e_c)
    e_c /= np.max(e_c)
    # combine
    samples = np.array([e_m, e_c]).reshape(2, -1, 1)
    np.save('mts.npy', samples)
    return samples

