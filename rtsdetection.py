# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:33:01 2019

@author: n.boichuk
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import scipy.signal
from scipy.optimize import curve_fit
import random

# my function of smoothing, based on averaging with nearest neighbours
# in the range of window. If window is too big (for members at the edges)
# window is adjusted
def smoothing(data, window = 3):
    #window is only odd
    delta = (window - 1) // 2
    averaged_data = data.copy()
    for i in range(0, len(data)):
        y_temp = 0
        number = 0
        for shift in range(-delta, delta+1):
            if (i + shift >= 0 and i + shift < len(data)):
                number += 1
                y_temp += data[i + shift]
        averaged_data[i] = y_temp / number
    return averaged_data


#try to analize only a part
current = np.loadtxt("T04_RTS_PBS_5_timetrace_extracted.dat", unpack = True, skiprows = 1)
x =  6000
current = current[x:x+1000]
Fs = 10000
time = [i / Fs for i in range(0, len(current))]

averaged_current = smoothing(current, 15)

#derivative
derc = np.diff(current) / np.diff(time)
dert = (np.array(time)[:-1] + np.array(time)[1:]) / 2

m = np.mean(current)
st = np.std(current)
print(m)
print(st)
points = []
for y in current:
    if (y - m) > 0:
        points.append(5e-5)
    else:
        #print('yes')
        points.append(-5e-5)

plt.plot(time, current, 'red')
plt.plot(time, averaged_current, 'blue')
dercsmooth = smoothing(derc/0.5e4+20e-5, 1)
m = np.mean(dercsmooth)
st = np.std(dercsmooth)
print(m)
print(st)
points = []
for y in dercsmooth:
    if abs(y - m) > 3*st:
        points.append(-5e-5)
    else:
        #print('yes')
        points.append(-10e-5)
plt.plot(dert, dercsmooth, 'green')
plt.plot(time[1:], points, 'g.-')
plt.show()




'''
BINS = 30
plt.hist(averaged_current, bins=BINS)  # arguments are passed to np.histogram
hist, bin_edges = np.histogram(averaged_current, bins = BINS)
plt.xlim(bin_edges[0], bin_edges[-1])
EnergyMaxCount = max(hist)
EnergyMaxIndex = list(hist).index(EnergyMaxCount)
EnergyMax = (bin_edges[EnergyMaxIndex]+bin_edges[EnergyMaxIndex+1])/2
plt.show()
'''
