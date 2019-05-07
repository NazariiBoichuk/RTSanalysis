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

def derivative(datax, datay):
    dydx = np.diff(datay) / np.diff(datax)
    dx = (np.array(datax)[:-1] + np.array(datax)[1:]) / 2
    return dx, dydx


#try to analize only a part
current = np.loadtxt("T04_RTS_PBS_1_timetrace_extracted.dat", unpack = True, skiprows = 1)
x =  0
#cut only when work in IDE
current = current[x:x+10000]
Fs = 10000
time = [i / Fs for i in range(0, len(current))]

averaged_current = smoothing(current, 7)
smoothcoef = 1
current = smoothing(current, smoothcoef)

#derivative
dert, derc = derivative(time, current)


plt.plot(time, current, 'red')
plt.plot(time, averaged_current, 'blue')
m = np.mean(derc)
st = np.std(derc)
scale = np.std(current) / np.max(derc)
print(m)
print(st)
high_der = []
amplitude = []
amplheigh = []
i = 0
surround = smoothcoef
for y in derc:
    if abs(y - m) > 3*st:
        high_der.append(1)
        amplitude.append(np.mean(current[i+1:i+1+surround]-current[i-surround:i]))
    else:
        high_der.append(0)
        amplitude.append(0)
    i += 1
high_der = np.array(high_der)
amplitude = np.array(amplitude)
for a in amplitude:
    if a != 0:
        amplheigh.append(abs(a))

for i in range(0, len(amplitude)):
    if abs(abs(amplitude[i])-np.mean(amplheigh))>3*np.std(amplheigh):
        amplitude[i] = 0

last = 0
i = 0

while i < len(amplitude):
    if (amplitude[i] != 0):
        j = i
        while(amplitude[j] != 0):
            j += 1
        max_ampl = max(amplitude[i:j])
        max_i = np.argmax(amplitude[i:j])+i
        for k in range(i,j):
            amplitude[k] = 0
        amplitude[max_i] = max_ampl
        i = j
        continue

    else:
        i += 1

plt.plot(dert, derc * scale + 9*scale, 'green')

plt.plot(dert, amplitude + 3*scale, 'red')

plt.plot(dert, high_der * scale + 6 * scale, 'g.-')
plt.show()




'''
#how to use histogram
BINS = 30
plt.hist(averaged_current, bins=BINS)  # arguments are passed to np.histogram
hist, bin_edges = np.histogram(averaged_current, bins = BINS)
plt.xlim(bin_edges[0], bin_edges[-1])
EnergyMaxCount = max(hist)
EnergyMaxIndex = list(hist).index(EnergyMaxCount)
EnergyMax = (bin_edges[EnergyMaxIndex]+bin_edges[EnergyMaxIndex+1])/2
plt.show()
'''
