import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import scipy.signal
from scipy.optimize import curve_fit
import random
def smoothing(data, window):
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

current = np.loadtxt("T04_RTS_PBS_0_timetrace_extracted.dat", unpack = True, skiprows = 1) 
Fs = 10000
time = [i / Fs for i in range(0, len(current))]

averaged_current = smoothing(current, 15)

temp_seq = current[4*Fs:5*Fs]
m = np.mean(temp_seq)
st = np.std(temp_seq)
print(m)
print(st)
points = []
for y in current:
    if abs(y - m) < 4 * st:
        points.append(1e-5)
    else:
        points.append(-2e-5)

plt.plot(time, current, 'red')
plt.plot(time, averaged_current, 'blue')
plt.plot(time, points, 'go-')
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