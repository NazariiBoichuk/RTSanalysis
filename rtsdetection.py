# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:33:01 2019

@author: n.boichuk
"""
import numpy as np
import matplotlib.pyplot as plt


def smooth(x,window_len=11):
    #window_len only be odd
    add_points = window_len // 2
    print(add_points)
    y = 0
    return y    
'''
current = np.loadtxt("T04_RTS_PBS_24_timetrace_extracted.dat", skiprows = 1)
Fs = 10000 # should be loaded from file
time = [x / Fs for x in range(0, len(current))]
#sizes are same
print(len(current))
print(len(time))
data = np.vstack((time, current)).T
print(data)
sm = smooth(current, window_len = 300, window='flat')
#plt.ioff()
print(len(sm))
plt.plot(time, current)
plt.plot(time, sm)
plt.show()
'''
smooth([1,1,1,1,1,1,1,2], 3)