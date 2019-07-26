# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:33:01 2019
@author: n.boichuk
For console part
D:
cd D:\Boichuk\PROGRAMMING\RTSanalysis
C:\ProgramData\Anaconda3\python.exe rtsdetection.py
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import scipy.signal
from scipy.optimize import curve_fit
import random

def findFiles(folder):
    '''Returns sorted list of files that are in the folder'''
    files = []
    for root, dirs, file in os.walk(folder):  
        files = file
        break
    files.sort()
    return files

# my function of smoothing, based on averaging with nearest neighbours
# in the range of window. If window is too big (for members at the edges)
# window is adjusted
def smoothing(data, window = 1):
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
    #first method going forward
    #dydx = np.diff(datay) / np.diff(datax)
    #dx = (np.array(datax)[:-1] + np.array(datax)[1:]) / 2
    
    #second method combining symethrical formula + for edges
    dx = datax
    dydx = [0 for i in range(0, len(datay))]
    dydx[0] = (datay[1] - datay[0]) / (datax[1] - datax[0])
    dydx[-1] = (datay[-1] - datay[-2]) / (datax[-1] - datax[-2])
    for i in range (1, len(datay)-1):
        dydx[i] = (datay[i+1] - datay[i-1]) / (datax[i+1] - datax[i-1])
    return np.array(dx), np.array(dydx)

def histoplot(data, bins_number = 10):
    plt.hist(data, bins=bins_number)  # arguments are passed to np.histogram
    hist, bin_edges = np.histogram(data, bins = bins_number)
    plt.xlim(bin_edges[0], bin_edges[-1])
    plt.yscale('log')
    #plt.yscale('log')
    dataMaxCount = max(hist)
    dataMaxIndex = list(hist).index(dataMaxCount)
    #dataMax = (bin_edges[dataMaxIndex]+bin_edges[dataMaxIndex+1])/2
    plt.show()

def readTimeTrace(filename_to_analyse):
    current = np.loadtxt(filename_to_analyse, unpack = True, skiprows = 1)
    fp = open(filename_to_analyse, 'r')
    line = fp.readline()
    fp.close()
    Fs = 1
    if line.find('Fs=') == 0:
        frequency = line[3:]
        Fs = float(frequency)
    time = [i / Fs for i in range(0, len(current))]
    return time, current

def RTSanalysis2lvl(time, current, coefSmooth = 1, coefThreshold = 3):
    start_pos = 0
    window_size = 2000
    current = current[start_pos:start_pos + window_size]
    time =  time[start_pos:start_pos + window_size]
    current = smoothing(current,coefSmooth)
    #averaged_current = smoothing(current, 11)
    
    #derivative
    dert, derc = derivative(time, current)
    
    #detect jumps comparing the derivatives. When too high - jumps is detected
    #feel amplitude with the size of jump in absolute units
    m = np.mean(derc)
    st = np.std(derc)
    
    amplitude = []
    amplheight = []
    i = 0
    surround = 2 # comparison with the amplitude current values
    threshold_coef_for_detection = coefThreshold # usually I use 3
    for y in derc:
        if abs(y - m) > threshold_coef_for_detection*st:
            #ValueError: operands could not be broadcast together with shapes (2,) (0,)
            #when i =1,2...W
            rightedge2 = i+1+surround
            rightedge1 = i+1
            if (rightedge2 > len(current)): rightedge2 = len(current)
            if (rightedge1 >= len(current)): rightedge1 = -1        
            leftedge2 = i+1
            leftedge1 = i-surround
            if (leftedge1 < 0): leftedge1 = 0
            amplitude.append(np.mean(current[rightedge1:rightedge2])-np.mean(current[leftedge1:leftedge2]))
        else:
            amplitude.append(0)
        i += 1
    amplitude = np.array(amplitude)
    
    #detect only jumps amplitudes in absolute values
    #leave only high amplitudes
    for a in amplitude:
        if a != 0:
            amplheight.append(abs(a))
    '''
    m = np.mean(amplheight)
    st = np.std(amplheight)
    for i in range(0, len(amplitude)):
        if abs(abs(amplitude[i])-m) > 3*st:
            amplitude[i] = 0
    '''
    #leave only one close amplitude to have it as a signal jump
    #neighbours are summed to get an absolute value of jump
    last = 0 #last sign to detect changes
    i = 0
    while i < len(amplitude):
        if (amplitude[i] != 0):
            last = np.sign(amplitude[i])
            j = i
            while(np.sign(amplitude[j]) == last):
                j += 1
                if j == len(amplitude): break
            max_ampl = sum(amplitude[i:j])
            max_i = np.argmax(amplitude[i:j])+i
            for k in range(i,j):
                amplitude[k] = 0
            amplitude[max_i] = max_ampl
            i = j
            continue
        else:
            i += 1
    
    #form a signal detection levels
    countlevels=[0 for i in range(0, len(current))]
    for i in range(1, len(amplitude)):
        countlevels[i] = countlevels[i-1] + np.sign(amplitude[i])
    countlevels = np.array(countlevels)
     
    #to form a two-state signal detection levels
    count2levels=[0 for i in range(0, len(countlevels))]
    last = 0 # direction of changes
    for i in range(1, len(countlevels)):
        delta = countlevels[i] - countlevels[i-1]
        if delta == 0:
            count2levels[i] = count2levels[i-1]
        else:
            if np.sign(delta) == last:
                count2levels[i] = count2levels[i-1]
            else:
                count2levels[i] = count2levels[i-1] + np.sign(amplitude[i])
            last = np.sign(amplitude[i])
    count2levels = np.array(count2levels)
    return count2levels
    ''' 
    #plt.subplot('411')
    plt.plot(time, current, 'red')
    #plt.plot(time, averaged_current, 'blue')
    
    #plt.subplot('412')
    scale = (max(current)-min(current)) /(max(derc)-min(derc))
    shift = 1.2 * (max(current)-min(current)) 
    plt.plot(dert, derc* scale + 1 *shift, 'g*-')
    #plt.subplot('413')
    scale = (max(current)-min(current)) /(max(amplitude)-min(amplitude))
    plt.plot(dert, amplitude * scale + 2*shift, 'r*-')
    #plt.subplot('414')
    scale = (max(current)-min(current)) /(max(countlevels)-min(countlevels))
    plt.plot(dert, countlevels * scale - 3* shift, 'b.-')
    
    scale = (max(current)-min(current)) /(max(count2levels)-min(count2levels))
    plt.plot(dert, count2levels * scale - 1* shift, 'b.-')
    plt.show()
    '''
    '''
    folder = filename_to_analyse[:-4]
    try:
        os.mkdir(folder)
    except:
        pass
    sl = int(Fs)
    scale = 0.25 * (max(current)-min(current)) /(max(count2levels)-min(count2levels))
    shift = 1.2 * (max(current)-min(current)) 
    plt.clf()
    plt.plot(time[sl:sl+1000], current[sl:sl+1000], 'red')
    plt.plot(dert[sl:sl+1000], count2levels[sl:sl+1000] * scale - 0.75* shift, 'b,-')
    plt.savefig(folder + '/testpic1.png')
    sl = 2 * int(Fs)
    plt.clf()
    plt.plot(time[sl:sl+1000], current[sl:sl+1000], 'red')
    plt.plot(dert[sl:sl+1000], count2levels[sl:sl+1000] * scale - 0.75* shift, 'b,-')
    plt.savefig(folder + '/testpic2.png')
    sl = 3 * int(Fs)
    plt.clf()
    plt.plot(time[sl:sl+1000], current[sl:sl+1000], 'red')
    plt.plot(dert[sl:sl+1000], count2levels[sl:sl+1000] * scale - 0.75* shift, 'b,-')
    plt.savefig(folder + '/testpic3.png')
    
    #count2levels is working nicely!
    #withdrow time constants
    
    t1 = list()
    t2 = list()
    lasttimechange = 0
    for i in range(1, len(countlevels)):
        delta = count2levels[i] - count2levels[i-1]
        if delta != 0:
            if np.sign(delta) > 0:
                t1.append(dert[i] - dert[lasttimechange])
            else:
                t2.append(dert[i] - dert[lasttimechange])
            lasttimechange = i
            
    
    with open(folder + '/Time1.dat', 'w') as f:
        f.write("Averaged = %s\n" % np.mean(t1))
        for item in t1:
            f.write("%s\n" % item)
    with open(folder + '/Time2.dat', 'w') as f:
        f.write("Averaged = %s\n" % np.mean(t2))
        for item in t2:
            f.write("%s\n" % item)  
    with open(folder + '/Ampl.dat', 'w') as f:
        for item in amplitude:
            f.write("%s\n" % item)
    with open(folder + '/Binary.dat', 'w') as f:
        for item in count2levels:
            f.write("%s\n" % item) 
    '''
    '''
    BINS = 35
    histoplot(t1, BINS)
    histoplot(t2, BINS)
    
    plt.subplot('211')
    f, Pxx = scipy.signal.periodogram(current, Fs)
    plt.plot(f[1::], smoothing(Pxx[1::],31))
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.subplot('212')
    plt.plot(time, current)
    plt.show()
    '''

filename_to_analyse = 'T17_Noise_LG_After_plasma_21_timetrace_extracted.dat'
time, current = readTimeTrace(filename_to_analyse)
twolevel = RTSanalysis2lvl(time, current)