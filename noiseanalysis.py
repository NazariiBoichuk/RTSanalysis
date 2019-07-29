import numpy as np
#https://www.youtube.com/watch?v=pTrfnHleY0U

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

def RTSanalysis2lvl(time, current, coefSmooth = 1, coefThreshold = 3, coefNeighbour = 1, forceM = 0, forceSt = 0):
    
    #start_pos = 0
    #window_size = 2000
    #current = current[start_pos:start_pos + window_size]
    #time =  time[start_pos:start_pos + window_size]
    current = smoothing(current, coefSmooth)
    #averaged_current = smoothing(current, 11)
    
    #derivative
    dert, derc = derivative(time, current)
    
    #detect jumps comparing the derivatives. When too high - jumps is detected
    #feel amplitude with the size of jump in absolute units
    if (forceM == 0 and forceSt == 0):
        m = np.mean(derc)
        st = np.std(derc)
    else:
        m = forceM
        st = forceSt
    #st does not change a lot if you have enough point, let us say more than 1000
    #if the signal is stationar. AND RTS is

    amplitude = []
    i = 0
    surround = coefNeighbour # comparison with the amplitude current values
    threshold_coef_for_detection = coefThreshold # usually I use 3
    for y in derc:
        if abs(y - m) > threshold_coef_for_detection*st:
            rightedge2 = i+1+surround
            rightedge1 = i+1
            if (rightedge2 > len(current)): rightedge2 = len(current)
            if (rightedge1 >= len(current)): rightedge1 = -1        
            leftedge2 = i
            leftedge1 = i-surround
            if (leftedge1 < 0): 
                leftedge1 = 0
                if (leftedge2 == 0): leftedge2 = 1
            amplitude.append(np.mean(current[rightedge1:rightedge2])-np.mean(current[leftedge1:leftedge2]))
        else:
            amplitude.append(0)
        i += 1
    amplitude = np.array(amplitude)
    
    '''
    #this part seems to be not important. Where is a logic?
    
    #detect only jumps amplitudes in absolute values
    #leave only high amplitudes
    amplheight = []
    for a in amplitude:
        if a != 0:
            amplheight.append(abs(a))
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
            #could also think about idea amplitude[i] = max_ampl
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
    #make only 0-1 level, but not -1-0
    if (np.where(count2levels == -1)[0].size != 0):
        count2levels += 1
    return count2levels

if __name__ == '__main__':
    y = np.loadtxt('T17_Noise_LG_After_plasma_21_timetrace_extracted.dat', unpack = True, skiprows = 1)
    y = y [:10000]
    fp = open('T17_Noise_LG_After_plasma_21_timetrace_extracted.dat', 'r')
    line = fp.readline()
    fp.close()
    if line.find('Fs=') == 0:
        frequency = line[3:]
        Fs = float(frequency)
       
    #add timeline
    x = [i / Fs for i in range(0, len(y))]
    RTSanalysis2lvl(x, y)
