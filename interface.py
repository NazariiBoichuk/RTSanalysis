import matplotlib.pyplot as plt
import matplotlib.backend_bases
from matplotlib.widgets import Slider, Button
import numpy as np
#https://www.youtube.com/watch?v=pTrfnHleY0U

win_size = 0.1

def on_press(event):
    global win_size
    global x
    global y
    print('you pressed', event.button, event.xdata, event.ydata)
    if (event.inaxes == axTimeTrace):

        
        axZoom.set_xlim(event.xdata,event.xdata+win_size)

        mainFig.canvas.flush_events()
        event.canvas.draw()

             
def val_update(val):
    global win_size
    global x
    global y
    win_size = sl1ZoomSize.val
    x1, x2 = axZoom.get_xlim()
    xav = x1/2+x2/2
    x1 = xav - win_size/2
    x2 = xav + win_size/2
    axZoom.clear()
    axZoom.set_xlim(x1, x2)
    axZoom.plot(x,y)   
    axZoom.plot(x,RTSanalysis2lvl(x,y,coefSmooth = int(sl3AmpWin.val), coefThreshold = sl2Threshold.val)*(max(y)-min(y)))
    plt.draw()

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

def RTSanalysis2lvl(time, current, coefSmooth = 1, coefThreshold = 3, coefNeighbour = 1):
    
    #start_pos = 0
    #window_size = 2000
    #current = current[start_pos:start_pos + window_size]
    #time =  time[start_pos:start_pos + window_size]
    current = smoothing(current,coefSmooth)
    #averaged_current = smoothing(current, 11)
    
    #derivative
    dert, derc = derivative(time, current)
    
    #detect jumps comparing the derivatives. When too high - jumps is detected
    #feel amplitude with the size of jump in absolute units
    m = np.mean(derc)
    st = np.std(derc)
    print(m, st)
 
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
    
mainFig = plt.figure()
mainFig.suptitle('Click the position on timetrace for zooming in')
axTimeTrace = mainFig.add_subplot(211)
#x = list(range(0,100))
#y = [math.sin(x[i]) for i in range (0,100)]
y = np.loadtxt('T17_Noise_LG_After_plasma_21_timetrace_extracted.dat', unpack = True, skiprows = 1)
fp = open('T17_Noise_LG_After_plasma_21_timetrace_extracted.dat', 'r')
line = fp.readline()
fp.close()
if line.find('Fs=') == 0:
    frequency = line[3:]
    Fs = float(frequency)
   
#add timeline
x = [i / Fs for i in range(0, len(y))]

axTimeTrace.plot(x[::10],y[::10])
axZoom = mainFig.add_subplot(212)
plt.subplots_adjust(left=0.1, bottom=0.35)




axSliderZoomSize = plt.axes([0.1, 0.2, .8, 0.05])
sl1ZoomSize = Slider(ax = axSliderZoomSize,
             label = 'Windows Size',
             valmin = 0.001,
             valmax = 1,
             valinit=0.1,
             valfmt='%1.2f',
             closedmax=True)

axSliderThreshold = plt.axes([0.1, 0.1, 0.8, 0.05])
sl2Threshold = Slider(ax = axSliderThreshold, 
                      label = 'Threshold', 
                      valmin = 0.5, 
                      valmax = 5,
                      valinit = 3,
                      valfmt = '%1.2f',
                      closedmax = True)

axSliderAmpWin = plt.axes([0.1, 0.0, 0.8, 0.05])
sl3AmpWin = Slider(ax = axSliderAmpWin, 
                      label = 'Amplitude windows', 
                      valmin = 1, 
                      valmax = 10,
                      valinit = 2,
                      closedmax = True)

mainFig.canvas.mpl_connect('button_press_event', on_press)
sl1ZoomSize.on_changed(val_update)
sl2Threshold.on_changed(val_update)

sl3AmpWin.on_changed(val_update)
plt.show()