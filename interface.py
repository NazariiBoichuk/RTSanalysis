import matplotlib.pyplot as plt
import matplotlib.backend_bases
from matplotlib.widgets import Slider, Button
import numpy as np
import os
#https://www.youtube.com/watch?v=pTrfnHleY0U
from noiseanalysis import RTSanalysis2lvl, readTimeTrace, derivative

win_size = 0.1
file_path = ''
mean = 0
stdev = 0

def on_press(event):
    global win_size
    print('you pressed', event.button, event.xdata, event.ydata)
    if (event.inaxes == axTimeTrace):
        axZoom.set_xlim(event.xdata - win_size/2,event.xdata + win_size/2)
        mainFig.canvas.flush_events()
        event.canvas.draw()
        val_update(event)
    if (event.inaxes == axZoom):
        axZoom.set_xlim(event.xdata - win_size/2,event.xdata + win_size/2)
        mainFig.canvas.flush_events()
        event.canvas.draw()

def zoom_update(val):
    global win_size
    global x
    global y
    win_size = 10 ** sl1ZoomSize.val / 1000
    x1, x2 = axZoom.get_xlim()
    xav = x1/2+x2/2
    x1 = xav - win_size/2
    x2 = xav + win_size/2
    axZoom.set_xlim(x1, x2)
             
def val_update(val):
    global win_size
    global x
    global y
    win_size = 10 ** sl1ZoomSize.val / 1000
    x1, x2 = axZoom.get_xlim()
    xav = x1/2+x2/2
    x1 = xav - win_size/2
    x2 = xav + win_size/2
    axZoom.clear()
    axZoom.set_xlim(x1, x2)
    if (len(x) != 0):
        if (x1 < x[0]): x1 = x[0]
        if (x2 > x[-1]): x2 = x[-1]
        i = 0
        while (x[i] < x1): i += 1
        left = i
        i = 0
        while (x[i] < x2): i += 1
        right = i
        x_temp = x[left:right]
        y_temp = y[left:right]
        #x_temp = x
        #sy_temp = y
        axZoom.plot(x_temp,y_temp)   
        if (len(x_temp) > 1):
            axZoom.plot(x_temp,
                        (RTSanalysis2lvl(x_temp,
                                         y_temp,
                                         coefNeighbour = int(sl3AmpWin.val), 
                                         coefThreshold = sl2Threshold.val,
                                         coefSmooth = int(sl4Smooth.val),
                                         forceM = mean,
                                         forceSt = stdev
                                         )- 0.5)*(max(y)-min(y)) + np.mean(y))
    plt.draw()

def open_file(event):
    global y
    global x
    global file_path
    global mean
    global stdev
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    if (file_path == ''): return
    x, y = readTimeTrace(file_path)
    dilution = int(len(y)/10000) + 1
    axTimeTrace.clear()
    axTimeTrace.plot(x[::dilution],y[::dilution])

    dert, derc = derivative(time, current)
    mean = np.mean(derc)
    stdev = np.std(derc)
    val_update(event)
   
def run_alalysis(event):
    folder = file_path[:-4]
    try:
        os.mkdir(folder)
    except:
        pass
    t1 = list()
    t2 = list()
    lasttimechange = 0
    count2levels = RTSanalysis2lvl(x, y, coefNeighbour = int(sl3AmpWin.val), 
                    coefThreshold = sl2Threshold.val,
                    coefSmooth = int(sl4Smooth.val))
    for i in range(1, len(count2levels)):
        delta = count2levels[i] - count2levels[i-1]
        if delta != 0:
            if np.sign(delta) > 0:
                t1.append(x[i] - x[lasttimechange])
            else:
                t2.append(x[i] - x[lasttimechange])
            lasttimechange = i
    
    #plot histograms quickly
    bins_number = 30
    plt.figure('Time1')
    plt.hist(t1, bins=bins_number)  # arguments are passed to np.histogram
    hist, bin_edges = np.histogram(t1, bins = bins_number)
    plt.xlim(bin_edges[0], bin_edges[-1])
    plt.yscale('log')
    plt.savefig(folder + '/time1dist.png', dpi = 300)
    plt.figure('Time2')
    plt.hist(t2, bins=bins_number)  # arguments are passed to np.histogram
    hist, bin_edges = np.histogram(t2, bins = bins_number)
    plt.xlim(bin_edges[0], bin_edges[-1])
    plt.yscale('log')
    plt.savefig(folder + '/time2dist.png', dpi = 300)
    
    with open(folder + '/Time1.dat', 'w') as f:
        f.write("Averaged = %s\n" % np.mean(t1))
        for item in t1:
            f.write("%s\n" % item)
    with open(folder + '/Time2.dat', 'w') as f:
        f.write("Averaged = %s\n" % np.mean(t2))
        for item in t2:
            f.write("%s\n" % item)  
    with open(folder + '/Binary.dat', 'w') as f:
        for item in count2levels:
            f.write("%s\n" % item) 
    print('ready')

y = []
x = []
mainFig = plt.figure()
mainFig.suptitle('Click the position on timetrace for zooming in')
axTimeTrace = mainFig.add_subplot(211)
axZoom = mainFig.add_subplot(212)
plt.subplots_adjust(left=0.1, bottom=0.4, right = 0.9, top = 0.9)

axSliderZoomSize = plt.axes([0.2, 0.25, .7, 0.03])
sl1ZoomSize = Slider(ax = axSliderZoomSize,
             label = 'Zoom Window Size',
             valmin = 0.001,
             valmax = 3,
             valinit=2.5,
             valfmt='%1.2f',
             closedmax=True)

axSliderThreshold = plt.axes([0.2, 0.21, 0.7, 0.03])
sl2Threshold = Slider(ax = axSliderThreshold, 
                      label = 'Threshold', 
                      valmin = 0.5, 
                      valmax = 5,
                      valinit = 2,
                      valfmt = '%1.2f',
                      closedmax = True)

axSliderAmpWin = plt.axes([0.2, 0.17, 0.7, 0.03])
sl3AmpWin = Slider(ax = axSliderAmpWin, 
                      label = 'Amplitude neighbour', 
                      valmin = 1, 
                      valmax = 10,
                      valinit = 1,
                      valfmt = '%i',
                      valstep = 1,
                      closedmax = True)

axSliderSmooth = plt.axes([0.2, 0.13, 0.7, 0.03])
sl4Smooth = Slider(ax = axSliderSmooth, 
                      label = 'Presmoothing', 
                      valmin = 1, 
                      valmax = 10,
                      valinit = 1,
                      valfmt = '%i',
                      valstep = 1,
                      closedmax = True)

openax = plt.axes([0.1, 0.9, 0.1, 0.04])
buttonOpen = Button(openax, 'Open', color='green', hovercolor='0.975')

runax = plt.axes([0.8, 0.9, 0.1, 0.04])
buttonRun = Button(runax, 'Run', color='yellow', hovercolor='0.5')

mainFig.canvas.mpl_connect('button_press_event', on_press)
sl1ZoomSize.on_changed(zoom_update)
sl2Threshold.on_changed(val_update)
sl3AmpWin.on_changed(val_update)
sl4Smooth.on_changed(val_update)
buttonOpen.on_clicked(open_file)
buttonRun.on_clicked(run_alalysis)

plt.show()