import matplotlib.pyplot as plt
import matplotlib.backend_bases
from matplotlib.widgets import Slider, Button, CheckButtons
import numpy as np
import os
#https://www.youtube.com/watch?v=pTrfnHleY0U
from noiseanalysis import RTSanalysis2lvl, readTimeTrace, derivative, smoothing
from datetime import datetime
import csv


#for storing all data imported from file
y = []
x = []
#standard windows size for zooming region
win_size = 0.05
#if file was not yet imported, file_path is empty
file_path = ''
#to increase the speed of counting, mean and stdev will store the info about derivative
mean = 0
stdev = 0
#is responsible for showing what region is zoomed in
selection_area = None
#positions of cutting if needed
cutX1 = 0
cutX2 = 0
#if something was cut from the x and y, it will show that we need to create a new file
createNew = False

def on_press(event):
    global win_size
    global selection_area
    global cutX1, cutX2
    print('You pressed', event.button, event.xdata, event.ydata)

    #cutting has higher priority to zooming and selecting
    if (cutRegime.get_status()[0] == True and (event.inaxes == axTimeTrace or event.inaxes == axZoom)):
        print(event.xdata, 'is chose as one edge for cutting')
        cutX1 = event.xdata
        return

    #if it is not cutting
    if (event.inaxes == axTimeTrace):
        axZoom.set_xlim(event.xdata - win_size/2,event.xdata + win_size/2)
        mainFig.canvas.flush_events()
        event.canvas.draw()
        
        #selection_area
        try:
            selection_area.remove()
        except:
            print('Zoomed in was not shown before')
        selection_area = axTimeTrace.axvspan(event.xdata - win_size/2, event.xdata + win_size/2, color='red', alpha=0.5)
        print('New selection is shown')
        val_update(event)

    if (event.inaxes == axZoom):
        axZoom.set_xlim(event.xdata - win_size/2,event.xdata + win_size/2)
        mainFig.canvas.flush_events()
        event.canvas.draw()

        #selection_area
        try:
            selection_area.remove()
        except:
            print('Zoomed in was not shown before')
        selection_area = axTimeTrace.axvspan(event.xdata - win_size/2, event.xdata + win_size/2, color='red', alpha=0.5)
        print('New selection is shown')
    

def on_release(event):
    global cutX1, cutX2
    global x, y 
    global createNew
    print('You released', event.button, event.xdata, event.ydata)
    #if we were in cutting Regime
    if (cutRegime.get_status()[0] == True and (event.inaxes == axTimeTrace or event.inaxes == axZoom)):
        print(event.xdata, 'is selected as another edge for cutting')
        cutX2 = event.xdata
        posToCutLeft = 0
        posToCutRight = 0
        #if it was just a sudden click, coordinates are the same - do nothing
        if (cutX1 == cutX2):
            return
        else:
            #sort the cutX1 is on the left and cutX2 is on the right
            if (cutX1 > cutX2):
                temp = cutX2
                cutX2 = cutX1
                cutX1 = temp
            print('Remove the date from [',cutX1,cutX2,'] region')
            #set the cutX in range of real x values
            if cutX2 > x[-1]: cutX2 = x[-1]
            if cutX1 < x[0]: cutX1 = x[0]
            posToCutLeft = 0
            for i in range(0, len(x)):
                if x[i] >= cutX1:
                    posToCutLeft = i
                    break
            posToCutRight = len(x) - 1
            for i in range(len(x) - 1, -1, -1):
                if x[i] <= cutX2:
                    posToCutRight = i
                    break
            print('Region to be removed [',x[posToCutLeft],x[posToCutRight],']')
            
            createNew = True
            x = np.concatenate((x[:posToCutLeft],x[posToCutRight+1:len(x)]))
            y = np.concatenate((y[:posToCutLeft],y[posToCutRight+1:len(y)]))
            print('Data was modified, createNewFile if [Run]',createNew)

            dilution = int(len(y)/10000) + 1
            axTimeTrace.clear()
            axTimeTrace.plot(x[::dilution],y[::dilution])
            val_update(event)

        return

def zoom_update(val):
    global win_size
    global x
    global y
    global selection_area
    win_size = 10 ** sl1ZoomSize.val / 200
    x1, x2 = axZoom.get_xlim()
    xav = x1/2+x2/2
    x1 = xav - win_size/2
    x2 = xav + win_size/2
    axZoom.set_xlim(x1, x2)

    #selection_area
    try:
        selection_area.remove()
    except:
        print('Zoomed in was not shown before')
    selection_area = axTimeTrace.axvspan(x1, x2, color='red', alpha=0.5)
    print('New selection is shown')
             
def val_update(val):
    global win_size
    global x
    global y
    win_size = 10 ** sl1ZoomSize.val / 200
    x1, x2 = axZoom.get_xlim()
    xav = x1/2 + x2/2
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
        #temp x and y are used to have a quick upgrade for zoom region
        #x_temp = x
        #y_temp = y
        axZoom.plot(x_temp, smoothing(y_temp, int(sl4Smooth.val)) )
        #shiftCoeft is used as adding or not part
        if (checkShift.get_status()[0] == True):
            shiftCoef = 1
        else:
            shiftCoef =-1
        if (len(x_temp) > 1):
            axZoom.plot(x_temp,
                        (RTSanalysis2lvl(x_temp,
                                         y_temp,
                                         coefNeighbour = int(sl3AmpWin.val), 
                                         coefThreshold = sl2Threshold.val,
                                         coefSmooth = int(sl4Smooth.val),
                                         forceM = mean,
                                         forceSt = stdev
                                         ) + shiftCoef * 0.5)*(max(y)-min(y)) + np.mean(y))
        print('New fitting was applied')
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
    if (file_path == ''): 
        print('No file was selected')
        return
    print(file_path, 'is selected')
    mainFig.suptitle(file_path)
    x, y = readTimeTrace(file_path)
    createNew = False
    dilution = int(len(y)/10000) + 1
    axTimeTrace.clear()
    axTimeTrace.plot(x[::dilution],y[::dilution])

    dert, derc = derivative(x, y)
    mean = np.mean(derc)
    stdev = np.std(derc)
    val_update(event)
   
def run_alalysis(event):
    global file_path
    global createNew
    filep = file_path
    if (createNew):
        now = datetime.now()
        dt_string = now.strftime("%m%d%H%M%S")
        filep = file_path[:-4] + dt_string + '.dat'
        print('New file', filep, 'was created')
        with open(filep, 'w') as datfile:
            writer = csv.writer(datfile, delimiter='\t')
            writer.writerows(zip(y, x))
    folder = filep[:-4]
    try:
        os.mkdir(folder)
    except:
        pass
    t1 = list()
    t2 = list()
    lasttimechange = 0
    count2levels = RTSanalysis2lvl(x, y, coefNeighbour = int(sl3AmpWin.val), 
                    coefThreshold = sl2Threshold.val,
                    coefSmooth = int(sl4Smooth.val),
                    forceM = mean,
                    forceSt = stdev)

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
    plt.clf()
    plt.figure('Time2')
    plt.hist(t2, bins=bins_number)  # arguments are passed to np.histogram
    hist, bin_edges = np.histogram(t2, bins = bins_number)
    plt.xlim(bin_edges[0], bin_edges[-1])
    plt.yscale('log')
    plt.savefig(folder + '/time2dist.png', dpi = 300)
    plt.clf()
    with open(folder + '/Time1.dat', 'w') as f:
        f.write("Event = %s Averaged = %s Std = %s Sterror = %s\n" % (len(t1), np.mean(t1), np.std(t1), np.std(t1)/np.sqrt(len(t1))))
        for item in t1:
            f.write("%s\n" % item)
    with open(folder + '/Time2.dat', 'w') as f:
        f.write("Event = %s Averaged = %s Std = %s Sterror = %s\n" % (len(t2), np.mean(t2), np.std(t2), np.std(t1)/np.sqrt(len(t1))))
        for item in t2:
            f.write("%s\n" % item)  
    with open(folder + '/Binary.dat', 'w') as f:
        i = 0
        for item in count2levels:
            f.write("%s\t%s\n" % (item, x[i]))
            i += 1
    print('Data was stored to', folder)



mainFig = plt.figure()
axTimeTrace = mainFig.add_subplot(211)
axZoom = mainFig.add_subplot(212)
plt.subplots_adjust(left=0.1, bottom=0.4, right = 0.9, top = 0.9)

axSliderZoomSize = plt.axes([0.2, 0.25, .7, 0.03])
sl1ZoomSize = Slider(ax = axSliderZoomSize,
             label = 'Zoom Window Size',
             valmin = 0.001,
             valmax = 3,
             valinit=2.5,
             valfmt='%1.3f',
             closedmax=True)

axSliderThreshold = plt.axes([0.2, 0.21, 0.7, 0.03])
sl2Threshold = Slider(ax = axSliderThreshold, 
                      label = 'Threshold', 
                      valmin = 0.25, 
                      valmax = 7,
                      valinit = 2,
                      valfmt = '%1.3f',
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
                      valmax = 31,
                      valinit = 1,
                      valfmt = '%i',
                      valstep = 1,
                      closedmax = True)

openax = plt.axes([0.1, 0.9, 0.1, 0.04])
buttonOpen = Button(openax, 'Open', color='green', hovercolor='0.975')

runax = plt.axes([0.8, 0.9, 0.1, 0.04])
buttonRun = Button(runax, 'Run', color='yellow', hovercolor='0.5')

axCheckShift = plt.axes([0.9, 0.4, 0.08, 0.20])
checkShift = CheckButtons(axCheckShift, ['Shift'], [True])

axCut = plt.axes([0.9, 0.7, 0.08, 0.20])
cutRegime = CheckButtons(axCut, ['Cut'], [False])

mainFig.canvas.mpl_connect('button_press_event', on_press)
mainFig.canvas.mpl_connect('button_release_event', on_release)
sl1ZoomSize.on_changed(zoom_update)
sl2Threshold.on_changed(val_update)
sl3AmpWin.on_changed(val_update)
sl4Smooth.on_changed(val_update)
buttonOpen.on_clicked(open_file)
buttonRun.on_clicked(run_alalysis)
checkShift.on_clicked(val_update)

plt.show()