import matplotlib.pyplot as plt
import matplotlib.backend_bases
from matplotlib.widgets import Slider, Button
import math
#https://www.youtube.com/watch?v=pTrfnHleY0U

def on_press(event):
    print('you pressed', event.button, event.xdata, event.ydata)
    if (event.inaxes == ax1):
        ax2.clear()
        ax2.set_xlim(event.xdata,event.xdata+10)
        ax2.plot(x,y)
        fig1.canvas.flush_events()
        event.canvas.draw()


fig1 = plt.figure()
fig1.suptitle('mouse hover over figure or axes to trigger events')
ax1 = fig1.add_subplot(211)
x = list(range(0,100))
y = [math.sin(x[i]) for i in range (0,100)]
ax1.plot(x,y)
ax2 = fig1.add_subplot(212)
plt.subplots_adjust(left=0.1, bottom=0.35)

fig1.canvas.mpl_connect('button_press_event', on_press)


axSlider1 = plt.axes([0.1, 0.2, 0.8, 0.05])
sl1 = Slider(axSlider1, 'Slider 1', valmin=0, valmax = 100)

axSlider2 = plt.axes([0.1, 0.1, .8, 0.05])
sl2 = Slider(ax = axSlider2,
             label = 'Slider 2',
             valmin = 0,
             valmax = 100,
             valinit=20,
             valfmt='%1.2f',
             slidermin=sl1,
             closedmax=True)
             
def val_update(val):
    size = sl2.val
    x1, x2 = ax2.get_xlim()
    xav = x1/2+x2/2
    ax2.set_xlim(xav-size, xav + size)
    plt.draw()

sl2.on_changed(val_update)
plt.show()
