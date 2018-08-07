#!/usr/bin/python
#
# Copyright 2013 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner, June 2013
#
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ssp

# I can't remember why this was necessary, but it fails on my Mac
# from mpl_toolkits.axes_grid1 import make_axes_locatable

class Figure:
    """
    Essentially a matplotlib figure, but with added value.  In
    particular it always has sub-plots.
    """
    def __init__(self, rows=1, cols=1):
        self.rows = rows
        self.cols = cols
        self.next = 1
        self.fig = plt.figure()
        #self.fig.set_tight_layout(True)

    # Plot order: blue, green, red, ...
    def subplot(self):
        if self.next > self.rows * self.cols:
            raise OverflowError('Out of plots')
        axesSubplot = self.fig.add_subplot(self.rows, self.cols, self.next)
        self.next += 1
        return axesSubplot


    def specplot(self, ax, a, pcm):
        ax.imshow(np.transpose(10*np.log10(a)),
                  origin='lower', cmap='bone', aspect='auto')
        ax.set_yticks((0,a.shape[-1]-1))
        ax.set_yticklabels(('0', pcm.rate/2))

    def show(self):
        plt.show()

    def SpectrumPlot(self, data, pcm):
        return ssp.SpectrumPlot(self, data, pcm)

    def EnergyPlot(self, data, pcm):
        return ssp.EnergyPlot(self, data, pcm)

def zplot(fig, a):
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    arg = np.angle(a)
    mag = np.abs(a)
    ax.plot(arg, mag, 'r+')
    ax.set_rmax(1.0)


class FramePlot:
    """
    Base class for frame-based plot pairs.
    axT is time axis
    axF id frame axis
    """
    def __init__(self, fig, data):
        self.data = data
        self.axT = fig.subplot()
        divider = make_axes_locatable(self.axT)
        self.axF = divider.append_axes("right", size="25%", pad=0.1)
        self.axF.yaxis.set_ticks_position('right')
        self.axF.yaxis.set_label_position('right')
        self.axT.figure.canvas.mpl_connect('button_press_event', self)
        self.axT.figure.canvas.mpl_connect('button_release_event', self)
        self.axT.figure.canvas.mpl_connect('motion_notify_event', self)
        self.press = False

    def __call__(self, event):
        if event.inaxes != self.axT:
            return
        elif event.name == 'button_release_event':
            self.press = False
            return
        elif event.name == 'button_press_event':
            self.press = True
        if self.press:
            self._plotF(event.xdata)
            self.axF.figure.canvas.draw()


class SpectrumPlot(FramePlot):
    """
    Figure with spectrum.
    """
    def __init__(self, fig, data, pcm):
        FramePlot.__init__(self, fig, data)
        self.max = 10*np.log10(np.max(data))
        self.min = 10*np.log10(np.min(data))
        fig.specplot(self.axT, self.data, pcm)
        self.axT.set_ylabel('Frequency (Hz)')
        self._plotF(0)

    def _plotF(self, frame):
        self.axF.clear()
        self.axF.plot(10*np.log10(self.data[frame])-self.max)
        self.axF.set_xlim(0, self.data.shape[-1]-1)
        self.axF.set_ylim(np.max([self.min-self.max, -90]), 0)
        self.axF.set_ylabel('Level (dB)')
        self.axF.grid(True)

class EnergyPlot(FramePlot):
    """
    Figure with energy & time frames.
    """
    def __init__(self, fig, data, pcm):
        FramePlot.__init__(self, fig, data)
        e = ssp.Energy(data)
        self.max = 10*np.log10(np.amax(e))
        self.min = 10*np.log10(np.amin(e))
        self.axT.plot(10*np.log10(e)-self.max)
        self.axT.set_xlim(0, self.data.shape[0]-1)
        self.axT.set_ylim(np.max([self.min-self.max, -90]), 0)
        self.axT.set_ylabel('Level (dB)')
        self.axT.grid(True)
        self._plotF(0)

    def _plotF(self, frame):
        self.axF.clear()
        self.axF.plot(self.data[frame])
        self.axF.set_xlim(0, self.data.shape[-1]-1)
        self.axF.set_ylabel('Amplitude')
        self.axF.grid(True)
