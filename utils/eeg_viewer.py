# adapted from: https://github.com/labstreaminglayer/liblsl-Python/blob/master/pylsl/examples/ReceiveAndPlot.py
import os, sys
import numpy as np
import math
import pylsl
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from typing import List, Callable
from collections import deque
import matplotlib.pylab as pylab
import mne


# Basic parameters for the plotting window
plot_duration = 5  # how many seconds of data to show
update_interval = 60  # ms between screen updates
pull_interval = 500  # ms between each pull operation


class Inlet:
    """Base class to represent a plottable inlet"""
    def __init__(self, info: pylsl.StreamInfo):
        # create an inlet and connect it to the outlet we found earlier.
        # max_buflen is set so data older the plot_duration is discarded
        # automatically and we only pull data new enough to show it

        # Also, perform online clock synchronization so all streams are in the
        # same time domain as the local lsl_clock()
        # (see https://labstreaminglayer.readthedocs.io/projects/liblsl/ref/enums.html#_CPPv414proc_clocksync)
        # and dejitter timestamps
        self.inlet = pylsl.StreamInlet(info, max_buflen=plot_duration,
                                       processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter)
        # store the name and channel count
        self.name = info.name()
        self.channel_count = info.channel_count()
        self.anal_stream = None

    def pull_and_plot(self, plot_time: float, plt: pg.PlotItem):
        """Pull data from the inlet and add it to the plot.
        :param plot_time: lowest timestamp that's still visible in the plot
        :param plt: the plot the data should be shown on
        """
        # We don't know what to do with a generic inlet, so we skip it.
        pass


class DataInlet(Inlet):
    """A DataInlet represents an inlet with continuous, multi-channel data that
    should be plotted as multiple lines."""
    dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]

    def __init__(self, info: pylsl.StreamInfo, plt: pg.PlotItem, static_offset: int = 0,
                 anal_stream: Callable = None, cmap_name='GnBu'):
        super().__init__(info)
        self.static_offset = static_offset
        self.anal_stream = anal_stream

        # calculate the size for our buffer, i.e. two times the displayed data
        bufsize = (2 * math.ceil(info.nominal_srate() * plot_duration), info.channel_count())
        self.buffer = np.empty(bufsize, dtype=self.dtypes[info.channel_format()])
        empty = np.array([])

        # create one curve object for each channel/line that will handle displaying the data
        cm = pylab.get_cmap(cmap_name)
        colors = [(np.array(cm(i / self.channel_count)[:3]) * 255).astype(int)
                  for i in range(self.channel_count)]  # courtesy of retarted qt plotting 
        self.curves = [pg.PlotCurveItem(x=empty, y=empty, autoDownsample=True, pen=pg.mkPen(colors[i]))
                       for i in range(self.channel_count)]
        for curve in self.curves:
            plt.addItem(curve)

    def pull_and_plot(self, plot_time, plt):
        # pull the data
        _, ts = self.inlet.pull_chunk(timeout=0.0,
                                      max_samples=self.buffer.shape[0],
                                      dest_obj=self.buffer)
        # ts will be empty if no samples were pulled, a list of timestamps otherwise
        if ts:
            ts = np.asarray(ts)
            y = self.buffer[0:ts.size, :]
            this_x = None
            old_offset = 0
            new_offset = 0
            for ch_ix in range(self.channel_count):
                # we don't pull an entire screen's worth of data, so we have to
                # trim the old data and append the new data to it
                old_x, old_y = self.curves[ch_ix].getData()
                # the timestamps are identical for all channels, so we need to do
                # this calculation only once
                if ch_ix == 0:
                    # find the index of the first sample that's still visible,
                    # i.e. newer than the left border of the plot
                    old_offset = old_x.searchsorted(plot_time)
                    # same for the new data, in case we pulled more data than
                    # can be shown at once
                    new_offset = ts.searchsorted(plot_time)
                    # append new timestamps to the trimmed old timestamps
                    this_x = np.hstack((old_x[old_offset:], ts[new_offset:]))
                
                # append new data to the trimmed old data
                this_y = np.hstack((old_y[old_offset:], y[new_offset:, ch_ix])) #- ch_ix - self.static_offset))
                # replace the old data
                self.curves[ch_ix].setData(this_x, this_y)


class MarkerInlet(Inlet):
    """A MarkerInlet shows events that happen sporadically as vertical lines"""
    def __init__(self, info: pylsl.StreamInfo, formatter: Callable = None):
        super().__init__(info)
        self.formatter = formatter if formatter else lambda m: (str(m), m)
        self.color_pens = [pg.mkPen((np.array(pylab.cm.tab20(i)[:3]) * 255).astype(int)) for i in range(20)]

    def pull_and_plot(self, plot_time, plt):
        strings, timestamps = self.inlet.pull_chunk(0)
        if timestamps:
            for string, ts in zip(strings, timestamps):
                label, marker_id = self.formatter(string)
                marker_id = marker_id[0] if isinstance(marker_id, list) else marker_id
                plt.addItem(pg.InfiniteLine(ts, angle=90, movable=False, label=label, pen=self.color_pens[marker_id % 20]))


class UnicornStreamAnal:

    class cmdcol:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    GOOD_CHAR = '#'
    BAD_CHAR = 'X'

    def __init__(self) -> None:
        self.call_i = 0
        self.update_freq = 1
        self.neeg_chans = 8
        self.sampling_freq = 250
        self.ndots_hl = 7

        ascii_path = 'resources/unicorn_ascii.txt'
        with open(ascii_path, 'rt') as f:
            self.ascii_art = f.read()
        self.ascii_art = np.array(list(self.ascii_art))

        self.chans = range(1, self.neeg_chans + 1)
        self.clean_rng = (-100, 100)  # (-100, 100)
        self.chan_poz = [np.sort(np.where(self.ascii_art == str(ch))[0]) for ch in self.chans]
        self.chan_num_poz = [cp[2] for cp in self.chan_poz]  # assumed that always the 3rd chan character pos is the middle one
        self.ndots = (self.ascii_art == '.').sum()

        self.clean_filter = mne.filter.create_filter(None, self.sampling_freq, l_freq=1, h_freq=30, verbose=False)

    def _is_clean(self, x, y):  # per channel
        filt_y = np.convolve(y, self.clean_filter, 'same')
        filt_y = mne.filter.notch_filter(filt_y, self.sampling_freq, [50, 60])
        last_1_sec = filt_y[-self.sampling_freq:]
        return np.all((self.clean_rng[0] < last_1_sec) & (last_1_sec < self.clean_rng[1]))

    def __call__(self, xy):
        self.call_i += 1
        xy = xy[:len(self.chans)]  # first 8 channels only

        if self.call_i % self.update_freq == 0:
            are_clean = [self._is_clean(x, y) for x, y in xy]
            
            for is_clean, chan_pos in zip(are_clean, self.chan_poz):
                self.ascii_art[chan_pos] = UnicornStreamAnal.GOOD_CHAR if is_clean else UnicornStreamAnal.BAD_CHAR
            for cp, chan_i in zip(self.chan_num_poz, self.chans):
                self.ascii_art[cp] = str(chan_i)
            
            dot_counter = 0
            ascii_print = []
            for c in self.ascii_art:

                active_dot = False
                if c == UnicornStreamAnal.GOOD_CHAR:
                    ascii_print.append(UnicornStreamAnal.cmdcol.OKGREEN)
                elif c == UnicornStreamAnal.BAD_CHAR:
                    ascii_print.append(UnicornStreamAnal.cmdcol.FAIL)
                elif c.isdigit():
                    ascii_print.append(UnicornStreamAnal.cmdcol.BOLD)
                elif c == '.':
                    dot_counter += 1
                    which_dot = self.call_i % self.ndots
                    if which_dot <= dot_counter < which_dot + self.ndots_hl:
                        active_dot = True
                        ascii_print.append(UnicornStreamAnal.cmdcol.WARNING)
                ascii_print.append(c)

                if c in (UnicornStreamAnal.GOOD_CHAR, UnicornStreamAnal.BAD_CHAR) or c.isdigit() or active_dot:
                    ascii_print.append(UnicornStreamAnal.cmdcol.ENDC)
            
            ascii_print = ''.join(ascii_print)
            
            nsamples = len(xy[0][0])
            mean_min_y = np.mean([y.min() for x, y in xy])
            mean_max_y = np.mean([y.max() for x, y in xy])

            min_maxes = {xyi + 1: (round(y.min(), 2), round(y.max(), 2)) for xyi, (x, y) in enumerate(xy)}

            os.system('cls')
            print(ascii_print)
            print(f'Samples: {nsamples}, {nsamples / self.sampling_freq:.2f} '
                  f'sec | mean v: ({mean_min_y:.2f}, {mean_max_y:.2f}) |'
                  f'\nv: {dict(list(min_maxes.items())[:len(self.chans) // 2])}'
                  f'\n   {dict(list(min_maxes.items())[len(self.chans) // 2:])}', flush=True)


def main():
    # firstly resolve all streams that could be shown
    inlets: List[Inlet] = []
    print("looking for streams")
    streams = pylsl.resolve_streams()

    # Create the pyqtgraph window
    pw = pg.plot(title='LSL Plot')
    plt = pw.getPlotItem()
    plt.enableAutoRange(x=False, y=True)

    # stream anal
    anal_freq = 50
    unicorn_stream_anal = UnicornStreamAnal()
    step = 0

    # iterate over found streams, creating specialized inlet objects that will
    # handle plotting the data
    accum_static_offset = 0
    for info in streams:
        
        if info.type() == 'Markers':
            if info.nominal_srate() != pylsl.IRREGULAR_RATE \
                    or info.channel_format() != pylsl.cf_string:
                print('Invalid marker stream ' + info.name())
            print('Adding marker inlet: ' + info.name())

            formatter = None
            if info.name() == 'Gamepad Events':
                btn_map = {0: 'A', 1: 'B', 2: 'X', 3: 'Y', 4: 'Down', 5: 'Left', 6: 'Right', 7: 'Up',
                           8: 'L1', 9: 'R1', 10: 'Start', 11: 'L3', 12: 'R3', 13: 'Select', 14: 'Guide'}
                formatter = lambda m: (f'G-{btn_map[m[0]]}-{("OFF", "ON")[m[1]]}', m[0])

            inlets.append(MarkerInlet(info, formatter))

        elif info.nominal_srate() != pylsl.IRREGULAR_RATE \
                and info.channel_format() != pylsl.cf_string:
            
            stream_anal = None
            if info.name() == 'Unicorn':
                stream_anal = unicorn_stream_anal

            inlets.append(DataInlet(info, plt, accum_static_offset, stream_anal))
            accum_static_offset += inlets[-1].channel_count
        
        else:
            print('Don\'t know what to do with stream ' + info.name())
    
    print('Inlets:', [inl.name for inl in inlets])

    def scroll():
        """Move the view so the data appears to scroll"""
        # We show data only up to a timepoint shortly before the current time
        # so new data doesn't suddenly appear in the middle of the plot
        fudge_factor = pull_interval * .002
        plot_time = pylsl.local_clock()
        pw.setXRange(plot_time - plot_duration + fudge_factor, plot_time - fudge_factor)

    def update():
        # Read data from the inlet. Use a timeout of 0.0 so we don't block GUI interaction.
        mintime = pylsl.local_clock() - plot_duration
        # call pull_and_plot for each inlet.
        # Special handling of inlet types (markers, continuous data) is done in
        # the different inlet classes.
        for inlet in inlets:
            inlet.pull_and_plot(mintime, plt)
        
        # stream analysis
        # os.system('cls')
        for inlet in inlets:
            if inlet.anal_stream:
                xy = [inlet.curves[ch].getData() for ch in range(inlet.channel_count)]
                if len(xy) > 0 and len(xy[0]) > 0 and len(xy[0][0]) > 0:
                    inlet.anal_stream(xy)

    # create a timer that will move the view every update_interval ms
    update_timer = QtCore.QTimer()
    update_timer.timeout.connect(scroll)
    update_timer.start(update_interval)

    # create a timer that will pull and add new data occasionally
    pull_timer = QtCore.QTimer()
    pull_timer.timeout.connect(update)
    pull_timer.start(pull_interval)

    # Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        # QtGui.QApplication.instance().exec_()
        QtWidgets.QApplication.instance().exec_()


if __name__ == '__main__':
    main()
