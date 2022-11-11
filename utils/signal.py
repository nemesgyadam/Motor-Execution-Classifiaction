from scipy import signal
import numpy as np

def DCFilter(data):
    new_data = []
    for d in data:
        new_data.append(d-np.mean(d))
    return np.array(new_data)

def Notch(data, freq=50.0):
    res = []
    for channel in data:
        samp_freq = 250  # Sample frequency (Hz)
        notch_freq = freq  # Frequency to be removed from signal (Hz)
        quality_factor = 30.0  # Quality factor
        b, a = signal.iirnotch(notch_freq, quality_factor, samp_freq)
        channel =  signal.filtfilt(b, a, channel)
        res.append(channel)
    return np.array(res)
    
def Bandpass(data, lowcut=1.0, highcut=60.0):
    res = []
    for channel in data:
        samp_freq = 250  # Sample frequency (Hz)
        lowcut = lowcut  # Low cut frequency (Hz)
        highcut = highcut  # High cut frequency (Hz)
        b, a = signal.butter(4, [lowcut/samp_freq, highcut/samp_freq], btype='bandpass')
        channel =  signal.filtfilt(b, a, channel)
        res.append(channel)
    return np.array(res)