from scipy import signal
import numpy as np
from sklearn.preprocessing import StandardScaler


def DCFilter(data):
    new_data = []
    for d in data:
        new_data.append(d - np.mean(d))
    return np.array(new_data)


def Notch(data, freq=50.0):
    res = []
    original_shape = data.shape
    if len(data.shape) == 2:
        data = data.reshape(-1, data.shape[0], data.shape[1])
    for sample in data:
        for channel in sample:
            samp_freq = 250  # Sample frequency (Hz)
            notch_freq = freq  # Frequency to be removed from signal (Hz)
            quality_factor = 30.0  # Quality factor
            b, a = signal.iirnotch(notch_freq, quality_factor, samp_freq)
            channel = signal.filtfilt(b, a, channel)
            res.append(channel)
    return np.array(res).reshape(original_shape)


def Bandpass(data, lowcut=1.0, highcut=60.0, order=4, sample_rate=250):
    res = []
    original_shape = data.shape
    if len(data.shape) == 2:
        data = data.reshape(-1, data.shape[0], data.shape[1])
    for sample in data:
        for channel in sample:
            lowcut = lowcut  # Low cut frequency (Hz)
            highcut = highcut  # High cut frequency (Hz)
            b, a = signal.butter(
                order, [lowcut / sample_rate, highcut / sample_rate], btype="bandpass"
            )
            channel = signal.filtfilt(b, a, channel)
            res.append(channel)
    return np.array(res).reshape(original_shape)


def Normalize(data):
    res = []
    original_shape = data.shape
    if len(data.shape) == 2:
        data = data.reshape(-1, data.shape[0], data.shape[1])
    for sample in data:
        scaler = StandardScaler()
        scaler.fit(sample)
        sample = scaler.transform(sample)
        res.append(sample)
    return np.array(res).reshape(original_shape)



def Resample(data, new_length=128):
    res = []
    original_shape = data.shape
    target_shape  =  []
    for d in original_shape:
        target_shape.append(d)
    target_shape[-1] = new_length
    if len(data.shape) == 2:
        data = data.reshape(-1, data.shape[0], data.shape[1])
    for sample in data:
        for channel in sample:
            res.append(signal.resample(channel, new_length))
    return np.array(res).reshape(target_shape)

