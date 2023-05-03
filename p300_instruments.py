import os

import pandas as pd
import pyxdf
import struct
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from copy import deepcopy
import json
from mi_sanity import gen_erds_plots
import pickle
import h5py
from glob import glob
from matplotlib.colors import TwoSlopeNorm
from matplotlib.pyplot import cm
from datetime import datetime

import matplotlib
matplotlib.use('Qt5Agg')


csvs = ['../p300/eeg_data20230503072054.csv', '../p300/eeg_data20230503072725.csv']

csv_path = csvs[0]

# df = pd.read_csv(csv_path)

with open(csv_path, 'rt') as f:
    lines = list(f)

timestamps = lines[-3].split(';')[:-1]
flashes = lines[-2].split(';')[:-1]
btn_presses = lines[-1].split(';')[:-1]

assert len(timestamps) == len(lines) - 3

flashes = [f.split(',') for f in flashes]
btn_presses = [b.split(',') for b in btn_presses]

to_timestamp = lambda t: datetime.strptime(t, '%Y-%m-%d %H:%M:%S:%f')
timestamps = np.array([to_timestamp(t).timestamp() for t in timestamps])
flashes = np.array([[to_timestamp(t).timestamp(), int(i)] for t, i in flashes])
btn_presses = [[to_timestamp(t).timestamp(), int(i)] for t, i in btn_presses]

stream_byte = [l.split(';')[:-1] for l in lines[:-3]]
stream_byte = np.array([[int(xi) for xi in x] for x in stream_byte], dtype=np.uint8)

stream = np.array([[f.view('float32') for f in np.array_split(s, 17)] for s in stream_byte])
stream = stream[..., 0].transpose()

eeg = stream[:8]


sfreq = 250  # Hz
ch_names = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']  # unicorn
verbose = False


eeg -= np.mean(eeg, axis=0)

btn_begs = btn_presses[:-1:2]
btn_ends = btn_presses[1::2]

btn_runs = [b[1] for b in btn_begs]
timestamp_runs = []
eeg_runs = []
flash_runs = []

for (btn_beg_t, btnb), (btn_end_t, btne) in zip(btn_begs, btn_ends):
    assert btnb == btne
    btn = btnb

    r = (btn_beg_t < timestamps) & (timestamps < btn_end_t)
    timestamp_runs.append(timestamps[r])
    eeg_runs.append(eeg[:, r])
    flash_runs.append(flashes[(btn_beg_t < flashes[:, 0]) & (flashes[:, 0] < btn_end_t)])


timestamp_run = timestamp_runs[0]
eeg_run = eeg_runs[0]
flash_run = flash_runs[0]
btn = btn_begs[0][1]

eeg_info = mne.create_info(ch_types='eeg', ch_names=ch_names, sfreq=sfreq)
raw = mne.io.RawArray(eeg_run, eeg_info, verbose=verbose)
std_1020 = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(std_1020)

filt_raw_eeg = raw.copy().pick_types(eeg=True).filter(1, 40, h_trans_bandwidth=5, n_jobs=4, verbose=verbose) \
    .notch_filter(50, trans_bandwidth=1, n_jobs=4, verbose=verbose)

events_run_i = np.array([np.argmin(np.abs(timestamp_run - f)) for f in flash_run[:, 0]])
events_run = np.stack([events_run_i, np.zeros_like(events_run_i), flash_run[:, 1].astype(events_run_i.dtype)], axis=1)

baseline = (-.2, 0)

event_ids = {0, 1, 2, 3}
target_event_ids = {str(btn - 1): btn - 1}
epochs_target = mne.Epochs(filt_raw_eeg, events_run, event_id=target_event_ids, baseline=None, verbose=verbose,
                           tmin=baseline[0], tmax=.5)
epochs_target.apply_baseline(baseline, verbose=verbose)

rest_event_ids = {str(e): e for e in event_ids.difference(target_event_ids.keys())}
epochs_rest = mne.Epochs(filt_raw_eeg, events_run, event_id=rest_event_ids, baseline=None, verbose=verbose,
                         tmin=baseline[0], tmax=.5)
epochs_rest.apply_baseline(baseline, verbose=verbose)

for elec in ['Fz', 'Cz', 'Pz', 'Oz']:
    epochs_target.average(picks=elec).plot(titles=f'target {elec}')
    epochs_rest.average(picks=elec).plot(titles=f'rest {elec}')
