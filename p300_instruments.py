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
from copy import deepcopy
import pickle

import matplotlib
matplotlib.use('Qt5Agg')


def remove_artifacts(raw, perc=90):
    # Extract the EEG data and channel names from the Raw object
    data, times = raw[:, :]
    data[np.abs(data) > np.percentile(np.abs(data), perc, axis=-1)[:, None]] = 0.
    clean_raw = mne.io.RawArray(data, raw.info)
    return clean_raw


def filter_bad_epochs(epochs: mne.Epochs, percent_to_keep=90, copy=False, verbose=False) -> mne.Epochs:
    """Drops those where the peak-to-peak values are greater than 'percent_to_keep' percent of all the epochs"""
    epoch_data = epochs.get_data()
    peak_to_peak = np.max(np.max(epoch_data, axis=-1) - np.min(epoch_data, axis=-1), axis=1)
    limit_value = np.percentile(peak_to_peak, percent_to_keep)
    # bad_epochs = epochs[peak_to_peak > limit_value]
    # bad_epochs.plot(scalings=50, block=True)
    epochs = epochs.copy() if copy else epochs
    epochs.drop_bad(reject=dict(eeg=int(limit_value)), verbose=verbose)
    # epochs.drop(peak_to_peak > limit_value)
    return epochs


# TODO TRY CUTTING FREQ AT 20HZ


csvs = ['../p300/eeg_data20230503072054.csv', '../p300/eeg_data20230503072725.csv',
        '../p300/eeg_data20230515085424.csv', '../p300/eeg_data20230522074211.csv']

csv_i = 2
csv_path = csvs[csv_i]

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
cut_freq = (1, 20)
baseline = (-.2, -.01)
filter_percentile = 90
ch_names = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']  # unicorn
verbose = False
p300_chans = ['Fz', 'Cz', 'Pz', 'Oz', 'PO7', 'PO8']

# TODO
# eeg -= np.mean(eeg, axis=0)

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

times = None
all_epochs_target, all_epochs_rest = [], []
training_data = dict()

for run_i in range(len(eeg_runs)):  # TODO !!!!! range(len(eeg_runs)):   range(3), range(1, 5)
    timestamp_run = timestamp_runs[run_i]
    eeg_run = eeg_runs[run_i]
    flash_run = flash_runs[run_i]
    btn = btn_begs[run_i][1]

    eeg_info = mne.create_info(ch_types='eeg', ch_names=ch_names, sfreq=sfreq)
    raw = mne.io.RawArray(eeg_run, eeg_info, verbose=verbose)
    std_1020 = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(std_1020)

    filt_raw_eeg = raw.copy().pick_types(eeg=True).filter(*cut_freq, h_trans_bandwidth=5, n_jobs=4, verbose=verbose) \
        .notch_filter(50, trans_bandwidth=1, n_jobs=4, verbose=verbose)
    # filt_raw_eeg = remove_artifacts(filt_raw_eeg, filter_percentile)

    events_run_i = np.array([np.argmin(np.abs(timestamp_run - f)) for f in flash_run[:, 0]])
    events_run = np.stack([events_run_i, np.zeros_like(events_run_i), flash_run[:, 1].astype(events_run_i.dtype)], axis=1)

    event_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8}
    target_event_ids = {str(btn - 1): btn - 1}
    epochs_target = mne.Epochs(filt_raw_eeg, events_run, event_id=target_event_ids, baseline=None, verbose=verbose,
                               tmin=baseline[0], tmax=.5)
    epochs_target.apply_baseline(baseline, verbose=verbose)
    epochs_target = filter_bad_epochs(epochs_target, filter_percentile, copy=False, verbose=verbose)

    rest_event_ids = {str(e): e for e in event_ids.difference(target_event_ids.keys())}
    epochs_rest = mne.Epochs(filt_raw_eeg, events_run, event_id=rest_event_ids, baseline=None, verbose=verbose,
                             tmin=baseline[0], tmax=.5)
    epochs_rest.apply_baseline(baseline, verbose=verbose)
    epochs_rest = filter_bad_epochs(epochs_rest, filter_percentile, copy=False, verbose=verbose)

    ylim = (min(epochs_rest.get_data().mean((0, 1)).min(), epochs_target.get_data().mean((0, 1)).min()) * 1e6 * 2,
            max(epochs_rest.get_data().mean((0, 1)).max(), epochs_target.get_data().mean((0, 1)).max()) * 1e6 * 2)
    fig, axes = plt.subplots(len(p300_chans) + 1, 2)
    fig.suptitle(f'btn: {btn}')

    # epochs_target.plot(butterfly=True, scalings=80, n_epochs=3, picks=elecs)

    for ei, elec in enumerate(p300_chans):
        epochs_target.average(picks=elec).plot(titles=f'target {elec}', show=False, axes=axes[ei, 0], hline=[0])
        epochs_rest.average(picks=elec).plot(titles=f'rest {elec}', show=False, axes=axes[ei, 1], hline=[0])
        # for avg in epochs_rest.average(picks=elec, by_event_type=True):
        #     avg.plot(titles=f'rest {elec}', show=False, axes=axes[ei, 1])
        # axes[ei, 0].set_ylim(ylim)
        # axes[ei, 1].set_ylim(ylim)

    epochs_target.average(picks=p300_chans).plot(titles=f'target', show=False, axes=axes[-1, 0], hline=[0])
    epochs_rest.average(picks=p300_chans).plot(titles=f'rest', show=False, axes=axes[-1, 1], hline=[0])

    all_epochs_target.append(deepcopy(epochs_target.average(picks=p300_chans).data))
    all_epochs_rest.append(deepcopy(epochs_rest.average(picks=p300_chans).data))

    times = epochs_target.times
    training_data[btn] = dict(target=epochs_target.get_data(picks=p300_chans),
                              rest=epochs_rest.get_data(picks=p300_chans), times=times)

    _, ax = plt.subplots()
    target = epochs_target.get_data(picks=p300_chans).mean((0, 1))
    rest = epochs_rest.get_data(picks=p300_chans).mean((0, 1))
    times = epochs_target.times
    ax.plot(epochs_target.times, target, label='target combined', color='red')
    ax.plot(epochs_target.times, rest, label='rest combined', color='black')
    ax.set_title(f'btn: {btn}')
    xlim = ax.get_xlim()
    ax.hlines([0], *xlim, colors='gray')
    ax.set_xlim(xlim)
    ax.legend()

    # for avg in epochs_rest.average(picks=elecs, by_event_type=True):
    #     avg.plot(titles=f'rest', show=False, axes=axes[-1, 1])
    # axes[-1, 0].set_ylim(ylim)
    # axes[-1, 1].set_ylim(ylim)
    plt.tight_layout()

    plt.show(block=True)

with open(f'tmp/p300_data_{csv_i}_all.pkl', 'wb') as f:
    pickle.dump(training_data, f)

final_target = np.asarray(all_epochs_target).mean(axis=(0, 1))
final_rest = np.asarray(all_epochs_rest).mean(axis=(0, 1))

_, ax = plt.subplots()
target = final_target
rest = final_rest
ax.plot(times, target, label='target combined', color='red')
ax.plot(times, rest, label='rest combined', color='black')
ax.set_title(f'COMBINED')
xlim = ax.get_xlim()
ax.hlines([0], *xlim, colors='gray')
ax.set_xlim(xlim)
ax.legend()

plt.show(block=True)

