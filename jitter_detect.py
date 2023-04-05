import os
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

eeg_sfreq = 250
gamepad_sfreq = 125
verbose = False
# recording path
rec_path = '../recordings/sub-jitter1/ses-S001/eeg/sub-jitter1_ses-S001_task-me-l-r-lr_run-001_eeg.xdf'
exp_cfg_path = './config/lr_finger/exp_me_l_r_lr_stim-w-dots.json'

# experiment config
with open(exp_cfg_path, 'rt') as f:
    exp_cfg = json.load(f)

# channel info
shared_sfreq = max(eeg_sfreq, gamepad_sfreq)
eeg_ch_names = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']  # unicorn
gamepad_ch_names = ['LeftX', 'LeftY', 'RightX', 'RightY', 'L2', 'R2']

# load streams from xdf
streams, header = pyxdf.load_xdf(rec_path)
modalities = {s['info']['name'][0]: s for s in streams}

eeg = modalities['Unicorn']['time_series'].T[:8, :]
eeg_t = modalities['Unicorn']['time_stamps']

events = modalities['exp-marker']['time_series'].flatten()
events_t = modalities['exp-marker']['time_stamps'].flatten()

# determine timepoint zero, and offset all signals by it
T0 = eeg_t[0]  # offset other signals to eeg
eeg_t -= T0
events_t -= T0

# remove negative timepoints and corresponding samples
events = events[events_t >= 0]
events_t = events_t[events_t >= 0]

# # plotting commands for debugging
# ti=4;plt.plot(gamepad_i, gamepad[ti, :]);plt.scatter(triggers_on_i[ti-4], gamepad[ti, triggers_on_i[ti-4]], marker='X', color='red');plt.show()
# ti=4;plt.plot(gamepad_i, np.diff(gamepad[ti, :], append=0));plt.scatter(triggers_on_i[ti-4], np.diff(gamepad[ti, :], append=0)[triggers_on_i[ti-4]], marker='X', color='red');plt.show()
# ti=4;plt.plot(gamepad_i, gamepad[ti, :]);plt.plot(gamepad_i, np.diff(gamepad[ti, :], append=0));plt.scatter(triggers_on_i[ti-4], np.diff(gamepad[ti, :], append=0)[triggers_on_i[ti-4]], marker='X', color='red');plt.show()

# create events from exp-marker stream and combine it with gamepad events
# first select eeg sample indices that are the closest to the event times
events_i_v2 = np.asarray([np.argmin(np.abs(eeg_t - et)) for et in events_t])
exp_events = np.stack([events_i_v2, np.zeros_like(events_i_v2), events], axis=1)

# event dictionary - event_name: marker
event_dict = {**{ename: einfo['marker'] for ename, einfo in exp_cfg['tasks'].items()},
                **{ename: einfo['marker'] for ename, einfo in exp_cfg['events'].items() if 'marker' in einfo}}

# preprocess numpy eeg, before encaptulating into mne raw
# common median referencing - substract median at each timepoint

# create raw mne eeg array and add events
# adding gamepad to eeg channels just complictes things at this point,
#   as filtering/preprocessing pipeline is separate for the two
# eeg_gamepad_info = mne.create_info(ch_types=['eeg'] * len(eeg_ch_names) + ['misc'] * len(trigger_lr_i),
#                                    ch_names=eeg_ch_names + ['L2', 'R2'], sfreq=eeg_sfreq)
# raw_w_gamepad = mne.io.RawArray(np.concatenate([eeg, gamepad[trigger_lr_i, :]]), eeg_gamepad_info)
eeg_info = mne.create_info(ch_types='eeg', ch_names=eeg_ch_names, sfreq=eeg_sfreq)
raw = mne.io.RawArray(eeg, eeg_info, verbose=verbose)
std_1020 = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(std_1020)
# raw.plot_sensors(show_names = True)
# plt.show()

# # add stim channel - not needed, events need to be provided separately to the epoching method anyways
stim_info = mne.create_info(['STI'], raw.info['sfreq'], ['stim'])  # separate events = stim channel
stim_raw = mne.io.RawArray(np.zeros((1, len(raw.times))), stim_info)
raw.add_channels([stim_raw], force_update_info=True)
raw.add_events(exp_events, stim_channel='STI')
#raw.plot(block = True, scalings=10, events=exp_events)

# 4 sec, min
from scipy.signal import find_peaks
eeg_p = raw.get_data()[4]
#eeg_p -= eeg_p.min()
eeg_p -= eeg_p.mean()
eeg_p = np.abs(eeg_p)
peaks = find_peaks(eeg_p, height = 5, distance=4 * raw.info['sfreq'])
manual_peak_index = np.where(eeg_p[100000:100500] == eeg_p[100000:100500].max())[0] + 100000
manual_peak_value = max(eeg_p[100000:100500])
peak_indexes = np.append(peaks[0], manual_peak_index[0])
peak_values = np.append(peaks[1]['peak_heights'], manual_peak_value)

event_timepoint = events_t * raw.info['sfreq']
indecies = np.searchsorted(event_timepoint, peak_indexes) - 1
closest_values = event_timepoint[indecies]
time_diffs = (peak_indexes - closest_values) / raw.info['sfreq']
marker_peak_combined = np.stack([peak_indexes, peak_values, closest_values, time_diffs])
plt.plot(eeg_p)
plt.scatter(marker_peak_combined[0], marker_peak_combined[1], s=10, c='red')
plt.show()

# raw.plot()
