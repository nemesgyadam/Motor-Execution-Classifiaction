import pyxdf
import struct
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from copy import deepcopy
import json


# def eeg2voltage(f: float):
#      eegv = struct.unpack('>i', b'\x00' + payload[(3+ch*3):(6+ch*3)])[0]
#     # apply two’s complement to the 32-bit signed integral value if the sign bit is set
#     if (eegv & 0x00800000):
#         eegv = eegv | 0xFF000000
#     eeg[ch] = float(eegv) * 4500000. / 50331642.


# https://gist.github.com/robertoostenveld/6f5f765268847f684585be9e60ecfb67
# https://github.com/unicorn-bi/Unicorn-Suite-Hybrid-Black/blob/master/Unicorn%20.NET%20API/UnicornLSL/UnicornLSL/MainUI.cs#L338

# recording path
base_path = '../recordings'
subject = '0717b399'
session = 6
rec_path = f'../recordings/sub-{subject}/ses-S{session:03d}/eeg/sub-{subject}_ses-S{session:03d}_task-me-l-r-lr_run-001_eeg.xdf'

# experiment config
exp_cfg_path = 'config/lr_finger/exp_me_l_r_lr_stim-w-dots.json'
with open(exp_cfg_path, 'rt') as f:
    exp_cfg = json.load(f)

# channel info
eeg_sfreq = 250
gamepad_sfreq = 125
shared_sfreq = max(eeg_sfreq, gamepad_sfreq)

eeg_ch_names = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
gamepad_ch_names = ['LeftX', 'LeftY', 'RightX', 'RightY', 'L2', 'R2']
# eeg_info = mne.create_info(ch_types='eeg', ch_names=eeg_ch_names, sfreq=eeg_sfreq)
# gamepad_info = mne.create_info(ch_types='misc', ch_names=gamepad_ch_names, sfreq=gamepad_sfreq)

# load streams from xdf
streams, header  = pyxdf.load_xdf(rec_path)
modalities = {s['info']['name'][0]: s for s in streams}

eeg = modalities['Unicorn']['time_series'].T[:8, :]
eeg_t = modalities['Unicorn']['time_stamps']

events = modalities['exp-marker']['time_series'].flatten()
events_t = modalities['exp-marker']['time_stamps'].flatten()

gamepad = modalities['Gamepad']['time_series'].T
gamepad_t = modalities['Gamepad']['time_stamps'].flatten()

assert round((1/np.diff(gamepad_t)).mean(), 2) == 125.00, f'gamepad should be sampled at 125, not: {round((1/np.diff(gamepad_t)).mean(), 2)}'

# determine timepoint zero, and offset all signals by it
T0 = eeg_t[0]  # offset other signals to eeg
eeg_t -= T0
events_t -= T0
gamepad_t -= T0

# remove negative timepoints and corresponding samples
events = events[events_t >= 0]
events_t = events_t[events_t >= 0]
gamepad = gamepad[:, gamepad_t >= 0]
gamepad_t = gamepad_t[gamepad_t >= 0]

# upsample gamepad to eeg freq
gamepad_f = interp1d(gamepad_t, gamepad, kind='linear', copy=True,
                     fill_value='extrapolate', bounds_error=False)
gamepad_t = eeg_t
gamepad = gamepad_f(gamepad_t)

# get index of timepoints (as opposed to seconds)
# needed for creating mne events
eeg_i = (eeg_t * shared_sfreq).astype(np.int32)
events_i = (events_t * shared_sfreq).astype(np.int32)
gamepad_i = (gamepad_t * shared_sfreq).astype(np.int32)
max_i = max(eeg_i[-1], events_i[-1], gamepad_i[-1])

# create events from gamepad action
# events need to be in the format of:
#   https://mne.tools/dev/generated/mne.io.RawArray.html#mne.io.RawArray.add_events
trigger_pull_dist_t = 3  # in sec
trigger_on_event_markers = [601, 602]  # L2, R2
trigger_off_event_markers = [611, 612]  # L2, R2
trigger_lr_i = [4, 5]  # L2, R2 indices in gamepad stream


def detect_trigger_pulls(x, sfreq, pull_dist_t, pull_t=0.1):
    x = deepcopy(x)
    dx = np.diff(x, append=0)
    dx[dx < 0] = 0
    peaks = find_peaks(dx, distance=int(pull_dist_t * sfreq))[0]

    drng = int(pull_t * sfreq)
    good_boi_peak = [dx[peak - drng:peak + drng].sum() > .5 for peak in peaks]
    return peaks[good_boi_peak]


triggers_on_i  = [gamepad_i[detect_trigger_pulls( gamepad[trigger_i, :], shared_sfreq, trigger_pull_dist_t)] for trigger_i in trigger_lr_i]
triggers_off_i = [gamepad_i[detect_trigger_pulls(-gamepad[trigger_i, :], shared_sfreq, trigger_pull_dist_t)] for trigger_i in trigger_lr_i]

# TODO check if the same number of on-off happened, pair on-offs, remove ones that can't be paired
print('trigger on/off times should be equal:')
print('L2:', triggers_on_i[0].shape[0], '==', triggers_off_i[0].shape[0])
print('R2:', triggers_on_i[1].shape[0], '==', triggers_off_i[1].shape[0])

# ti=4;plt.plot(gamepad_i, gamepad[ti, :]);plt.scatter(triggers_on_i[ti-4], gamepad[ti, triggers_on_i[ti-4]], marker='X', color='red');plt.show()
# ti=4;plt.plot(gamepad_i, np.diff(gamepad[ti, :], append=0));plt.scatter(triggers_on_i[ti-4], np.diff(gamepad[ti, :], append=0)[triggers_on_i[ti-4]], marker='X', color='red');plt.show()
# ti=4;plt.plot(gamepad_i, gamepad[ti, :]);plt.plot(gamepad_i, np.diff(gamepad[ti, :], append=0));plt.scatter(triggers_on_i[ti-4], np.diff(gamepad[ti, :], append=0)[triggers_on_i[ti-4]], marker='X', color='red');plt.show()

trigger_on_markers = [np.repeat(ev_mark, len(i)) for i, ev_mark in zip(triggers_on_i, trigger_on_event_markers)]
trigger_off_markers = [np.repeat(ev_mark, len(i)) for i, ev_mark in zip(triggers_off_i, trigger_off_event_markers)]

# events data structure: n_events x 3: [[sample_i, 0, marker]...]
trigger_i = np.concatenate(triggers_on_i + triggers_off_i)
trigger_markers = np.concatenate(trigger_on_markers + trigger_off_markers)
gamepad_lr_events = np.stack([trigger_i, np.zeros_like(trigger_i), trigger_markers], axis=1)

# create events from exp-marker stream and combine it with gamepad events
exp_events = np.stack([events_i, np.zeros_like(events_i), events], axis=1)
all_events = np.concatenate([gamepad_lr_events, exp_events], axis=0)

# create raw mne eeg array and add events
eeg_gamepad_info = mne.create_info(ch_types=['eeg'] * len(eeg_ch_names) + ['misc'] * len(trigger_lr_i),
                                   ch_names=eeg_ch_names + ['L2', 'R2'], sfreq=eeg_sfreq)
raw = mne.io.RawArray(np.concatenate([eeg, gamepad[trigger_lr_i, :]]), eeg_gamepad_info)

stim_info = mne.create_info(['STI'], raw.info['sfreq'], ['stim'])  # separate events=stim channel
stim_raw = mne.io.RawArray(np.zeros((1, len(raw.times))), stim_info)
raw.add_channels([stim_raw], force_update_info=True)
raw.add_events(all_events, stim_channel='STI')

event_dict = {'left': 111, 'right': 112, 'left-right': 113,'nothing': 131, 'baseline':110, 'break':140, 'session_beg': 100, 
              'session_end': 200, 'eyes-open-beg': 101,'eyes-open-end': 201, 'eyes-closed-beg': 102,
              'eyes-closed-end': 202, 'eyes_move_beg': 103, 'eyes_move_end': 203,
              **{f'{trig}-on': mark for mark, trig in zip(trigger_on_event_markers, ['L2', 'R2'])},
              **{f'{trig}-off': mark for mark, trig in zip(trigger_off_event_markers, ['L2', 'R2'])}}

# preprocess eeg
bandpass_freq = (0.5, 80)
notch_freq = [50, 100]

# TODO remove baseline

filt_raw_eeg = raw.copy().pick_types(eeg=True).filter(*bandpass_freq).notch_filter(notch_freq)

# TODO autoreject for epoch removal:
#   https://github.com/autoreject/autoreject/
#   https://autoreject.github.io/stable/auto_examples/plot_autoreject_workflow.html

# break up into epochs
task_event_ids = [task['marker'] for task in exp_cfg['tasks'].values()]
break_event_id = exp_cfg['events']['break']['marker']

epochs_on_task = mne.Epochs(filt_raw_eeg, all_events, event_id=task_event_ids, baseline=(None, 0), 
                            tmin=-1, tmax=exp_cfg['event-duration']['task'][0])
epochs_on_break = mne.Epochs(filt_raw_eeg, all_events, event_id=break_event_id, baseline=(None, 0),
                             tmin=-exp_cfg['event-duration']['task'][0] / 2, tmax=exp_cfg['event-duration']['break'][0])

epochs_on_task.apply_baseline()
epochs_on_break.apply_baseline()

# some plots
filt_raw_eeg.plot(scalings=80, events=all_events)
epochs_on_task.plot(scalings=50, events=all_events, n_epochs=4, title='locked on task')
epochs_on_break.plot(scalings=50, events=all_events, n_epochs=4, title='locked on break')

# ERPs
epochs_on_task.average().plot(scalings=50)
epochs_on_break.average().plot(scalings=50)
# TODO ERPs per task

# TODO spectral analysis

pass
