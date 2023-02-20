import pyxdf
import struct
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import find_peaks


# def eeg2voltage(f: float):
#      eegv = struct.unpack('>i', b'\x00' + payload[(3+ch*3):(6+ch*3)])[0]
#     # apply two’s complement to the 32-bit signed integral value if the sign bit is set
#     if (eegv & 0x00800000):
#         eegv = eegv | 0xFF000000
#     eeg[ch] = float(eegv) * 4500000. / 50331642.


# https://gist.github.com/robertoostenveld/6f5f765268847f684585be9e60ecfb67
# https://github.com/unicorn-bi/Unicorn-Suite-Hybrid-Black/blob/master/Unicorn%20.NET%20API/UnicornLSL/UnicornLSL/MainUI.cs#L338


rec_path1 = '../recordings/sub-6808dfab/ses-S001/eeg/sub-6808dfab_ses-S001_task-me-l-r-lr_run-001_eeg.xdf'
rec_path2 = '../recordings/sub-6808dfab/ses-S002/eeg/sub-6808dfab_ses-S002_task-me-l-r-lr_run-001_eeg.xdf'
rec_path3 = '../recordings/sub-6808dfab/ses-S003/eeg/sub-6808dfab_ses-S003_task-me-l-r-lr_run-001_eeg.xdf'

# channel info
eeg_sfreq = 250
gamepad_sfreq = 125
shared_sfreq = max(eeg_sfreq, gamepad_sfreq)

eeg_ch_names = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
gamepad_ch_names = ['LeftX', 'LeftY', 'RightX', 'RightY', 'L2', 'R2']
eeg_info = mne.create_info(ch_types='eeg', ch_names=eeg_ch_names, sfreq=eeg_sfreq)
gamepad_info = mne.create_info(ch_types='misc', ch_names=gamepad_ch_names, sfreq=gamepad_sfreq)

# load streams from xdf
streams, header  = pyxdf.load_xdf(rec_path1)
modalities = {s['info']['name'][0]: s for s in streams}

eeg = modalities['Unicorn']['time_series'].T[:8, :]
eeg_t = modalities['Unicorn']['time_stamps']

events = modalities['exp-marker']['time_series'].flatten()
events_t = modalities['exp-marker']['time_stamps'].flatten()

gamepad = modalities['Gamepad']['time_series'].T
gamepad_t = modalities['Gamepad']['time_stamps'].flatten()

assert round((1/np.diff(gamepad_t)).mean(), 2) == 125.00, f'gamepad should be sampled at 125, not: {round((1/np.diff(gamepad_t)).mean(), 2)}'

# determine timepoint zero, and offset all signals by it
T0 = min(gamepad_t.min(), events_t.min(), eeg_t.min())
eeg_t -= T0
events_t -= T0
gamepad_t -= T0

# get index of timepoints (as opposed to seconds)
eeg_i = (eeg_t * shared_sfreq).astype(np.int32)
events_i = (events_t * shared_sfreq).astype(np.int32)
gamepad_i = (eeg_t * shared_sfreq).astype(np.int32)
max_i = max(eeg_i.max(), events_i.max(), gamepad_i.max())

# TODO add gamepad analog signals to eeg signal (at same sample rate, separate channel)

# create events from gamepad action
# events need to be in the format of:
#   https://mne.tools/dev/generated/mne.io.RawArray.html#mne.io.RawArray.add_events
gamepad_event_markers = [601, 602]  # L2, R2
gamepad_lr_i = [4, 5]  # L2, R2 indices in gamepad stream
gamepad_lr_fun_on = lambda x: find_peaks(np.diff(x, append=0), distance=gamepad_sfreq * 2, threshold=0)
gamepad_lr_fun_off = lambda x: find_peaks(np.diff(-x, append=0), distance=gamepad_sfreq * 2, threshold=0)

# TODO create events from exp-marker stream and combine it with gamepad events

# TODO create raw eeg+gamepad array and add events

raw_eeg = mne.io.RawArray(eeg, eeg_info)
event_dict = {'left': 111, 'right': 112, 'left-right': 113,'nothing': 131, 'baseline':110, 'break':140, 'session_beg': 100,
              'session_end':200, 'eyes-open-beg':101,'eyes-open-end':201, 'eyes-closed-beg': 102,
              'eyes-closed-end': 202, 'eyes_move_beg': 103, 'eyes_move_end': 203}

raw_array = np.full(shape=streams[1]['time_stamps'].shape, fill_value=raw.first_samp)
to_extract = np.full(shape=streams[1]['time_stamps'].shape, fill_value=streams[1]['time_stamps'][0])
correct_time_stamp_array = (streams[1]['time_stamps'] - to_extract) * 250

filt_raw = raw.copy().filter(0.5, 124).notch_filter([50, 100])
filt_raw.plot(scalings=100, events=events)

exit(0)

data, header = pyxdf.load_xdf(rec_path3)
modalities = dict()  # {'exp-marker': None, 'Gamepad': None, 'Gamepad Events': None, 'Unicorn': None}
for d in data:
    modalities[d['info']['name'][0]] = d

sess_data = modalities['Unicorn']['time_series'].T
eeg = sess_data[:8, :]


# consecutive same absolute values
diff = np.diff(eeg, axis=1)
same_lenz = []
for eegchan in range(diff.shape[0]):

    gotit = False
    count = 0
    same_lens = []
    for i in range(diff.shape[1]):
        d = diff[eegchan, i]
        if d == 0 and not gotit:
            gotit = True
        elif d != 0 and gotit:
            gotit = False
            same_lens.append(count)#((i - count, count))
            count = 0
        
        if d == 0:
            count += 1

    same_lenz.append(same_lens)


samez = np.array([s for ss in same_lenz for s in ss])
print(np.unique(samez, return_counts=True))

# print(same_lenz)

exit(1)


# TODO talan bajtcsuszas lesz, az utso bajt mindig ugyanaz, az utolso 2 sokszor hasonlit, de channel-enkent mas a hasonlosag az utsoelotti bajtnal
for eegchan in range(8):
    uniq_1sttolast_bytes, uniq_2ndtolast_bytes, uniq_3rdtolast_bytes = [], [], []

    for i in range(0, eeg.shape[1], 1):  # 10000 step
        uniq_1sttolast_bytes.append(bytes(eeg[eegchan, i])[-1])
        uniq_2ndtolast_bytes.append(bytes(eeg[eegchan, i])[-2])
        uniq_3rdtolast_bytes.append(bytes(eeg[eegchan, i])[-3])
        # print(bytes(eeg[eegchan, i])[-2:])
    
    print('-'*50)
    a = np.array(uniq_2ndtolast_bytes)
    # plt.plot(a - a.min())

    fixed = a - a.min()

    GOOD = []

    eegvs = []
    for i in range(0, eeg.shape[1], 1):
        GOOD.append(bytes(eeg[eegchan, i])[-1])


        b1 = bytes(eeg[eegchan, i])[:-2] + bytes([fixed[i]]) + b'\x00'
        b2 = struct.unpack('>i', b1)[0]

        eegv = struct.unpack('>i', b'\x00\x00' + bytes(eeg[eegchan, i])[:-2])[0]  # aaaaaaaaah
        if (eegv & 0x00800000):
            eegv = eegv | 0xFF000000
        eegv = float(eegv) * 4500000. / 50331642.
        eegvs.append(eegv)
        # print(eegv)
    plt.plot(eegvs)
    plt.show()
    break

    # print(f'chan{eegchan+1} 1sttolast: {np.unique(uniq_1sttolast_bytes)} | {len(np.unique(uniq_1sttolast_bytes))}')
    # print(f'chan{eegchan+1} 2ndtolast: {np.unique(uniq_2ndtolast_bytes)} | {len(np.unique(uniq_2ndtolast_bytes))}')
    # print(f'chan{eegchan+1} 3rdtolast: {np.unique(uniq_3rdtolast_bytes)}')

print('done')
plt.show()
