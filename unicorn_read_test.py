import pyxdf
import struct
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from copy import deepcopy


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
