import os, sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle
import mne
import pygsheets
from tqdm import tqdm
from matplotlib.colors import TwoSlopeNorm
from eeg_analysis import get_sessions


def concat_tfr_epochs(epochs_list: list, eeg_info, times, freqs, event_ids):
    all_events = np.concatenate([epochs.events for epochs in epochs_list], axis=0)
    all_events[:, 0] = np.arange(all_events.shape[0]) * 5000
    all_epochs = np.concatenate([epochs.data for epochs in epochs_list], axis=0)

    tfr_epochs = mne.time_frequency.EpochsTFR(eeg_info, all_epochs, times, freqs, verbose=False,
                                              events=all_events, event_id=event_ids)
    return tfr_epochs


def plot_avg_tfr(tfr_avg_ev: mne.time_frequency.AverageTFR, event_name, title, out_folder):
    vmin, vmax = -1, 1.5
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    channels = tfr_avg_ev.ch_names
    fig, axes = plt.subplots(1, len(channels) + 1, figsize=(14, 4),
                             gridspec_kw={"width_ratios": [10] * len(channels) + [1]})

    for ch, ax in enumerate(axes[:-1]):  # for each channel
        tfr_avg_ev.plot([ch], cmap="RdBu_r", cnorm=cnorm, axes=ax, colorbar=False,
                              show=False, mask=None, mask_style=None)
        ax.set_title(epochs.ch_names[ch], fontsize=10)
        ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event

    cbar_src = axes[0].images[-1] if len(axes[0].images) > 0 else axes[0].collections[0]
    fig.colorbar(cbar_src, cax=axes[-1]).ax.set_yscale("linear")
    fig.suptitle(f'ERDS: {title}')
    os.makedirs(out_folder, exist_ok=True)
    fig.savefig(f'{out_folder}/{subject}_erds_{event_name}.png')
    return fig


def load_subject_prep_data(prep_data_path, subject, part='task', channels=('C3', 'Cz', 'C4'), norm_c34_w_cz=False,
                           verbose=False):

    streams_path = f'{prep_data_path}/{subject}/{subject}_streams.h5'
    meta_path = f'{prep_data_path}/{subject}/{subject}_meta.pckl'

    # load hdf5 + meta data and run gen_erds_plots
    streams_data = h5py.File(streams_path, 'r')

    with open(meta_path, 'rb') as f:
        meta_data = pickle.load(f)

    eeg_info = meta_data['eeg_info']
    freqs = streams_data.attrs['freqs']
    times = streams_data.attrs[f'on_{part}_times'][:]
    on_events = streams_data[f'on_{part}_events'][:]
    iaf = np.median(np.asarray(meta_data['iafs']))
    event_ids = meta_data[f'{part}_event_ids']

    # C3-Cz, C4-Cz
    epochs_orig = streams_data[f'epochs_on_{part}'][:]
    tfr_orig = streams_data[f'tfr_epochs_on_{part}'][:]
    if norm_c34_w_cz:
        cz_i = eeg_info['ch_names'].index('Cz')
        c3_i = eeg_info['ch_names'].index('C3')
        c4_i = eeg_info['ch_names'].index('C4')
        tfr_orig[:, c3_i] = 2 * tfr_orig[:, c3_i] - tfr_orig[:, cz_i]
        tfr_orig[:, c4_i] = 2 * tfr_orig[:, c4_i] - tfr_orig[:, cz_i]

    # create epochs
    tmin = meta_data[f'on_{part}_times'][0]
    edited_on_events = np.copy(on_events)
    edited_on_events[:, 0] = np.arange(on_events.shape[0]) * 5000  # to avoid cross-session collision of events
    epochs = mne.EpochsArray(epochs_orig, eeg_info, events=edited_on_events, tmin=tmin, event_id=event_ids, baseline=None)

    tfr_epochs = mne.time_frequency.EpochsTFR(eeg_info, tfr_orig, times, freqs, verbose=verbose,
                                              events=edited_on_events, event_id=event_ids)
    tfr_epochs = tfr_epochs.pick(channels)

    return epochs, tfr_epochs, eeg_info, freqs, times, on_events, edited_on_events, event_ids, iaf


def load_experiments_table():
    try:
        key_path = 'keys\\experiment-377414-94e458f24082.json'
        gc = pygsheets.authorize(service_account_file=key_path)
        sheet = gc.open('Experiments')
        experiment_sheet = sheet[0]
    except:
        raise RuntimeError('no connection lol')
    return experiment_sheet.get_as_df()


prep_data_path = 'out_bl-1--0.05_tfr-multitaper-percent_reac-0.6_bad-95_f-2-40-100'
subjects, sessions = get_sessions(prep_data_path, subj_prefix='')
exps = load_experiments_table()

handedness_tfr = {'L': [], 'R': []}
for subject, sess in tqdm(zip(subjects, sessions), 'subjects'):
    epochs, tfr_epochs, eeg_info, freqs, times, on_events, edited_on_events, event_ids, iaf = \
        load_subject_prep_data(prep_data_path, subjects[0])

    subj_rows = exps.loc[exps['ParticipantID'] == subjects[0]]
    first_row = subj_rows.iloc[0]

    handedness_tfr[first_row['LRHandedness']].append({eid: tfr_epochs[eid].average() for eid in event_ids.keys()})
    # handedness_tfr[first_row['LRHandedness']].append(tfr_epochs)

# left_hand = concat_tfr_epochs(handedness_tfr['L'], eeg_info, times, freqs, event_ids)
# right_hand = concat_tfr_epochs(handedness_tfr['R'], eeg_info, times, freqs, event_ids)

for handedness in ['L', 'R']:
    if len(handedness_tfr[handedness]) == 0:
        print(f'no {handedness} handedness!', file=sys.stderr)
        continue

    for si, subj in enumerate(subjects):
        left_task = handedness_tfr[handedness][si]['left']
        right_task = handedness_tfr[handedness][si]['right']
        plot_avg_tfr(left_task, 'left', f'{subj}-left', out_folder=f'out/handedness/{handedness}')
        plot_avg_tfr(right_task, 'right', f'{subj}-right', out_folder=f'out/handedness/{handedness}')

print('done')
