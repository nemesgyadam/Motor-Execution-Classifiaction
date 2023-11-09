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

from eeg_analysis import plot_erds


def concat_tfr_epochs(epochs_list: list, eeg_info, times, freqs, event_ids):
    all_events = np.concatenate([epochs.events for epochs in epochs_list], axis=0)
    all_events[:, 0] = np.arange(all_events.shape[0]) * 5000
    all_epochs = np.concatenate([epochs.data for epochs in epochs_list], axis=0)

    tfr_epochs = mne.time_frequency.EpochsTFR(eeg_info, all_epochs, times, freqs, verbose=False,
                                              events=all_events, event_id=event_ids)
    return tfr_epochs


def plot_avg_tfr(tfr_avg_ev: mne.time_frequency.AverageTFR, subj, event_name, title, out_folder, ch_names):
    vmin, vmax = -1, 1.5
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    channels = tfr_avg_ev.ch_names
    fig, axes = plt.subplots(1, len(channels) + 1, figsize=(14, 4),
                             gridspec_kw={"width_ratios": [10] * len(channels) + [1]})

    for ch, ax in enumerate(axes[:-1]):  # for each channel
        tfr_avg_ev.plot([ch], cmap="RdBu_r", cnorm=cnorm, axes=ax, colorbar=False,
                              show=False, mask=None, mask_style=None)
        ax.set_title(ch_names, fontsize=10)
        ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event

    cbar_src = axes[0].images[-1] if len(axes[0].images) > 0 else axes[0].collections[0]
    fig.colorbar(cbar_src, cax=axes[-1]).ax.set_yscale("linear")
    fig.suptitle(f'ERDS: {title}')
    os.makedirs(out_folder, exist_ok=True)
    fig.savefig(f'{out_folder}/{subj}_erds_{event_name}.png')
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


def main(part='task'):  # task | pull
    prep_data_path = 'out_bl-1--0.05_tfr-multitaper-percent_reac-0.6_bad-95_f-2-40-100'
    subjects, sessions = get_sessions(prep_data_path, subj_prefix='')
    exps = load_experiments_table()
    channels = ('C3', 'Cz', 'C4')

    for handedness in ['L', 'R']:
        os.makedirs(f'out/handedness/{part}/{handedness}', exist_ok=True)

    iafs = {}
    handedness_tfr = {'L': [], 'R': []}
    handedness_subj = {'L': [], 'R': []}
    eeg_info, freqs, times, event_ids = None, None, None, None

    for subj, sess in tqdm(zip(subjects, sessions), 'subjects', total=len(subjects)):
        epochs, tfr_epochs, eeg_info, freqs, times, on_events, edited_on_events, event_ids, iaf = \
            load_subject_prep_data(prep_data_path, subj, part, channels)
        iafs[subj] = iaf

        subj_rows = exps.loc[exps['ParticipantID'] == subj]
        first_row = subj_rows.iloc[0]

        handedness = first_row['LRHandedness']
        handedness_tfr[handedness].append({eid: tfr_epochs[eid].average() for eid in event_ids.keys()})
        handedness_subj[handedness].append(subj)
        # handedness_tfr[first_row['LRHandedness']].append(tfr_epochs)

        # plot ERDS
        # fois = {'theta': (4, 7), 'wide_mu': (iaf - 2, iaf + 2), 'tight_mu': (iaf - 1, iaf + 1),
        #         'wide_beta': (13, 30), 'tight_beta': (18, 30), 'tighter_beta': (16, 24),
        #         'wide_gamma': (30, 48), 'low_gamma': (25, 40), 'high_gamma': (40, 48)}
        fois = {'wide_mu': (iaf - 2, iaf + 2), 'tight_mu': (iaf - 1, iaf + 1),
                'wide_beta': (13, 30), 'tight_beta': (18, 30), 'tighter_beta': (16, 24)}

        fig = plot_erds(tfr_epochs, fois, event_ids, channels, freqs, times, shift=False)
        fig_shifted = plot_erds(tfr_epochs, fois, event_ids, channels, freqs, times, shift=True)

        out_folder = f'out/handedness/{part}/{handedness}'
        fig.savefig(f'{out_folder}/{subj}_erds_fois.png', bbox_inches='tight', pad_inches=0, dpi=350)
        fig_shifted.savefig(f'{out_folder}/{subj}_erds_fois_shifted.png', bbox_inches='tight', pad_inches=0, dpi=350)
        plt.close('all')

    # left_hand = concat_tfr_epochs(handedness_tfr['L'], eeg_info, times, freqs, event_ids)
    # right_hand = concat_tfr_epochs(handedness_tfr['R'], eeg_info, times, freqs, event_ids)

    all_events, all_task_epochs_data, by_handedness_iafs = [], [], []
    new_event_ids = {'left': 0, 'right': 1}

    for handedness in ['L', 'R']:
        print('=' * 50, handedness, '=' * 50)
        if len(handedness_tfr[handedness]) == 0:
            print(f'no {handedness} handedness!', file=sys.stderr)
            continue

        # per subject
        out_folder = f'out/handedness/{part}/{handedness}'
        for subj, tfr in zip(handedness_subj[handedness], handedness_tfr[handedness]):
            left_task = tfr['left']
            right_task = tfr['right']
            plot_avg_tfr(left_task, subj, 'left', f'{subj}-left', out_folder=out_folder, ch_names=channels)
            plot_avg_tfr(right_task, subj, 'right', f'{subj}-right', out_folder=out_folder, ch_names=channels)

        # combined
        for task in ['left', 'right']:
            task_avg = np.stack([tfrs[task].data for tfrs in handedness_tfr[handedness]]).mean(axis=0)
            task_avg = mne.time_frequency.AverageTFR(eeg_info, task_avg, times, freqs, nave=handedness_tfr[handedness])
            plot_avg_tfr(task_avg, 'ALL', task, f'ALL-{task}', out_folder=out_folder, ch_names=channels)

        # plot ERDS: given handedness, left and right hand
        mean_iaf = np.mean(list(iafs.values()))
        fois = {'wide_mu': (mean_iaf - 2, mean_iaf + 2), 'tight_mu': (mean_iaf - 1, mean_iaf + 1),
                'wide_beta': (13, 30), 'tight_beta': (18, 30), 'tighter_beta': (16, 24)}

        left_tfrs = [tfrs['left'].data for tfrs in handedness_tfr[handedness]]
        right_tfrs = [tfrs['right'].data for tfrs in handedness_tfr[handedness]]
        task_epochs_data = np.stack(left_tfrs + right_tfrs)
        events = np.stack([np.arange(0, len(left_tfrs) + len(right_tfrs)) * 10000,  # time
                           np.zeros(len(left_tfrs) + len(right_tfrs), dtype=np.int32),  # whatev
                           np.concatenate([np.zeros(len(left_tfrs), dtype=np.int32),  # event_ids..
                                           np.ones(len(right_tfrs), dtype=np.int32)])], axis=1)  # 0->left, 1->right

        by_handedness_iafs.append(mean_iaf)
        all_events.append(events)
        all_task_epochs_data.append(task_epochs_data)

        task_epochs = mne.time_frequency.EpochsTFR(eeg_info, task_epochs_data, times, freqs,
                                                   events=events, event_id=new_event_ids)
        fig = plot_erds(task_epochs, fois, new_event_ids, channels, freqs, times, shift=False)
        fig_shifted = plot_erds(task_epochs, fois, new_event_ids, channels, freqs, times, shift=True)
        fig.savefig(f'{out_folder}/ALL_erds_fois_hand-{handedness}.png', bbox_inches='tight', pad_inches=0, dpi=350)
        fig_shifted.savefig(f'{out_folder}/ALL_erds_fois_hand-{handedness}_shifted.png', bbox_inches='tight', pad_inches=0, dpi=350)
        plt.close('all')

    # left and right handed combined
    all_events = np.concatenate(all_events)
    all_task_epochs_data = np.concatenate(all_task_epochs_data)
    by_handedness_iafs = np.mean(by_handedness_iafs)
    fois = {'wide_mu': (by_handedness_iafs - 2, by_handedness_iafs + 2), 'tight_mu': (by_handedness_iafs - 1, by_handedness_iafs + 1),
            'wide_beta': (13, 30), 'tight_beta': (18, 30), 'tighter_beta': (16, 24)}
    all_task_epochs = mne.time_frequency.EpochsTFR(eeg_info, all_task_epochs_data, times, freqs,
                                                   events=all_events, event_id=new_event_ids)

    fig = plot_erds(all_task_epochs, fois, new_event_ids, channels, freqs, times, shift=False)
    fig_shifted = plot_erds(all_task_epochs, fois, new_event_ids, channels, freqs, times, shift=True)
    fig.savefig(f'out/handedness/{part}/ALL_erds_fois_hand-ALL.png', bbox_inches='tight', pad_inches=0, dpi=350)
    fig_shifted.savefig(f'out/handedness/{part}/ALL_erds_fois_hand-ALL_shifted.png', bbox_inches='tight', pad_inches=0, dpi=350)
    plt.close('all')

    # plot iafs
    print('king iaf:', iafs['158c8bc7'])
    plt.hist(list(iafs.values()), bins=30)
    plt.savefig(f'out/iafs.png', bbox_inches='tight', pad_inches=0, dpi=350)
    plt.close('all')

    print('done')


if __name__ == '__main__':
    main(part='task')
    main(part='pull')
