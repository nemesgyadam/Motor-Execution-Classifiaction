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

from eeg_analysis import plot_erds, plot_erds_raw


def concat_tfr_epochs(epochs_list: list, eeg_info, times, freqs, event_ids):
    all_events = np.concatenate([epochs.events for epochs in epochs_list], axis=0)
    all_events[:, 0] = np.arange(all_events.shape[0]) * 10000
    all_epochs = np.concatenate([epochs.data for epochs in epochs_list], axis=0)

    tfr_epochs = mne.time_frequency.EpochsTFR(eeg_info, all_epochs, times, freqs, verbose=False,
                                              events=all_events, event_id=event_ids)
    return tfr_epochs


def plot_avg_tfr(tfr_avg_ev: mne.time_frequency.AverageTFR, subj, event_name, title, out_folder, ch_names):
    vmin, vmax = [-.5, 1.]
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    channels = tfr_avg_ev.ch_names
    fig, axes = plt.subplots(1, len(channels) + 1, figsize=(14, 4),
                             gridspec_kw={"width_ratios": [10] * len(channels) + [1]})

    for ch, ax in enumerate(axes[:-1]):  # for each channel
        tfr_avg_ev.plot([ch], cmap="RdBu_r", cnorm=cnorm, axes=ax, colorbar=False,
                              show=False, mask=None, mask_style=None)
        ax.set_title(ch_names[ch], fontsize=10)
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


def main(part, prep_data_path, recompute_grping=False):  # task | pull
    subjects, sessions = get_sessions(prep_data_path, subj_prefix='')
    exps = load_experiments_table()
    channels = ('C3', 'Cz', 'C4')
    eid_map = {'left': 'Left finger', 'right': 'Right finger', 'left-pull': 'Left finger', 'right-pull': 'Right finger',
               'left-right': 'Left-right fingers', 'left-right-pull': 'Left-right fingers', 'nothing': 'Nothing'}

    iafs = {}
    handedness_tfr = {'L': [], 'R': []}
    handedness_subj = {'L': [], 'R': []}
    age_tfr = {'lt-25': [], 'gt-25': []}
    age_subj = {'lt-25': [], 'gt-25': []}
    gender_tfr = {'f': [], 'm': []}
    gender_subj = {'f': [], 'm': []}
    famil_tfr = {'couple times': [], 'regular': []}
    famil_subj = {'couple times': [], 'regular': []}

    handedness_erds = {'L': [], 'R': []}
    age_erds = {'lt-25': [], 'gt-25': []}
    gender_erds = {'f': [], 'm': []}
    famil_erds = {'couple times': [], 'regular': []}
    eeg_info, freqs, times, event_ids = None, None, None, None

    # TODO !!!
    subjects = ['0717b399']

    if recompute_grping:
        for subj in tqdm(subjects, 'subjects', total=len(subjects)):
            epochs, tfr_epochs, eeg_info, freqs, times, on_events, edited_on_events, event_ids, iaf = \
                load_subject_prep_data(prep_data_path, subj, part, channels)
            iafs[subj] = iaf
            # if not ('left' in event_ids and 'right' in event_ids):
            #     print('!' * 30, 'SKIPPING SUBJECT: ', subj, '!' * 30, file=sys.stderr)
            #     print('event ids:', event_ids, file=sys.stderr)
            #     continue

            subj_rows = exps.loc[exps['ParticipantID'] == subj]
            first_row = subj_rows.iloc[0]

            # IAF
            # https://neuroscenter.com/wp-content/uploads/2022/07/Individual-Alpha-Peak-Frequency-an-Important-Biomarker-for-Live-Z-Score-Training-Neurofeedback-in-Adolescents-with-Learning-Disabilities.pdf
            iaf_ok = iaf if 7.5 < iaf < 12 else 10
            if iaf_ok != iaf:
                print(f'IAF({iaf}) is off for subject {subj}', file=sys.stderr)

            # frequencies of interest
            # fois = {'theta': (4, 7), 'wide_mu': (iaf - 2, iaf + 2), 'tight_mu': (iaf - 1, iaf + 1),
            #         'wide_beta': (13, 30), 'tight_beta': (18, 30), 'tighter_beta': (16, 24),
            #         'wide_gamma': (30, 48), 'low_gamma': (25, 40), 'high_gamma': (40, 48)}
            # fois = {'wide_mu': (iaf - 2, iaf + 2), 'tight_mu': (iaf - 1, iaf + 1),
            #         'wide_beta': (13, 30), 'tight_beta': (18, 30), 'tighter_beta': (16, 24)}

            # from intro of: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6381427/
            # fois = {'theta-delta': (2, 7),
            #         'low_alpha': (iaf_ok - 3, iaf_ok), 'high_alpha': (iaf_ok, iaf_ok + 3),
            #         'alpha': (iaf_ok - 3, iaf_ok + 3),
            #         'low_beta': (13, 20), 'mid_beta': (18, 25), 'high_beta': (25, 35)}

            fois = {'Alpha': (iaf_ok - 3, iaf_ok + 3), 'Beta': (13, 30)}

            bands = {fname: (f0 <= freqs) & (freqs <= f1) for fname, (f0, f1) in fois.items()}

            # setup groupings
            handedness = first_row['LRHandedness']
            age = first_row['Age']
            age_str = 'lt-25' if age < 25 else 'gt-25'
            gender = first_row['Gender']
            famil = first_row['GamepadFamiliarity']
            avgs = {eid_map[eid]: tfr_epochs[eid].average() for eid in event_ids.keys()}
            erdss = {eid_map[eid]: {bname: tfr_epochs[eid].data[:, :, band, :].mean(axis=2).astype(np.float16)
                                    for bname, band in bands.items()}
                     for eid in event_ids.keys()}

            if handedness in handedness_tfr:
                handedness_tfr[handedness].append(avgs)
                handedness_subj[handedness].append(subj)
                handedness_erds[handedness].append(erdss)
            if age in age_tfr:
                age_tfr[age_str].append(avgs)
                age_subj[age_str].append(subj)
                age_erds[age_str].append(erdss)
            if gender in gender_tfr:
                gender_tfr[gender].append(avgs)
                gender_subj[gender].append(subj)
                gender_erds[gender].append(erdss)
            if famil in famil_tfr:
                famil_tfr[famil].append(avgs)
                famil_subj[famil].append(subj)
                famil_erds[famil].append(erdss)

            fig = plot_erds(tfr_epochs, fois, event_ids, channels, freqs, times, shift=False)
            fig_shifted = plot_erds(tfr_epochs, fois, event_ids, channels, freqs, times, shift=True)

            out_folder = f'out/handedness/{part}/{handedness}'
            os.makedirs(out_folder, exist_ok=True)
            fig.savefig(f'{out_folder}/{subj}_erds_fois.png', bbox_inches='tight', pad_inches=0, dpi=250)
            fig_shifted.savefig(f'{out_folder}/{subj}_erds_fois_shifted.png', bbox_inches='tight', pad_inches=0, dpi=250)
            plt.close('all')

        # left_hand_tfrs = {task: concat_tfr_epochs([tfr[task] for tfr in handedness_tfr['L']], eeg_info, times, freqs, event_ids)
        #                   for task in ['left', 'right']}
        # with open('out/post_anal_handedness.pkl', 'wb') as f:
        #     pickle.dump({'L': concat_tfr_epochs(handedness_tfr['L'], eeg_info, times, freqs, event_ids),
        #                  'R': concat_tfr_epochs(handedness_tfr['R'], eeg_info, times, freqs, event_ids),
        #                  'subj': handedness_subj}, f)

        grping_names = ['handedness', 'age', 'gender', 'famil']
        grping_tfrs = [handedness_tfr, age_tfr, gender_tfr, famil_tfr]
        grping_subjs = [handedness_subj, age_subj, gender_subj, famil_subj]
        grping_erdss = [handedness_erds, age_erds, gender_erds, famil_erds]
        by_grping_iafs = {grping: [] for grping in grping_names}

        with open('tmp/panal.pkl', 'wb') as f:
            pickle.dump([grping_names, grping_tfrs, grping_subjs, grping_erdss, by_grping_iafs, iafs,
                         eeg_info, freqs, times, event_ids], f)

    else:  # load back grping
        with open('tmp/panal.pkl', 'rb') as f:
            grping_names, grping_tfrs, grping_subjs, grping_erdss, by_grping_iafs,\
                iafs, eeg_info, freqs, times, event_ids = \
                pickle.load(f)

    exit()  # TODO rm !!!

    # plot that shit
    all_events, all_task_epochs_data = [], []
    new_event_ids = {'Left finger': 0, 'Right finger': 1, 'Left-right fingers': 2}

    for grping_name, grping_tfr, grping_subj, grping_erds in zip(grping_names, grping_tfrs, grping_subjs, grping_erdss):

        for grp in grping_tfr.keys():
            print('=' * 50, grp, '=' * 50)
            if len(grping_tfr[grp]) == 0:
                print(f'no {grp} {grping_name}!', file=sys.stderr)
                continue

            # per subject
            out_folder = f'out/{grping_name}/{part}/{grp}'
            for subj, tfr in zip(grping_subj[grp], grping_tfr[grp]):
                left_task = tfr[eid_map['left']]
                right_task = tfr[eid_map['right']]
                plot_avg_tfr(left_task, subj, 'Left finger', f'{subj}-left', out_folder=out_folder, ch_names=channels)
                plot_avg_tfr(right_task, subj, 'Right finger', f'{subj}-right', out_folder=out_folder, ch_names=channels)

            # combined
            stored_event_ids = grping_tfr[next(iter(grping_tfr.keys()))][0].keys()
            for task in stored_event_ids:
                task_avg = np.stack([tfrs[task].data for tfrs in grping_tfr[grp]]).mean(axis=0)
                task_avg = mne.time_frequency.AverageTFR(eeg_info, task_avg, times, freqs, nave=grping_tfr[grp])
                plot_avg_tfr(task_avg, 'ALL', task, f'ALL-{task}', out_folder=out_folder, ch_names=channels)

            # plot ERDS: given handedness or other grouping, left and right hand
            mean_iaf = np.mean(list(iafs.values()))
            # fois = {'wide_mu': (mean_iaf - 2, mean_iaf + 2), 'tight_mu': (mean_iaf - 1, mean_iaf + 1),
            #         'wide_beta': (13, 30), 'tight_beta': (18, 30), 'tighter_beta': (16, 24)}
            fois = {'theta-delta': (2, 7),
                    'low_alpha': (mean_iaf - 3, mean_iaf), 'high_alpha': (mean_iaf, mean_iaf + 3),
                    'alpha': (mean_iaf - 3, mean_iaf + 3),
                    'low_beta': (13, 20), 'mid_beta': (18, 25), 'high_beta': (25, 35)}

            left_tfrs = [tfrs[eid_map['left']].data for tfrs in grping_tfr[grp]]
            right_tfrs = [tfrs[eid_map['right']].data for tfrs in grping_tfr[grp]]
            lr_tfrs = [tfrs[eid_map['left-right']].data for tfrs in grping_tfr[grp]]

            # plot all together for given handedness or other grouping - avg of avg of session
            task_epochs_data = np.stack(left_tfrs + right_tfrs + lr_tfrs)
            events = np.stack([np.arange(0, len(left_tfrs) + len(right_tfrs) + len(lr_tfrs)) * 10000,  # time, every 10 sec, whatev
                               np.zeros(len(left_tfrs) + len(right_tfrs) + len(lr_tfrs), dtype=np.int32),  # whatev
                               np.concatenate([np.zeros(len(left_tfrs), dtype=np.int32),  # event_ids: 0->left
                                               np.ones(len(right_tfrs), dtype=np.int32),  # 1->right
                                               np.zeros(len(lr_tfrs), dtype=np.int32) + 2])], axis=1)  # 2->left-right

            by_grping_iafs[grping_name].append(mean_iaf)
            all_events.append(events)
            all_task_epochs_data.append(task_epochs_data)

            task_epochs = mne.time_frequency.EpochsTFR(eeg_info, task_epochs_data, times, freqs,
                                                       events=events, event_id=new_event_ids)
            fig = plot_erds(task_epochs, fois, new_event_ids, channels, freqs, times, shift=False)
            fig_shifted = plot_erds(task_epochs, fois, new_event_ids, channels, freqs, times, shift=True)
            fig.savefig(f'{out_folder}/ALL_erds_fois_{grping_name}-{grp}.png', bbox_inches='tight', pad_inches=0, dpi=350)
            fig_shifted.savefig(f'{out_folder}/ALL_erds_fois_{grping_name}-{grp}_shifted.png', bbox_inches='tight', pad_inches=0, dpi=350)

            # plot grand avg ERDS
            erdss = grping_erds[grp]
            band_names = next(iter(erdss[0].values())).keys()
            ev_ids = erdss[0].keys()
            comb_erdss = {ev: {band: np.concatenate([erds[ev][band] for erds in erdss])
                               for band in band_names}
                          for ev in ev_ids}
            fig_comb = plot_erds_raw(comb_erdss, new_event_ids, band_names, channels, freqs, times)
            fig_comb.savefig(f'{out_folder}/ALL_erds_fois_grand_{grping_name}-{grp}.png', bbox_inches='tight', pad_inches=0, dpi=350)

            plt.close('all')

    # # left and right handed combined
    # all_events = np.concatenate(all_events)
    # all_task_epochs_data = np.concatenate(all_task_epochs_data)
    # by_handedness_iafs = np.mean(by_handedness_iafs)
    # fois = {'wide_mu': (by_handedness_iafs - 2, by_handedness_iafs + 2), 'tight_mu': (by_handedness_iafs - 1, by_handedness_iafs + 1),
    #         'wide_beta': (13, 30), 'tight_beta': (18, 30), 'tighter_beta': (16, 24)}
    # all_task_epochs = mne.time_frequency.EpochsTFR(eeg_info, all_task_epochs_data, times, freqs,
    #                                                events=all_events, event_id=new_event_ids)
    #
    # fig = plot_erds(all_task_epochs, fois, new_event_ids, channels, freqs, times, shift=False)
    # fig_shifted = plot_erds(all_task_epochs, fois, new_event_ids, channels, freqs, times, shift=True)
    # fig.savefig(f'out/handedness/{part}/ALL_erds_fois_hand-ALL.png', bbox_inches='tight', pad_inches=0, dpi=350)
    # fig_shifted.savefig(f'out/handedness/{part}/ALL_erds_fois_hand-ALL_shifted.png', bbox_inches='tight', pad_inches=0, dpi=350)
    # plt.close('all')

    # plot iafs
    print('king iaf:', iafs['158c8bc7'])
    plt.hist(list(iafs.values()), bins=30)
    plt.savefig(f'out/iafs.png', bbox_inches='tight', pad_inches=0, dpi=350)
    plt.close('all')

    print('done')


# TODO try set norm_c34_w_cz=True


if __name__ == '__main__':
    # out_bl-1--0.05_tfr-multitaper-percent_reac-0.6_bad-95_f-2-40-100
    # out_bl-1--0.05_tfr-multitaper-logratio_reac-0.5_bad-95_f-2-40-100
    main(part='pull', prep_data_path='out_bl-1--0.05_tfr-multitaper-logratio_reac-0.6_bad-95_f-2-40-100', recompute_grping=True)
    main(part='task', prep_data_path='out_bl-1--0.05_tfr-multitaper-logratio_reac-0.6_bad-95_f-2-40-100', recompute_grping=False)
