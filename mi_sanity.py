import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import mne
import copy
import pandas as pd
import seaborn as sns

from matplotlib.colors import TwoSlopeNorm
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test


# sanity check to see if we can see ERDS differences between EEG recordings of motor imaginery


class MotorImageryDataset:
    def __init__(self, dataset='A01T.npz'):
        if not dataset.endswith('.npz'):
            dataset += '.npz'

        self.data = np.load(dataset)

        self.Fs = 250 # 250Hz from original paper

        # keys of data ['s', 'etyp', 'epos', 'edur', 'artifacts']

        self.raw = self.data['s'].T
        self.events_type = self.data['etyp'].T  # .T unnecessary..
        self.events_position = self.data['epos'].T
        self.events_duration = self.data['edur'].T
        self.artifacts = self.data['artifacts'].T

        # Types of motor imagery
        self.mi_types = {769: 'left', 770: 'right', 771: 'foot', 772: 'tongue', 783: 'unknown', 1023: 'rejected'}

    def get_trials_from_channel(self):

        channels = np.arange(22)  # leave ecog alone

        startrial_code = 768
        starttrial_events = self.events_type == startrial_code
        idxs = [i for i, x in enumerate(starttrial_events[0]) if x]
        print('all events:', np.unique(self.events_type[0, np.array(idxs) + 1]))

        # common median ref
        self.raw = common_median_ref(self.raw)

        trials, classes, cues, baselines, mis, breaks = [], [], [], [], [], []
        for index in idxs:
            type_e = self.events_type[0, index+1]
            class_e = self.mi_types[type_e]
            if class_e in ['unknown', 'rejected']:
                continue
            classes.append(class_e)

            trial_start = self.events_position[0, index]
            trial_stop = trial_start + self.events_duration[0, index]
            trial = self.raw[channels, trial_start:trial_stop]
            trials.append(trial)

            bl_dur = self.events_position[0, index + 1] - trial_start
            bl_stop = trial_start + bl_dur
            bl = self.raw[channels, trial_start:bl_stop]
            baselines.append(bl)

            cue_start = trial_start + bl_dur
            cue_dur = self.events_duration[0, index + 1]
            cue = self.raw[channels, cue_start:cue_start + cue_dur]
            cues.append(cue)

            mi_start = trial_start + self.Fs * 3
            mi_stop = trial_start + self.Fs * 6
            mi = self.raw[channels, mi_start:mi_stop]
            mis.append(mi)

            br_start = trial_start + self.Fs * 6
            br_stop = int(trial_start + self.Fs * 7.5)
            br = self.raw[channels, br_start:br_stop]
            breaks.append(br)

        return trials, classes, baselines, cues, mis, breaks


def common_median_ref(eeg):  # re-referencing can be beneficial when electrodes cover the head evenly
    common_med = np.median(eeg, axis=0, keepdims=True)
    return eeg - common_med


def notch(eeg, freqs=[50, 100], fs=250):  # 50 is already filtered out, not the harmonics tho
    return mne.filter.notch_filter(eeg, fs, freqs)


def baseline_corr(eeg, bl):
    return eeg - np.mean(bl, axis=-1, keepdims=True)


def gen_erp_plots(epochs, ds_name, out_folder):
    ### ERP STUFF: https://mne.tools/stable/auto_tutorials/evoked/30_eeg_erp.html#sphx-glr-auto-tutorials-evoked-30-eeg-erp-py

    # # erp plots on cues
    # cue_epochs['right'].plot_image(picks='C3', title='P: Right - C3', show=False)
    # cue_epochs['left'].plot_image(picks='C4', title='P: Left - C4', show=False)

    # cue_epochs['right'].plot_image(picks='C1', title='P: Right - C1', show=False)
    # cue_epochs['left'].plot_image(picks='C2', title='P: Left - C2', show=False)

    # cue_epochs['right'].plot_image(picks='C4', title='N: Right - C4', show=False)
    # cue_epochs['left'].plot_image(picks='C3', title='N: Left - C3', show=False)

    # cue_epochs['right'].plot_image(picks='C2', title='N: Right - C2', show=False)
    # cue_epochs['left'].plot_image(picks='C1', title='N: Left - C1')

    # # topomaps on cue
    # cue_epochs['left'].average().plot_topomap(times=[0.2, 0.4, 0.6, 0.8, 1.], average=0.4)
    # cue_epochs['right'].average().plot_topomap(times=[0.2, 0.4, 0.6, 0.8, 1.], average=0.4)
    
    # erp gfp
    epochs['left'].average().plot(gfp=True, show=False)
    plt.title('left gfp')
    plt.savefig(f'{out_folder}/{ds_name}_gfp_left.png')
    epochs['right'].average().plot(gfp=True, show=False)
    plt.title('right gfp')
    plt.savefig(f'{out_folder}/{ds_name}_gfp_right.png')

    # epochs['right'].pick(['C1', 'C3']).average().plot(gfp='only', show=False)
    # plt.title('P: right gfp: C1, C3')
    # epochs['left'].pick(['C2', 'C4']).average().plot(gfp='only', show=False)
    # plt.title('P: left gfp: C2, C4')

    # epochs['left'].pick(['C1', 'C3']).average().plot(gfp='only', show=False)
    # plt.title('N: left gfp: C1, C3')
    # epochs['right'].pick(['C2', 'C4']).average().plot(gfp='only', show=False)
    # plt.title('N: right gfp: C2, C4')

    # plt.show(block=False)
    plt.close('all')

    # compare evoked
    evokeds = dict(left=list(epochs['left'].crop(-0.5, 1).iter_evoked()), right=list(epochs['right'].crop(-0.5, 1).iter_evoked()))
    mne.viz.plot_compare_evokeds(evokeds, picks=['C1', 'C3'], combine='mean', title='compare C1, C3', show=False)
    plt.savefig(f'{out_folder}/{ds_name}_erp_cmp_left-elec.png')
    mne.viz.plot_compare_evokeds(evokeds, picks=['C2', 'C4'], combine='mean', title='compare C2, C4', show=False)
    plt.savefig(f'{out_folder}/{ds_name}_erp_cmp_right-elec.png')
    # plt.show(block=False)
    plt.close('all')


def gen_erds_plots(epochs, ds_name, event_id, out_folder, freqs, comp_time_freq=True, comp_tf_clusters=True,
                   channels=('C3', 'C1', 'C2', 'C4'), baseline=None, verbose=False, apply_baseline=False, copy=True):
    
    ### ERDS: https://mne.tools/dev/auto_examples/time_frequency/time_frequency_erds.html
    tmin, tmax = baseline[0], None
    vmin, vmax = -1, 1.5
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    if isinstance(epochs, mne.Epochs) or isinstance(epochs, mne.EpochsArray):
        if copy:
            epochs = epochs.load_data().copy().pick(channels)
        else:
            epochs = epochs.load_data().pick(channels)
        # if baseline is not None:
        #     epochs.apply_baseline(baseline, verbose=verbose)
        tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs, use_fft=True,
                             return_itc=False, average=False, verbose=verbose)#, decim=2)
    elif isinstance(epochs, mne.time_frequency.EpochsTFR):
        if copy:
            epochs = epochs.copy().pick(channels)
        else:
            epochs = epochs.pick(channels)
        tfr = epochs
    else:
        raise ValueError('epochs must be type Epochs or EpochsTFR')
    
    tfr.crop(tmin, tmax)
    if apply_baseline:
        tfr.apply_baseline(baseline, mode='percent', verbose=verbose)

    kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
                  buffer_size=None, out_type='mask')  # for cluster test

    if comp_time_freq:
        for event in event_id.keys():
            # select desired epochs for visualization
            tfr_ev = tfr[event]
            fig, axes = plt.subplots(1, len(channels) + 1, figsize=(14, 4),
                                     gridspec_kw={"width_ratios": [10] * len(channels) + [1]})
            
            for ch, ax in enumerate(axes[:-1]):  # for each channel

                if comp_tf_clusters:
                    # positive clusters
                    _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch], tail=1, **kwargs)
                    # negative clusters
                    _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch], tail=-1, **kwargs)

                    # note that we keep clusters with p <= 0.05 from the combined clusters
                    # of two independent tests; in this example, we do not correct for
                    # these two comparisons
                    c = np.stack(c1 + c2, axis=2)  # combined clusters
                    p = np.concatenate((p1, p2))  # combined p-values
                    mask = c[..., p <= 0.05].any(axis=-1)
                    mask_style = "mask"
                else:
                    mask = None
                    mask_style = None

                # plot TFR (ERDS map with masking)
                tfr_ev.average().plot([ch], cmap="RdBu_r", cnorm=cnorm, axes=ax, colorbar=False,
                                      show=False, mask=mask, mask_style=mask_style)

                ax.set_title(epochs.ch_names[ch], fontsize=10)
                ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
                if ch != 0:
                    ax.set_ylabel("")
                    ax.set_yticklabels("")
            
            # it worked so far, but out of the blue, axis.images started disappearing, so i needed
            # an alternative method to get the underlying img (or quadmesh) that defines the colorbar
            # fucking waste of time
            cbar_src = axes[0].images[-1] if len(axes[0].images) > 0 else axes[0].collections[0]
            fig.colorbar(cbar_src, cax=axes[-1]).ax.set_yscale("linear")
            fig.suptitle(f"ERDS ({event})")
            os.makedirs(out_folder, exist_ok=True)
            fig.savefig(f'{out_folder}/{ds_name}_erds_{event}.png')
        
        # plt.show(block=False)
        plt.close('all')

    ### more ERDS
    df = tfr.to_data_frame(time_format=None, long_format=True)

    # Map to frequency bands:
    freq_bounds = {'_': 0,
                   'delta': 3,
                   'theta': 7,
                   'alpha': 13,
                   'beta': 35,
                   'gamma': 140}
    df['band'] = pd.cut(df['freq'], list(freq_bounds.values()),
                        labels=list(freq_bounds)[1:])

    # Filter to retain only relevant frequency bands:
    freq_bands_of_interest = ['delta', 'theta', 'alpha', 'beta']
    df = df[df.band.isin(freq_bands_of_interest)]
    df['band'] = df['band'].cat.remove_unused_categories()

    # Order channels for plotting:
    df['channel'] = df['channel'].cat.reorder_categories(channels, ordered=True)

    g = sns.FacetGrid(df, row='band', col='channel', margin_titles=True)
    g.map(sns.lineplot, 'time', 'value', 'condition', n_boot=10)
    axline_kw = dict(color='black', linestyle='dashed', linewidth=0.5, alpha=0.5)
    g.map(plt.axhline, y=0, **axline_kw)
    g.map(plt.axvline, x=0, **axline_kw)
    g.set(ylim=(None, 1.5))
    g.set_axis_labels("Time (s)", "ERDS (%)")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.add_legend(ncol=2, loc='lower center')
    g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)
    g.fig.savefig(f'{out_folder}/{ds_name}_erds.png')

    # plt.show(block=False)
    plt.close('all')


# doc: https://bbci.de/competition/iv/desc_2a.pdf


if __name__ == '__main__':
    base_dir = '../bcidatasetIV2a'
    datasets = sorted(glob(f'{base_dir}/*T.npz'))

    channel_names = ['Fz'] + list(map(str, range(2, 7, 1))) + ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6'] + list(map(str, range(14, 20))) + ['Pz', '21', '22']
    drop_chans = [c for c in channel_names if c not in ['Cz', 'Fz', 'Pz'] + [f'C{i}' for i in range(1, 7)]]
    info = mne.create_info(channel_names, 250, 'eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    baseline_t = -2.5  # sets tmin for epochs, -2 puts start of cue at 0, -3 puts start of MI at 0
    out_folder = 'out/cue-mi-mid_start'
    do_per_subject_comp = False
    os.makedirs(out_folder, exist_ok=True)

    all_trials, all_classes, all_baselines, all_cues, all_mis, all_breaks = [], [], [], [], [], []
    for ds in datasets:
        print(ds)
        ds_name = os.path.basename(ds)
        ds_name = ds_name[:ds_name.rfind('.')]

        datasetA1 = MotorImageryDataset(ds)
        trials, classes, baselines, cues, mis, breaks = datasetA1.get_trials_from_channel()
        trials, classes, baselines, cues, mis, breaks = map(np.asarray, [trials, classes, baselines, cues, mis, breaks])
        # electrodes referenced, sampled 250Hz, bandpass 0.5 - 100Hz, 50 Hz notch filter
        
        # manual baseline correction
        # trials, baselines, cues, mis, breaks = map(lambda eeg: baseline_corr(eeg, baselines), [trials, baselines, cues, mis, breaks])

        # add to cross-subject data
        for s, z in zip([trials, classes, baselines, cues, mis, breaks], [all_trials, all_classes, all_baselines, all_cues, all_mis, all_breaks]):
            z.append(s)

        if do_per_subject_comp:
            # events
            n_epochs = trials.shape[0]
            event_id = {e: i for i, e in enumerate(np.unique(classes))}
            events = np.asarray([event_id[c] for c in classes])
            # https://github.com/mne-tools/mne-python/blob/96a4bc2e928043a16ab23682fc818cf0a3e78aef/mne/utils/numerics.py#L198
            events = np.c_[np.arange(n_epochs), np.zeros(n_epochs, int), events]

            mi_epochs = mne.EpochsArray(mis, info=info, events=events, event_id=event_id)
            cue_epochs = mne.EpochsArray(cues, info=info, events=events, event_id=event_id)
            cue_n_mi_epochs = mne.EpochsArray(np.concatenate([cues, mis], axis=-1), info=info,
                                              events=events, event_id=event_id)
            bl_n_cue_epochs = mne.EpochsArray(np.concatenate([baselines, cues], axis=-1), info=info,
                                              events=events, event_id=event_id, tmin=baseline_t)
            bl_n_cue_n_mi_epochs = mne.EpochsArray(np.concatenate([baselines, cues, mis], axis=-1), info=info,
                                                   events=events, event_id=event_id, tmin=baseline_t)

            # remove channels
            mi_epochs, cue_epochs, cue_n_mi_epochs, bl_n_cue_epochs, bl_n_cue_n_mi_epochs = map(lambda eeg: eeg.drop_channels(drop_chans),
                [mi_epochs, cue_epochs, cue_n_mi_epochs, bl_n_cue_epochs, bl_n_cue_n_mi_epochs])

            # set montage
            mi_epochs, cue_epochs, cue_n_mi_epochs, bl_n_cue_epochs, bl_n_cue_n_mi_epochs = map(lambda eeg: eeg.set_montage(montage),
                [mi_epochs, cue_epochs, cue_n_mi_epochs, bl_n_cue_epochs, bl_n_cue_n_mi_epochs])

            # select epochs to use and create plots
            epochs = bl_n_cue_n_mi_epochs
            gen_erp_plots(epochs, ds_name, out_folder)
            gen_erds_plots(epochs, f'{ds_name}_mu', event_id, out_folder, freqs=np.arange(7, 13, 0.1),
                           baseline=(-1.5, 0), apply_baseline=True)
            gen_erds_plots(epochs, f'{ds_name}_beta', event_id, out_folder, freqs=np.arange(13, 30, 0.5),
                           baseline=(-1.5, 0), apply_baseline=True)
            gen_erds_plots(epochs, f'{ds_name}_smr', event_id, out_folder, freqs=np.arange(13, 15, 0.1),
                           baseline=(-1.5, 0), apply_baseline=True)
            gen_erds_plots(epochs, f'{ds_name}_allf', event_id, out_folder, freqs=np.arange(7, 40, 0.5),
                           baseline=(-1.5, 0), apply_baseline=True)


    ### run analysis on all epochs from all subjects
    all_trials, all_classes, all_baselines, all_cues, all_mis, all_breaks = \
        map(lambda x: np.concatenate(x, axis=0), [all_trials, all_classes, all_baselines, all_cues, all_mis, all_breaks])

    # events
    n_epochs = trials.shape[0]
    event_id = {e: i for i, e in enumerate(np.unique(classes))}
    events = np.asarray([event_id[c] for c in classes])
    events = np.c_[np.arange(n_epochs), np.zeros(n_epochs, int), events]

    # create cross-subject epochs
    mi_epochs = mne.EpochsArray(mis, info=info, events=events, event_id=event_id)
    cue_epochs = mne.EpochsArray(cues, info=info, events=events, event_id=event_id)
    cue_n_mi_epochs = mne.EpochsArray(np.concatenate([cues, mis], axis=-1), info=info, events=events, event_id=event_id)
    bl_n_cue_epochs = mne.EpochsArray(np.concatenate([baselines, cues], axis=-1), info=info, events=events, event_id=event_id, tmin=baseline_t)
    bl_n_cue_n_mi_epochs = mne.EpochsArray(np.concatenate([baselines, cues, mis], axis=-1), info=info, events=events, event_id=event_id, tmin=baseline_t)

    epochs = bl_n_cue_n_mi_epochs

    # remove channels
    mi_epochs, cue_epochs, cue_n_mi_epochs, bl_n_cue_epochs = map(lambda eeg: eeg.drop_channels(drop_chans),
        [mi_epochs, cue_epochs, cue_n_mi_epochs, bl_n_cue_epochs])

    # set montage
    mi_epochs, cue_epochs, cue_n_mi_epochs, bl_n_cue_epochs = map(lambda eeg: eeg.set_montage(montage),
        [mi_epochs, cue_epochs, cue_n_mi_epochs, bl_n_cue_epochs])

    # gen_erp_plots(epochs, 'combined', out_folder)
    gen_erds_plots(epochs, 'combined_mu', event_id, out_folder, freqs=np.arange(8, 13, 0.1),
                   baseline=(-1.5, 0), apply_baseline=True)
    gen_erds_plots(epochs, 'combined_beta', event_id, out_folder, freqs=np.arange(13, 30, 0.5),
                   baseline=(-1.5, 0), apply_baseline=True)
    gen_erds_plots(epochs, 'combined_smr', event_id, out_folder, freqs=np.arange(13.1, 15.1, 0.1),
                   baseline=(-1.5, 0), apply_baseline=True)
    gen_erds_plots(epochs, 'combined_allf', event_id, out_folder, freqs=np.arange(7.5, 40.5, 0.5),
                   baseline=(-1.5, 0), apply_baseline=True)
