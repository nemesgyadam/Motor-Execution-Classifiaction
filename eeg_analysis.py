import os
import sys

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
import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.colors import TwoSlopeNorm
from matplotlib.pyplot import cm
from common import CircBuff


def define_target_events2(events, reference_id, target_ids, sfreq, tmin, tmax, new_id=None, fill_na=None,
                          occur=True, or_occur=False):
    """Define new events by (non)-co-occurrence of existing events.

    This function can be used to evaluate events depending on the
    temporal lag to another event. For example, this can be used to
    analyze evoked responses which were followed by a button press within
    a defined time window.

    Parameters
    ----------
    events : ndarray
        Array as returned by mne.find_events.
    reference_id : int
        The reference event. The event defining the epoch of interest.
    target_ids : list[int]
        The target events that should (occur=True) or should not (occur=False) happen in the vicinity of reference_id.
    sfreq : float
        The sampling frequency of the data.
    tmin : float
        The lower limit in seconds from the target event.
    tmax : float
        The upper limit border in seconds from the target event.
    new_id : int
        New ID for the new event.
    fill_na : int | None
        Fill event to be inserted if target is not available within the time
        window specified. If None, the 'null' events will be dropped.
    occur: bool
        If True, check for cooccurrance, if False, check for avoidance.
    or_occur: bool
        If True, check for co-/non-occurrance of multiple targets with an OR connection, not the default AND

    Returns
    -------
    new_events : ndarray
        The new defined events.
    lag : ndarray
        Time lag between reference and target in milliseconds.
    """

    assert fill_na is None

    if new_id is None:
        new_id = reference_id

    tsample = 1e3 / sfreq
    imin = int(tmin * sfreq)
    imax = int(tmax * sfreq)

    new_events = []
    # lag = []  # no meaning of lag
    for event in events.copy().astype(int):
        if event[2] == reference_id:
            lower = event[0] + imin
            upper = event[0] + imax
            
            tcrit = (events[:, 0] > lower) & (events[:, 0] < upper)
            targ_crits = [events[:, 2] == tid for tid in target_ids]
            crit = [tcrit & targ_crit for targ_crit in targ_crits]

            # # plot neighboring events for testing
            # if not occur:
            #     plt.figure()
            #     plt.vlines(events[:, 0], -1, 1)
            #     plt.xlim(lower - 10, upper)
            #     for et, _, ev in events[tcrit]:
            #         plt.text(et - 4, .6, f'{ev}')
            #     plt.show()

            if occur:
                if not or_occur:  # default
                    crit = np.logical_and.reduce([c.any() for c in crit])
                else:  # OR
                    crit = np.logical_or.reduce([c.any() for c in crit])
                # crit = np.logical_and.reduce([np.any(tcrit & (events[:, 2] == tid)) for tid in target_ids])
            else:  # non-co-occurance
                if not or_occur:
                    crit = ~np.logical_or.reduce([c.any() for c in crit])
                else:  # OR, but negated, u know
                    crit = ~np.logical_and.reduce([c.any() for c in crit])
                # crit = ~np.logical_or.reduce([tcrit & (events[:, 2] == tid) for tid in target_ids])

            if crit:
                event[2] = new_id
                new_events += [event]

    new_events = np.array(new_events)
    return new_events if new_events.any() else np.array([]), None  # no lag computed, fuck it

# https://gist.github.com/robertoostenveld/6f5f765268847f684585be9e60ecfb67
# https://github.com/unicorn-bi/Unicorn-Suite-Hybrid-Black/blob/master/Unicorn%20.NET%20API/UnicornLSL/UnicornLSL/MainUI.cs#L338


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


class WTFException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class TDomPrepper:
    def __init__(self, epoch_len: int, sfreq: int, ch_names: list, bandpass_freq=(.5, 80), notch_freq=(50, 100),
                 common=np.mean, tmin_max=(-1.5, 2), crop_t=(-.2, None), baseline=(-1., -.05), filter_percentile=None):
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.bandpass_freq = bandpass_freq
        self.notch_freq = notch_freq
        self.common = common
        self.tmin_max = tmin_max
        self.baseline = baseline
        self.filter_percentile = filter_percentile

        crop_def = (0, np.inf)
        crop_start = 0 if crop_t[0] is None else int((tmin_max[0] - crop_t[0]) * sfreq)
        crop_end = epoch_len if crop_t[1] is None else int(np.abs(tmin_max[0]) * sfreq) + int(crop_t[1] * sfreq)
        self.crop_i = (crop_start, crop_end)

        self.eeg_info = mne.create_info(ch_types='eeg', ch_names=ch_names, sfreq=sfreq)
        self.std_1020 = mne.channels.make_standard_montage('standard_1020')
        self.epoch_event = np.array([[sfreq * np.abs(tmin_max[0]), 0, 420]], dtype=np.int32)
        self.event_id = {'whatever': 420}

        # filtering
        self.filter = mne.filter.create_filter(
            np.zeros((len(ch_names), epoch_len)), sfreq,
            l_freq=bandpass_freq[0], h_freq=bandpass_freq[1],
            l_trans_bandwidth='auto', h_trans_bandwidth=5,
            filter_length='auto', phase='zero', fir_window='hamming',
            fir_design="firwin")

        # tracked variables
        self.epoch_peak2peaks = CircBuff(1000, ordered=False)
        self.z_means = CircBuff(1000, len(ch_names), ordered=False)
        self.z_vars = CircBuff(1000, len(ch_names), ordered=False)

    def __call__(self, eeg: np.ndarray):  # channel x time
        if self.common:
            eeg -= self.common(eeg, axis=0)

        # fast-filter
        to_filt = np.pad(eeg, ((0, 0), (len(self.filter) * 1, len(self.filter) * 1)), mode='edge')
        filt_fun = lambda x: np.convolve(np.convolve(self.filter, x)[::-1], self.filter)[::-1][len(self.filter) - 1 + len(self.filter): -len(self.filter) - 1 - len(self.filter)]
        filt_eeg = np.apply_along_axis(filt_fun, axis=1, arr=to_filt)  # TODO test if same with filt_raw_eeg below (w/o notch)

        raw = mne.io.RawArray(eeg, self.eeg_info)
        raw.set_montage(self.std_1020)

        # filter
        filt_raw_eeg = raw.copy().pick_types(eeg=True).filter(*self.bandpass_freq, h_trans_bandwidth=5, n_jobs=1) \
            .notch_filter(self.notch_freq, trans_bandwidth=1, n_jobs=1)  # TODO create filter ahead

        # epoching
        epoch = mne.Epochs(filt_raw_eeg, self.epoch_event, event_id=self.event_id, baseline=None,
                           tmin=-self.baseline[0], tmax=None)
        epoch.apply_baseline(self.baseline)
        epoch = epoch.get_data()[0]

        if self.filter_percentile:
            peak2peak = np.max(np.max(epoch, axis=-1) - np.min(epoch, axis=-1))
            self.epoch_peak2peaks.add(peak2peak)
            perc = np.percentile(self.epoch_peak2peaks.get(), self.filter_percentile)

            if peak2peak > perc:
                epoch = None

        # TODO option to resample

        # crop & standardize
        if epoch:
            epoch = epoch[:, self.crop_i[0]:self.crop_i[1]]  # TODO test

            self.z_means.add(epoch.mean(axis=-1))
            self.z_vars.add(epoch.var(axis=-1))

            mean = self.z_means.get().mean(axis=0, keepdims=True)
            var = self.z_vars.get().mean(axis=0, keepdims=True)
            epoch = (epoch - mean) / np.sqrt(var)

        return epoch


def preprocess_session(rec_base_path, rec_name, subject, session, exp_cfg_path,
                       eeg_sfreq=250, gamepad_sfreq=125, bandpass_freq=(0.5, 80), notch_freq=(50, 100),
                       freqs=np.arange(2, 50, 0.2), do_plot=False, reaction_tmax=1.,
                       n_jobs=4, verbose=False, fig_output_path='out', motor_imaginary=False):
    
    print('-' * 50 + f'\nPROCESSING: {subject}/{session:03d}\n' + '-' * 50)
    os.makedirs(fig_output_path, exist_ok=True)

    # recording path
    rec_path = f'{rec_base_path}/sub-{subject}/ses-S{session:03d}/eeg/sub-{subject}_ses-S{session:03d}_task-me-l-r-lr_run-001_eeg.xdf'

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

    if len(eeg) == 0:
        raise WTFException('how')

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
    # needed for creating mne events - not really, as sample index it will be slightly off ackchyually
    # eeg_i = (eeg_t * shared_sfreq).astype(np.int32)
    # events_i = (events_t * shared_sfreq).astype(np.int32)
    # gamepad_i = (gamepad_t * shared_sfreq).astype(np.int32)

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

    triggers_on_i  = [detect_trigger_pulls( gamepad[trigger_i, :], shared_sfreq, trigger_pull_dist_t)
                      for trigger_i in trigger_lr_i]
    triggers_off_i = [detect_trigger_pulls(-gamepad[trigger_i, :], shared_sfreq, trigger_pull_dist_t)
                      for trigger_i in trigger_lr_i]

    # check if the same number of on-off happened, pair on-offs, remove ones that can't be paired
    print('trigger on/off times should be equal:')
    print('  L2:', triggers_on_i[0].shape[0], '==', triggers_off_i[0].shape[0])
    print('  R2:', triggers_on_i[1].shape[0], '==', triggers_off_i[1].shape[0])

    # # plotting commands for debugging
    # ti=4;plt.plot(gamepad_i, gamepad[ti, :]);plt.scatter(triggers_on_i[ti-4], gamepad[ti, triggers_on_i[ti-4]], marker='X', color='red');plt.show()
    # ti=4;plt.plot(gamepad_i, np.diff(gamepad[ti, :], append=0));plt.scatter(triggers_on_i[ti-4], np.diff(gamepad[ti, :], append=0)[triggers_on_i[ti-4]], marker='X', color='red');plt.show()
    # ti=4;plt.plot(gamepad_i, gamepad[ti, :]);plt.plot(gamepad_i, np.diff(gamepad[ti, :], append=0));plt.scatter(triggers_on_i[ti-4], np.diff(gamepad[ti, :], append=0)[triggers_on_i[ti-4]], marker='X', color='red');plt.show()

    # events data structure: n_events x 3: [[sample_i, 0, marker]...]
    trigger_on_markers  = [np.repeat(ev_mark, len(i)) for i, ev_mark in zip(triggers_on_i, trigger_on_event_markers)]
    trigger_off_markers = [np.repeat(ev_mark, len(i)) for i, ev_mark in zip(triggers_off_i, trigger_off_event_markers)]

    trigger_i = np.concatenate(triggers_on_i + triggers_off_i)
    trigger_markers = np.concatenate(trigger_on_markers + trigger_off_markers)
    gamepad_lr_events = np.stack([trigger_i, np.zeros_like(trigger_i), trigger_markers], axis=1)

    # create events from exp-marker stream and combine it with gamepad events
    # first select eeg sample indices that are the closest to the event times
    events_i_v2 = np.asarray([np.argmin(np.abs(eeg_t - et)) for et in events_t])
    exp_events = np.stack([events_i_v2, np.zeros_like(events_i_v2), events], axis=1)
    all_events = np.concatenate([gamepad_lr_events, exp_events], axis=0)

    # event dictionary - event_name: marker
    event_dict = {**{ename: einfo['marker'] for ename, einfo in exp_cfg['tasks'].items()},
                  **{ename: einfo['marker'] for ename, einfo in exp_cfg['events'].items() if 'marker' in einfo},
                  **{f'{trig}-on': mark for mark, trig in zip(trigger_on_event_markers, ['L2', 'R2'])},
                  **{f'{trig}-off': mark for mark, trig in zip(trigger_off_event_markers, ['L2', 'R2'])}}
    
    # preprocess numpy eeg, before encaptulating into mne raw
    # common mean referencing - substract mean at each timepoint
    eeg -= np.mean(eeg, axis=0)
    
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
    # stim_info = mne.create_info(['STI'], raw.info['sfreq'], ['stim'])  # separate events = stim channel
    # stim_raw = mne.io.RawArray(np.zeros((1, len(raw.times))), stim_info)
    # raw.add_channels([stim_raw], force_update_info=True)
    # raw.add_events(all_events, stim_channel='STI')
    
    # filter and run tfr on the whole recording (if possible)
    filt_raw_eeg = raw.copy().pick_types(eeg=True).filter(*bandpass_freq, h_trans_bandwidth=5, n_jobs=n_jobs, verbose=verbose) \
        .notch_filter(notch_freq, trans_bandwidth=1, n_jobs=n_jobs, verbose=verbose)
    
    # TODO autoreject for epoch removal:
    #   https://github.com/autoreject/autoreject/
    #   https://autoreject.github.io/stable/auto_examples/plot_autoreject_workflow.html
    #   https://mne.tools/stable/auto_tutorials/preprocessing/20_rejecting_bad_data.html#rejecting-epochs-based-on-channel-amplitude

    # filter epochs, remove when stimulus-action mismatch
    # create new event for when the right button is pressed
    left_success_marker = int(f'{event_dict["left"]}{event_dict["L2-on"]}')
    right_success_marker = int(f'{event_dict["right"]}{event_dict["R2-on"]}')
    lr_success_marker = int(f'{event_dict["left-right"]}2')
    nothing_success_marker = int(f'{event_dict["nothing"]}0')

    left_success_pull_marker = int(f'{event_dict["L2-on"]}{event_dict["left"]}')
    right_success_pull_marker = int(f'{event_dict["R2-on"]}{event_dict["right"]}')
    lr_success_pull_marker = int(f'2{event_dict["left-right"]}')

    # for motor imaginary tasks, the gamepad actions are ignored, and all trials succeed by default
    tmin = -.1 if motor_imaginary else 1e-4

    left_success = mne.event.define_target_events(all_events, event_dict['left'],
                                                  event_dict['left'] if motor_imaginary else event_dict['L2-on'],
                                                  shared_sfreq, tmin=tmin, tmax=reaction_tmax,
                                                  new_id=left_success_marker)
    right_success = mne.event.define_target_events(all_events, event_dict['right'],
                                                   event_dict['right'] if motor_imaginary else event_dict['R2-on'],
                                                   shared_sfreq, tmin=tmin, tmax=reaction_tmax,
                                                   new_id=right_success_marker)
    lr_success = define_target_events2(all_events, event_dict["left-right"],
                                       [event_dict["left-right"]] if motor_imaginary else [event_dict['L2-on'], event_dict['R2-on']],
                                       shared_sfreq, tmin=tmin, tmax=reaction_tmax, new_id=lr_success_marker,
                                       occur=True, or_occur=False)
    # when nothing should be pressed, it just becomes a fucking headache (implemented it myself in the end)
    # = mne is straight dogshit
    # nothing_success is occur=False, so no need to have different behavior for motor imaginary (gamepad is not used)
    nothing_success = define_target_events2(all_events, event_dict["nothing"], [event_dict['L2-on'], event_dict['R2-on']],
                                            shared_sfreq, tmin=1e-4, tmax=exp_cfg['event-duration']['task'][0],
                                            new_id=nothing_success_marker, occur=False, or_occur=False)

    if not motor_imaginary:
        left_success_pull = mne.event.define_target_events(all_events, event_dict['L2-on'], event_dict['left'],
                                                           shared_sfreq, tmin=-reaction_tmax, tmax=1e-4,
                                                           new_id=left_success_pull_marker)
        right_success_pull = mne.event.define_target_events(all_events, event_dict['R2-on'], event_dict['right'],
                                                            shared_sfreq, tmin=-reaction_tmax, tmax=1e-4,
                                                            new_id=right_success_pull_marker)
        lr_success_pull = mne.event.define_target_events(all_events, event_dict['R2-on'], event_dict['left-right'],
                                                         shared_sfreq, tmin=-reaction_tmax, tmax=1e-4,
                                                         new_id=lr_success_pull_marker)
    else:  # motor imaginary
        left_success_pull = right_success_pull = lr_success_pull = np.empty((0, 3), dtype=all_events.dtype), None
    # TODO anchored to left pull, not the avg of left and right pull

    # stats on delay
    if do_plot:
        fig = plt.figure()
        plt.title('button press delay')
        plt.hist(left_success[1], bins=20, label='left', alpha=.5)
        plt.hist(right_success[1], bins=20, label='right', alpha=.5)
        # TODO implement define_target_events2() proper lag array handling
        # plt.hist(lr_success[1], bins=10, label='left-right', alpha=.5)
        plt.legend()
        fig.savefig(f'{fig_output_path}/btn_press_delay_{rec_name}.png')
    
    # do the same for the break event
    break_marker = exp_cfg['events']['break']['marker']
    break_success_marker = int(f'{break_marker}0')
    break_success = define_target_events2(all_events, break_marker,
                                          [break_marker] if motor_imaginary else [event_dict['L2-off'], event_dict['R2-off']],
                                          shared_sfreq, tmin=tmin, tmax=reaction_tmax, new_id=break_success_marker,
                                          occur=True, or_occur=True)
    
    all_events = np.concatenate([all_events, left_success[0], right_success[0], lr_success[0], nothing_success[0],
                                 break_success[0], left_success_pull[0], right_success_pull[0],
                                 lr_success_pull[0]], axis=0)
    
    # some stats
    num_orig = [(all_events[:, 2] == event_dict[ev]).sum() for ev in ['left', 'right', 'left-right', 'nothing', 'break']]
    num_succ = [(all_events[:, 2] == ev).sum() for ev in [left_success_marker, right_success_marker, lr_success_marker,
                                                          nothing_success_marker, break_success_marker]]
    print('num of events successful/originally:',
          {name: f'{nsucc}/{nold}' for name, nold, nsucc in zip(['left', 'right', 'left-right', 'nothing'], num_orig, num_succ)})

    # order events
    all_events = all_events[np.argsort(all_events[:, 0]), :]

    # # test gamepad event alignment
    # gev = np.concatenate([left_success[0], right_success[0], lr_success[0], nothing_success[0],
    #                       break_success[0], left_success_pull[0], right_success_pull[0],
    #                       lr_success_pull[0]], axis=0)
    # graw = mne.io.RawArray(gamepad[[4, 5]], mne.create_info(ch_types='misc', ch_names=['L2', 'R2'],
    #                                                         sfreq=shared_sfreq))
    # graw.plot(scalings=3, events=gev, duration=10.)

    return dict(success_markers={'left': left_success_marker, 'right': right_success_marker,
                                 'left-right': lr_success_marker, 'nothing': nothing_success_marker,
                                 'break': break_success_marker, 'left-pull': left_success_pull_marker,
                                 'right-pull': right_success_pull_marker, 'left-right-pull': lr_success_pull_marker},
                filt_raw_eeg=filt_raw_eeg, gamepad=gamepad, events=all_events, event_dict=event_dict,
                eeg_info=eeg_info, freqs=freqs, eeg_ch_names=eeg_ch_names)


def epoch_on_task(sess_id: int, prep_results: dict, exp_cfg_path, freqs, baseline=(None, 0), tfr_mode='cwt',
                  tfr_baseline_mode='percent', n_cycles=None, n_jobs=4, verbose=False, filter_percentile=None):
    filt_raw_eeg = prep_results['filt_raw_eeg']
    all_events = prep_results['events']

    with open(exp_cfg_path, 'rt') as f:
        exp_cfg = json.load(f)
    
    # epoching
    task_event_ids = {task: marker for task, marker in prep_results['success_markers'].items()
                      if task in {'left', 'right', 'left-right', 'nothing'}}

    epochs_on_task = mne.Epochs(filt_raw_eeg, all_events, event_id=task_event_ids, baseline=None, verbose=verbose,
                                tmin=-exp_cfg['event-duration']['baseline'], tmax=exp_cfg['event-duration']['task'][0])
    epochs_on_task.apply_baseline(baseline, verbose=verbose)

    if filter_percentile is not None:
        epochs_on_task = filter_bad_epochs(epochs_on_task, filter_percentile, copy=False, verbose=verbose)
    
    # tfr: cwt or multitaper; multitaper is better if the signal is not that time-locked
    tfr_settings = dict(freqs=freqs, n_cycles=n_cycles, return_itc=False, average=False, n_jobs=n_jobs, verbose=verbose)
    if tfr_mode == 'cwt':
        tfr_epochs_on_task = mne.time_frequency.tfr_morlet(epochs_on_task, output='power', **tfr_settings)
    elif tfr_mode == 'multitaper':
        tfr_epochs_on_task = mne.time_frequency.tfr_multitaper(epochs_on_task, **tfr_settings)
    else:
        raise ValueError('invalid tfr_mode:', tfr_mode)

    # apply baseline on tfr
    tfr_epochs_on_task = tfr_epochs_on_task.apply_baseline(baseline, tfr_baseline_mode, verbose=verbose)

    return dict(epochs_on_task=epochs_on_task, tfr_epochs_on_task=tfr_epochs_on_task,
                on_task_times=tfr_epochs_on_task.times, task_event_ids=task_event_ids, task_baseline=baseline,
                on_task_events=epochs_on_task.events, on_task_drop_log=epochs_on_task.drop_log,
                on_task_session_idx=np.repeat(sess_id, len(epochs_on_task)))


def epoch_on_pull(sess_id, prep_results: dict, exp_cfg_path, freqs, baseline=(None, 0), tfr_mode='cwt',
                  tfr_baseline_mode='percent', n_cycles=None, n_jobs=4, verbose=False, filter_percentile=None,
                  reaction_tmax=1.):
    filt_raw_eeg = prep_results['filt_raw_eeg']
    all_events = prep_results['events']

    with open(exp_cfg_path, 'rt') as f:
        exp_cfg = json.load(f)

    # epoching
    pull_event_ids = {task: marker for task, marker in prep_results['success_markers'].items()
                      if task in {'left-pull', 'right-pull', 'left-right-pull'}}

    try:
        epochs_on_pull = mne.Epochs(filt_raw_eeg, all_events, event_id=pull_event_ids, baseline=None, verbose=verbose,
                                    tmin=-exp_cfg['event-duration']['baseline'],
                                    tmax=exp_cfg['event-duration']['task'][0] - reaction_tmax)
        epochs_on_pull.apply_baseline(baseline, verbose=verbose)

        if filter_percentile is not None:
            epochs_on_pull = filter_bad_epochs(epochs_on_pull, filter_percentile, copy=False, verbose=verbose)

        # tfr: cwt or multitaper; multitaper is better if the signal is not that time-locked
        tfr_settings = dict(freqs=freqs, n_cycles=n_cycles, return_itc=False, average=False, n_jobs=n_jobs, verbose=verbose)
        if tfr_mode == 'cwt':
            tfr_epochs_on_pull = mne.time_frequency.tfr_morlet(epochs_on_pull, output='power', **tfr_settings)
        elif tfr_mode == 'multitaper':
            tfr_epochs_on_pull = mne.time_frequency.tfr_multitaper(epochs_on_pull, **tfr_settings)
        else:
            raise ValueError('invalid tfr_mode:', tfr_mode)

        # apply baseline on tfr
        tfr_epochs_on_pull = tfr_epochs_on_pull.apply_baseline(baseline, tfr_baseline_mode, verbose=verbose)

        return dict(epochs_on_pull=epochs_on_pull, tfr_epochs_on_pull=tfr_epochs_on_pull,
                    on_pull_times=tfr_epochs_on_pull.times, pull_event_ids=pull_event_ids, pull_baseline=baseline,
                    on_pull_events=epochs_on_pull.events, on_pull_drop_log=epochs_on_pull.drop_log,
                    on_pull_session_idx=np.repeat(sess_id, len(epochs_on_pull)))

    except ValueError:  # No matching events found..
        print('NO ON-PULL EPOCHS REGISTERED', file=sys.stderr)
        return dict(epochs_on_pull=None, tfr_epochs_on_pull=None,
                    on_pull_times=None, pull_event_ids=pull_event_ids, pull_baseline=baseline,
                    on_pull_events=None, on_pull_drop_log=None,
                    on_pull_session_idx=np.repeat(sess_id, 0))


def epoch_on_break(sess_id, prep_results: dict, exp_cfg_path, freqs, baseline=(None, 0), tfr_mode='cwt',
                   tfr_baseline_mode='percent', n_cycles=None, n_jobs=4, verbose=False, filter_percentile=None):
    filt_raw_eeg = prep_results['filt_raw_eeg']
    all_events = prep_results['events']

    with open(exp_cfg_path, 'rt') as f:
        exp_cfg = json.load(f)

    break_event_ids = dict(brk=prep_results['success_markers']['break'])
    epochs_on_break = mne.Epochs(filt_raw_eeg, all_events, event_id=break_event_ids, baseline=None, verbose=verbose,
                                 tmin=-exp_cfg['event-duration']['task'][0] / 2,
                                 tmax=exp_cfg['event-duration']['break'][0])

    baseline = (max(baseline[0], epochs_on_break.tmin), baseline[1])
    epochs_on_break.apply_baseline(baseline, verbose=verbose)

    if filter_percentile is not None:
        epochs_on_break = filter_bad_epochs(epochs_on_break, filter_percentile, copy=False, verbose=verbose)

    tfr_settings = dict(freqs=freqs, n_cycles=n_cycles, return_itc=False, average=False, n_jobs=n_jobs, verbose=verbose)

    if tfr_mode == 'cwt':
        tfr_epochs_on_break = mne.time_frequency.tfr_morlet(epochs_on_break, output='power', **tfr_settings)
    elif tfr_mode == 'multitaper':
        tfr_epochs_on_break = mne.time_frequency.tfr_multitaper(epochs_on_break, **tfr_settings)
    else:
        raise ValueError('invalid tfr_mode:', tfr_mode)

    tfr_epochs_on_break = tfr_epochs_on_break.apply_baseline(baseline, tfr_baseline_mode, verbose=verbose)

    return dict(epochs_on_break=epochs_on_break, tfr_epochs_on_break=tfr_epochs_on_break,
                on_break_times=tfr_epochs_on_break.times, break_event_ids=break_event_ids, break_baseline=baseline,
                on_break_events=epochs_on_break.events, on_break_drop_log=epochs_on_break.drop_log,
                on_break_session_idx=np.repeat(sess_id, len(epochs_on_break)))


def plot_some(epoch_prep_results: dict, fig_output_path, rec_name, freqs, baseline=(None, 0),
              evoked_picks=('Fz', 'C3', 'Cz', 'C4'), show_nepochs=16):

    filt_raw_eeg = epoch_prep_results['filt_raw_eeg']
    all_events = epoch_prep_results['events']
    event_dict = epoch_prep_results['event_dict']
    epochs_on_task = epoch_prep_results['epochs_on_task']
    task_event_ids = epoch_prep_results['task_event_ids']

    filt_raw_eeg.plot(scalings=80, events=all_events, title='all events', show=False).savefig(
        f'{fig_output_path}/raw.png')
    epochs_on_task.plot(scalings=80, events=all_events, n_epochs=show_nepochs, title='epochs locked on task',
                        show=False) \
        .savefig(f'{fig_output_path}/epochs-on-task.png')

    # ERPs
    epochs_on_task.average().plot(scalings=50, window_title='ERP locked on task', show=False, picks=evoked_picks) \
        .savefig(f'{fig_output_path}/epochs-on-task-avg.png')

    # ERPs per task
    epochs_on_task['left'].average().plot(scalings=50, window_title='left_trigger',
                                                      show=False, picks=evoked_picks) \
        .savefig(f'{fig_output_path}/epochs-on-task-left-avg.png')
    epochs_on_task['right'].average().plot(scalings=50, window_title='right_trigger',
                                                       show=False, picks=evoked_picks) \
        .savefig(f'{fig_output_path}/epochs-on-task-right-avg.png')
    epochs_on_task['left-right'].average().plot(scalings=50, window_title='left_right_trigger',
                                                            show=False, picks=evoked_picks) \
        .savefig(f'{fig_output_path}/epochs-on-task-lr-avg.png')
    epochs_on_task['nothing'].average().plot(scalings=50, window_title='no_trigger',
                                                         show=False, picks=evoked_picks) \
        .savefig(f'{fig_output_path}/epochs-on-task-nothing-avg.png')

    gen_erds_plots(epochs_on_task, rec_name, task_event_ids, out_folder=f'{fig_output_path}/tfr_raw', freqs=freqs,
                   comp_time_freq=True, comp_tf_clusters=False, channels=('C3', 'Cz', 'C4'), baseline=baseline,
                   apply_baseline=True)

    gen_erds_plots(epochs_on_task, rec_name, task_event_ids, out_folder=f'{fig_output_path}/tfr_clust', freqs=freqs,
                   comp_time_freq=True, comp_tf_clusters=True, channels=('C3', 'Cz', 'C4'), baseline=baseline,
                   apply_baseline=True)


def load_epochs_for_subject(path, epoch_type='epochs_on_task', verbose=False):
    epoch_paths = sorted(list(glob(f'{path}/*{epoch_type}*-epo.fif')))
    epochs = [mne.read_epochs(path, verbose=verbose) for path in epoch_paths]
    epochs = mne.concatenate_epochs(epochs, verbose=verbose)
    return epochs


def process_eyes_open_closed(raw: mne.io.Raw, events: np.ndarray, info: mne.Info, do_plot=False, verbose=False,
                             exp_cfg_path='config/lr_finger/exp_me_l_r_lr_stim-w-dots.json', n_jobs=8, output_path='out'):
    # experiment config
    with open(exp_cfg_path, 'rt') as f:
        exp_cfg = json.load(f)
    
    event_ids = {'open': exp_cfg['events']['eyes-open-beg']['marker'],
                 'closed': exp_cfg['events']['eyes-closed-beg']['marker']}
    
    eyes_open_dur = exp_cfg['event-duration']['eyes-open']
    eyes_closed_dur = exp_cfg['event-duration']['eyes-closed']
    
    # eye_epochs = mne.Epochs(raw, events, event_id=event_ids, verbose=verbose,
                            # tmin=0, tmax=exp_cfg['event-duration']['eyes-open'])
    
    # crop parts out of recording
    eyes_open_beg_i = events[events[:, 2] == event_ids['open'], 0][0]
    eyes_open_beg_t = eyes_open_beg_i / info['sfreq']
    
    eyes_closed_beg_i = events[events[:, 2] == event_ids['closed'], 0][0]
    eyes_closed_beg_t = eyes_closed_beg_i / info['sfreq']
    
    eyes_open = raw.copy().crop(eyes_open_beg_t, eyes_open_beg_t + eyes_open_dur, verbose=verbose)
    eyes_closed = raw.copy().crop(eyes_closed_beg_t, eyes_closed_beg_t + eyes_closed_dur, verbose=verbose)
    
    # compute psd on posterior electrodes for IAF
    eyes_open_psd = eyes_open.compute_psd('welch', picks=['PO7', 'Oz', 'PO8'], n_jobs=n_jobs, verbose=verbose, fmax=80)
    eyes_clossed_psd = eyes_closed.compute_psd('welch', picks=['PO7', 'Oz', 'PO8'], n_jobs=n_jobs, verbose=verbose, fmax=80)
    difference = eyes_clossed_psd.get_data().mean(axis=0) - eyes_open_psd.get_data().mean(axis=0)
    
    peak, _ = find_peaks(difference, distance=200)
    
    if do_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
        eyes_clossed_psd.plot(axes=ax1, average=True, show=False)
        eyes_open_psd.plot(axes=ax2, average=True, show=False)
        ax3.plot(eyes_clossed_psd.freqs, difference)

        ax1.axvline(x=eyes_clossed_psd.freqs[peak], ymax=100, color='red')
        ax2.axvline(x=eyes_clossed_psd.freqs[peak], ymax=100, color='red')#, eyes_clossed_psd.get_data().mean(axis=0)[peak])

        ax1.set_title('eyes closed')
        ax2.set_title('eyes open')
        ax3.set_title(f'difference: {difference.max():.2f} at {eyes_clossed_psd.freqs[peak[0]]:.2f} Hz (IAF)')

        fig.savefig(f'{output_path}/open-closed_eye.png')
    
    return eyes_open, eyes_closed, eyes_clossed_psd.freqs[peak]


def cross_correlation_shift3(data):  # chatgpt
    """
    Computes the cross-correlation between the reference vector (the one with the highest average cross-correlation
    with all other vectors) and all other vectors in a data matrix, finds the shift for each vector that maximizes
    the cross-correlation, shifts all vectors to their respective maximum correlation position, pads all vectors
    to have the same final length with NaN values, and returns the resulting padded and shifted matrix.

    Args:
        data (numpy.ndarray): A 2D numpy array of size (N, D) containing N vectors of dimension D.

    Returns:
        numpy.ndarray: A 2D numpy array of size (N, max_len) containing the shifted and padded vectors,
        where max_len is the length of the longest vector after shifting and padding.
    """
    # Find the reference vector with highest average cross-correlation with all other vectors
    corr = np.corrcoef(data)
    avg_corr = np.mean(corr, axis=0)
    ref_index = np.argmax(avg_corr)

    # Compute cross-correlation between reference vector and all other vectors
    corr = []
    for i in range(data.shape[0]):
        corr.append(np.correlate(data[ref_index], data[i], mode='full'))
    corr = np.array(corr)

    # Find the shifts for each vector
    shifts = np.argmax(corr, axis=1) - (data.shape[1] - 1)

    # Shift all vectors to their respective maximum correlation position
    padded_data = np.full((data.shape[0], data.shape[1] + np.abs(shifts).max()), np.nan)
    for i in range(data.shape[0]):
        shift = shifts[i]
        if shift >= 0:
            padded_data[i, shift:data.shape[1] + shift] = data[i, :]
        else:
            padded_data[i, :data.shape[1] + shift] = data[i, -shift:]

    return padded_data


def pad_2d_to_max_length(arr_list):  # chatgpt
    """
    Pads only the second dimensions of the 2D numpy arrays in a list to the same length, by appending NaN values to the end,
    up to the maximum length of all the second dimensions in the list.

    Args:
        arr_list (list): A list of 2D numpy arrays of different second dimension lengths.

    Returns:
        list: A list of 2D numpy arrays padded with NaN values to the same second dimension length as the longest
        second dimension in arr_list.
    """
    # Find the maximum length of all second dimensions
    max_len = np.max([arr.shape[1] for arr in arr_list])

    # Pad all second dimensions to the same length
    padded_arr_list = []
    for arr in arr_list:
        padded_arr = np.pad(arr, pad_width=((0, 0), (0, max_len - arr.shape[1])), mode='constant',
                            constant_values=np.nan)
        padded_arr_list.append(padded_arr)

    return padded_arr_list


def plot_erds(epochs, fois, event_ids, channels, freqs, times, shift=False, show_std=True):

    # TODO any smoothing?

    # TODO !!! LIMIT CROSS-CORRELATION SHIFT AMOUNT !!!

    fig, axes = plt.subplots(len(fois), len(channels), figsize=(8 * len(channels), len(fois) * 7))
    colors = cm.rainbow(np.linspace(0, 1, len(event_ids)))

    ylim = [-3, 5] if shift else [-1, 1.5]

    for foi_i, (foi_name, foi) in enumerate(fois.items()):  # freq rng of interest
        foi_freqs = (foi[0] <= freqs) & (freqs <= foi[1])

        for ev_i, event in enumerate(event_ids.keys()):
            tfr_f_ev = epochs[event].data[:, :, foi_freqs, :].mean(axis=2)
            if shift:
                tfr_f_ev = np.stack(pad_2d_to_max_length([cross_correlation_shift3(tfr_f_ev[:, ch_i])
                                    for ch_i in range(len(channels))]), axis=1)
                times = np.arange(tfr_f_ev.shape[-1])

            tfr_f_ev_mean = np.nanmean(tfr_f_ev, axis=0)
            tfr_f_ev_std = np.nanstd(tfr_f_ev, axis=0)

            for ch_i, ch in enumerate(channels):
                if show_std:
                    axes[foi_i, ch_i].fill_between(times, tfr_f_ev_mean[ch_i] - tfr_f_ev_std[ch_i],
                                                   tfr_f_ev_mean[ch_i] + tfr_f_ev_std[ch_i],
                                                   alpha=.3, color=colors[ev_i])
                label = event if ch_i == 0 else '_nolegend_'
                axes[foi_i, ch_i].plot(times, tfr_f_ev_mean[ch_i], label=label, alpha=.85, color=colors[ev_i])
                axes[foi_i, ch_i].set_ylim(ylim)
                if ev_i == 0:
                    axes[foi_i, ch_i].axvline([0], *ylim, color='black')
                    low_cf, high_cf = np.percentile(tfr_f_ev, [10, 90])
                    axes[foi_i, ch_i].axhline([low_cf], times[0], times[-1], color='gray')
                    axes[foi_i, ch_i].axhline([high_cf], times[0], times[-1], color='gray')
                if foi_i == 0:
                    axes[foi_i, ch_i].set_title(ch)
                if ch_i == 0:
                    axes[foi_i, ch_i].set_ylabel(foi_name)
                if foi_i == 0 and ev_i == len(event_ids.keys()) - 1 and ch_i == 0:
                    fig.legend(loc='lower center', ncols=len(event_ids))
                if shift:
                    axes[foi_i, ch_i].set_xticklabels([])
                    axes[foi_i, ch_i].set_xlabel('$shifted$')

    return fig


def combined_session_analysis(subject, streams_path, meta_path, output_path, channels=('C3', 'Cz', 'C4'),
                              norm_c34_w_cz=False, verbose=False, part='task', run_deprecated=False):
    # load hdf5 + meta data and run gen_erds_plots
    streams_data = h5py.File(streams_path, 'r')

    with open(meta_path, 'rb') as f:
        meta_data = pickle.load(f)

    eeg_info = meta_data['eeg_info']
    freqs = streams_data.attrs['freqs']
    times = streams_data.attrs[f'on_{part}_times'][:]
    on_task_events = streams_data[f'on_{part}_events'][:]
    iaf = np.median(np.asarray(meta_data['iafs']))
    event_ids = meta_data[f'{part}_event_ids']

    # C3-Cz, C4-Cz
    tfr_orig = streams_data[f'tfr_epochs_on_{part}'][:]
    if norm_c34_w_cz:
        cz_i = eeg_info['ch_names'].index('Cz')
        c3_i = eeg_info['ch_names'].index('C3')
        c4_i = eeg_info['ch_names'].index('C4')
        tfr_orig[:, c3_i] = 2 * tfr_orig[:, c3_i] - tfr_orig[:, cz_i]
        tfr_orig[:, c4_i] = 2 * tfr_orig[:, c4_i] - tfr_orig[:, cz_i]

    # create epochs
    epochs = mne.time_frequency.EpochsTFR(eeg_info, tfr_orig, times, freqs, verbose=verbose,
                                          events=on_task_events, event_id=meta_data[f'{part}_event_ids'])
    epochs = epochs.pick(channels)

    # time-freq spectrograms
    out_folder = f'{output_path}/figures/erds_combined_{part}_c34-cz-{norm_c34_w_cz}'
    vmin, vmax = -1, 1.5
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    for event in event_ids.keys():
        tfr_ev = epochs[event]
        fig, axes = plt.subplots(1, len(channels) + 1, figsize=(14, 4),
                                 gridspec_kw={"width_ratios": [10] * len(channels) + [1]})

        for ch, ax in enumerate(axes[:-1]):  # for each channel
            tfr_ev.average().plot([ch], cmap="RdBu_r", cnorm=cnorm, axes=ax, colorbar=False,
                                  show=False, mask=None, mask_style=None)
            ax.set_title(epochs.ch_names[ch], fontsize=10)
            ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event

        cbar_src = axes[0].images[-1] if len(axes[0].images) > 0 else axes[0].collections[0]
        fig.colorbar(cbar_src, cax=axes[-1]).ax.set_yscale("linear")
        fig.suptitle(f"ERDS ({event})")
        os.makedirs(out_folder, exist_ok=True)
        fig.savefig(f'{out_folder}/{subject}_erds_{event}.png')

    # ERDS plots
    fois = {'theta': (4, 7), 'wide_mu': (iaf - 2, iaf + 2), 'tight_mu': (iaf - 1, iaf + 1),
            'wide_beta': (13, 30), 'tight_beta': (18, 30), 'tighter_beta': (16, 24),
            'wide_gamma': (30, 48), 'low_gamma': (25, 40), 'high_gamma': (40, 48)}

    fig = plot_erds(epochs, fois, event_ids, channels, freqs, times, shift=False)
    fig_shifted = plot_erds(epochs, fois, event_ids, channels, freqs, times, shift=True)

    fig.savefig(f'{out_folder}/{subject}_erds_fois.png', bbox_inches='tight', pad_inches=0, dpi=350)
    fig_shifted.savefig(f'{out_folder}/{subject}_erds_fois_shifted.png', bbox_inches='tight', pad_inches=0, dpi=350)
    plt.close('all')

    # DEPRECATED
    if run_deprecated:
        # old algo wastes resources like no tomorrow
        tfr_orig = tfr_orig[..., ::2]
        times = times[::2]
        epochs = mne.time_frequency.EpochsTFR(eeg_info, tfr_orig, times, freqs, verbose=verbose,
                                              events=on_task_events, event_id=meta_data[f'{part}_event_ids'])
        epochs = epochs.pick(channels)

        gen_erds_plots(epochs, subject, meta_data[f'{part}_event_ids'],
                       out_folder=f'{output_path}/figures/combined_{part}_c34-cz-{norm_c34_w_cz}',
                       freqs=freqs, comp_time_freq=True, comp_tf_clusters=False, channels=channels,
                       baseline=meta_data[f'{part}_baseline'], apply_baseline=False, copy=False)

        # tf clustering can only be done when the 'percent' baselining option is used
        # percent scales the tfr data into having negative and positive values, which is then used in the plotting
        #   function to determine significant positive (synchronization) or negative (desync) values
        if meta_data['tfr_baseline_mode'] == 'percent':
            gen_erds_plots(epochs, subject, meta_data[f'{part}_event_ids'],
                           out_folder=f'{output_path}/figures/combined_clust_{part}_c34-cz-{norm_c34_w_cz}', freqs=freqs,
                           comp_time_freq=True, comp_tf_clusters=True, channels=channels,
                           baseline=meta_data[f'{part}_baseline'], apply_baseline=False, copy=False)

    streams_data.close()


def main(
    subject='0717b399',  # 0717b399 | 6808dfab
    session_ids=None,  # range(1, 12) | range(1, 4)
    freq_rng=(2, 40),  # min and max frequency to sample (see how below)
    nfreq=100,  # number of frequency to sample in freq_rng
    n_cycles=None,
    tfr_mode='multitaper',  # cwt | multitaper: multitaper is smoother on time
    do_plot=False,  # slows processing down a lot
    n_jobs=6,
    verbose=False,
    rerun_proc=False,  # whether to run preprocessiong regardless the h5 file is already created or not
    bandpass_freq=(.5, 50),
    notch_freq=(50, 100),
    baseline=(-1, -.05),  # normalize signal according to the baseline interval
    eeg_sfreq=250, gamepad_sfreq=125,  # sampling frequencies
    reaction_tmax=0.6,  # max amount of time between cue and gamepad trigger pull allowed in sec
    store_per_session_recs=False,  # store extra preprocessed fif and h5 files, native to mne
    # see: https://mne.tools/stable/generated/mne.time_frequency.EpochsTFR.html#mne.time_frequency.EpochsTFR.apply_baseline
    tfr_baseline_mode='percent',
    filter_percentile=95,  # filter out too large within-epoch, peak-to-peak signal differences (noisy epochs)
    combined_anal_channels=('C3', 'Cz', 'C4'),  # only affects the combined plots, defines the channels to show
    norm_c34_w_cz=False,  # whether to remove the time frequency signal of Cz from C3 and C4, only when plotting
    is_imaginary=None,
):

    assert bandpass_freq[0] <= freq_rng[0] and freq_rng[1] <= bandpass_freq[1]
    freqs = np.logspace(np.log(freq_rng[0]), np.log(freq_rng[1]), num=nfreq, base=np.e)

    recordings_path = '../recordings'
    exp_cfg_path = 'config/lr_finger/exp_me_l_r_lr_stim-w-dots.json'
    # output_path = f'out/{subject}'
    output_path = f'out_bl{baseline[0]}-{baseline[1]}_tfr-{tfr_mode}-{tfr_baseline_mode}_reac-{reaction_tmax}' \
                  f'_bad-{filter_percentile}_f-{freq_rng[0]}-{freq_rng[1]}-{nfreq}/{subject}'
    meta_path = f'{output_path}/{subject}_meta.pckl'
    streams_path = f'{output_path}/{subject}_streams.h5'
    os.makedirs(output_path, exist_ok=True)

    # conservatively high freq (and low time) resolution by default
    n_cycles = (np.log(freqs) * 2 + 2).astype(np.int32) if n_cycles is None else n_cycles

    if not os.path.isfile(streams_path) or rerun_proc:

        # prepare h5 dataset: combined raw, epochs, gamepad streams
        streams_data = h5py.File(streams_path, 'w')
        streams_info = {'filt_raw_eeg': {'dtype': 'float32', 'epoch': False, 'get': lambda x: x.get_data()},

                        'epochs_on_task': {'dtype': 'float32', 'epoch': True, 'get': lambda x: x.get_data()},
                        'epochs_on_pull': {'dtype': 'float32', 'epoch': True, 'get': lambda x: x.get_data() if x else x},
                        'epochs_on_break': {'dtype': 'float32', 'epoch': True, 'get': lambda x: x.get_data()},

                        'tfr_epochs_on_task': {'dtype': 'float32', 'epoch': True, 'get': lambda x: x.data},
                        'tfr_epochs_on_pull': {'dtype': 'float32', 'epoch': True, 'get': lambda x: x.data if x else x},
                        'tfr_epochs_on_break': {'dtype': 'float32', 'epoch': True, 'get': lambda x: x.data},

                        'gamepad': {'dtype': 'float16', 'epoch': False, 'get': lambda x: x},
                        'events': {'dtype': 'int32', 'epoch': True, 'get': lambda x: x},
                        'on_task_events': {'dtype': 'int32', 'epoch': True, 'get': lambda x: x},
                        'on_pull_events': {'dtype': 'int32', 'epoch': True, 'get': lambda x: x},
                        'on_break_events': {'dtype': 'int32', 'epoch': True, 'get': lambda x: x},

                        'on_task_session_idx': {'dtype': 'int32', 'epoch': True, 'get': lambda x: x},
                        'on_pull_session_idx': {'dtype': 'int32', 'epoch': True, 'get': lambda x: x},
                        'on_break_session_idx': {'dtype': 'int32', 'epoch': True, 'get': lambda x: x}}  # i know, it got out of hand

        # process sessions one-by-one
        # num_epochs = []
        meta_names = ['eeg_info', 'event_dict', 'on_task_times', 'task_event_ids', 'break_event_ids', 'eeg_ch_names',
                      'pull_event_ids', 'on_pull_times', 'on_break_times', 'task_baseline', 'pull_baseline',
                      'break_baseline']
        meta_data = {name: None for name in meta_names}
        meta_data['iafs'] = []  # individual alpha freq for each session
        meta_data['baseline'] = baseline
        meta_data['tfr_baseline_mode'] = tfr_baseline_mode
        meta_data['tfr_mode'] = tfr_mode
        meta_data['freqs'] = freqs
        meta_data['filter_percentile'] = filter_percentile
        meta_data['freq_rng'] = freq_rng
        meta_data['nfreq'] = nfreq
        meta_data['n_cycles'] = n_cycles
        meta_data['bandpass_freq'] = bandpass_freq
        meta_data['reaction_tmax'] = reaction_tmax
        meta_data['session_ids'] = session_ids

        for i, sid in enumerate(session_ids):

            rec_name = f'{subject}-{sid:03d}'
            fig_output_path = f'{output_path}/figures/{sid:03d}'

            # preprocess session
            try:
                sess = preprocess_session(recordings_path, rec_name, subject, sid, exp_cfg_path, eeg_sfreq, gamepad_sfreq,
                                          bandpass_freq, notch_freq, freqs, do_plot, reaction_tmax, n_jobs, verbose,
                                          fig_output_path, is_imaginary[i] if is_imaginary is not None else False)
            except WTFException:
                print(f'wtf on {subject}/{sid:03d}; skipping..')
                continue

            # create IAF plots
            _, _, iaf = process_eyes_open_closed(sess['filt_raw_eeg'], sess['events'], sess['eeg_info'],
                                                 do_plot=do_plot, output_path=fig_output_path, verbose=verbose)
            meta_data['iafs'].append(iaf)

            # epoch sessions
            task_epochs_sess = epoch_on_task(sid, sess, exp_cfg_path, freqs, baseline, tfr_mode, tfr_baseline_mode,
                                             n_cycles, verbose=verbose, filter_percentile=filter_percentile)
            pull_epochs_sess = epoch_on_pull(sid, sess, exp_cfg_path, freqs, baseline, tfr_mode,
                                             tfr_baseline_mode, n_cycles, verbose=verbose,
                                             filter_percentile=filter_percentile, reaction_tmax=reaction_tmax)
            break_epochs_sess = epoch_on_break(sid, sess, exp_cfg_path, freqs, baseline, tfr_mode, tfr_baseline_mode,
                                               n_cycles, verbose=verbose, filter_percentile=filter_percentile)
            sess = dict(**sess, **task_epochs_sess, **break_epochs_sess, **pull_epochs_sess)

            # plots per session
            if do_plot:
                plot_some(sess, fig_output_path, rec_name, freqs, baseline)

            # extract session data and store num epochs for current session
            sess_data = {name: info['get'](sess[name]) for name, info in streams_info.items()}
            # num_epochs.append(sess_data['epochs_on_task'].shape[0])

            # determine size of h5 datasets after first session processing
            # all epochs subsequent of sessions should have the same size (apart from the number of trials)
            if i == 0:
                for name, data in sess_data.items():
                    info = streams_info[name]
                    # only for epoched data, there is a separate dataset created for each non-epoch stream
                    if info['epoch']:
                        streams_data.create_dataset(name, (0, *data.shape[1:]), dtype=info['dtype'],
                                                    maxshape=(None, *data.shape[1:]))

                # assign meta data
                for meta_name in meta_names:
                    meta_data[meta_name] = sess[meta_name]

            # append streams to h5
            for name, data in sess_data.items():
                info = streams_info[name]
                ds_name = f'{name}_{sid}' if not info['epoch'] else name  # separate name/session for non epoched streams

                if info['epoch']:
                    if data is not None and data.size > 0:  # imaginary dataset / epochs_on_pull
                        streams_data[ds_name].resize(streams_data[ds_name].shape[0] + data.shape[0], axis=0)
                        streams_data[ds_name][-data.shape[0]:] = data
                else:  # not epoched stream, create new dataset
                    streams_data.create_dataset(ds_name, dtype=info['dtype'], data=data)

            # store individual session files
            raw_path = f'{output_path}/{subject}_{sid:03d}_raw-eeg.fif'

            if store_per_session_recs:
                sess['filt_raw_eeg'].save(raw_path, overwrite=True, verbose=verbose)

                for part in ['epochs_on_task', 'epochs_on_pull', 'epochs_on_break']:
                    epochs_path = f'{output_path}/{subject}_{sid:03d}_{part}-epo.fif'
                    tfr_epochs_path = f'{output_path}/{subject}_{sid:03d}_{part}-tfr.h5'
                    sess[part].save(epochs_path, overwrite=True, verbose=verbose)
                    sess[f'tfr_{part}'].save(tfr_epochs_path, overwrite=True, verbose=verbose)

        streams_data.attrs['on_task_times'] = meta_data['on_task_times']
        streams_data.attrs['on_pull_times'] = meta_data['on_pull_times']
        streams_data.attrs['on_break_times'] = meta_data['on_break_times']
        streams_data.attrs['freqs'] = meta_data['freqs']  # same for all epoch types

        streams_data.close()
        print('h5 epoch files saved')

        # save metadata
        with open(meta_path, 'wb') as f:
            pickle.dump(meta_data, f)

    # analysis, plots generated from all the sessions of one subject
    combined_session_analysis(subject, streams_path, meta_path, output_path, combined_anal_channels,
                              norm_c34_w_cz, verbose, 'task')
    combined_session_analysis(subject, streams_path, meta_path, output_path, combined_anal_channels,
                              norm_c34_w_cz, verbose, 'pull')
    combined_session_analysis(subject, streams_path, meta_path, output_path, combined_anal_channels,
                              norm_c34_w_cz, verbose, 'break')


if __name__ == '__main__':

    data_path = '../recordings'
    subjects = [os.path.basename(s)[4:] for s in glob(f'{data_path}/sub-*')]  # like ['0717b399', 'a9223e93']
    sessions = [sorted([int(os.path.basename(sess)[5:]) for sess in glob(f'{data_path}/sub-{subj}/ses-*')])
                for subj in subjects]

    for subject, sess_ids in zip(subjects, sessions):
        is_imaginary = np.zeros(len(sess_ids), dtype=bool)
        if subject == '0717b399':  # has motor imaginary
            is_imaginary[[-1, -2]] = True

        main(subject=subject, session_ids=sess_ids, rerun_proc=True, norm_c34_w_cz=True, do_plot=False,
             reaction_tmax=.6, is_imaginary=is_imaginary)
        # main(subject=subject, session_ids=rng, rerun_proc=False, norm_c34_w_cz=False)
