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


def define_target_events2(events, reference_id, target_ids, sfreq, tmin, tmax, new_id=None, fill_na=None, occur=True):
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
        The target events that should not happen in the vicinity of reference_id.
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

    Returns
    -------
    new_events : ndarray
        The new defined events.
    lag : ndarray
        Time lag between reference and target in milliseconds.
    """
    if new_id is None:
        new_id = reference_id

    tsample = 1e3 / sfreq
    imin = int(tmin * sfreq)
    imax = int(tmax * sfreq)

    new_events = []
    lag = []
    for event in events.copy().astype(int):
        if event[2] == reference_id:
            lower = event[0] + imin
            upper = event[0] + imax
            
            tcrit = (events[:, 0] > lower) & (events[:, 0] < upper)
            if occur:
                res = np.logical_and.reduce([np.any(tcrit & (events[:, 2] == tid)) for tid in target_ids])
            else:  # non-co-occurance
                res = ~np.logical_or.reduce([tcrit & (events[:, 2] == tid) for tid in target_ids])
            res = events[res]
            
            # res = events[(events[:, 0] > lower) &
                        #  (events[:, 0] < upper) & (events[:, 2] == target_id)]
            if res.any():
                lag += [event[0] - res[0][0]]
                event[2] = new_id
                new_events += [event]
            elif fill_na is not None:
                event[2] = fill_na
                new_events += [event]
                lag.append(np.nan)

    new_events = np.array(new_events)

    with np.errstate(invalid='ignore'):  # casting nans
        lag = np.abs(lag, dtype='f8')
    if lag.any():
        lag *= tsample
    else:
        lag = np.array([])

    return new_events if new_events.any() else np.array([]), lag

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


def preprocess_session(rec_base_path, rec_name, subject, session, exp_cfg_path,
                       eeg_sfreq=250, gamepad_sfreq=125, bandpass_freq=(0.5, 80), notch_freq=(50, 100),
                       freqs=np.arange(2, 50, 0.2), do_plot=False, reaction_tmax=1.,
                       n_jobs=4, verbose=False, fig_output_path='out'):
    
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

    # check if the same number of on-off happened, pair on-offs, remove ones that can't be paired
    print('trigger on/off times should be equal:')
    print('  L2:', triggers_on_i[0].shape[0], '==', triggers_off_i[0].shape[0])
    print('  R2:', triggers_on_i[1].shape[0], '==', triggers_off_i[1].shape[0])

    # # plotting commands for debugging
    # ti=4;plt.plot(gamepad_i, gamepad[ti, :]);plt.scatter(triggers_on_i[ti-4], gamepad[ti, triggers_on_i[ti-4]], marker='X', color='red');plt.show()
    # ti=4;plt.plot(gamepad_i, np.diff(gamepad[ti, :], append=0));plt.scatter(triggers_on_i[ti-4], np.diff(gamepad[ti, :], append=0)[triggers_on_i[ti-4]], marker='X', color='red');plt.show()
    # ti=4;plt.plot(gamepad_i, gamepad[ti, :]);plt.plot(gamepad_i, np.diff(gamepad[ti, :], append=0));plt.scatter(triggers_on_i[ti-4], np.diff(gamepad[ti, :], append=0)[triggers_on_i[ti-4]], marker='X', color='red');plt.show()

    # events data structure: n_events x 3: [[sample_i, 0, marker]...]
    trigger_on_markers = [np.repeat(ev_mark, len(i)) for i, ev_mark in zip(triggers_on_i, trigger_on_event_markers)]
    trigger_off_markers = [np.repeat(ev_mark, len(i)) for i, ev_mark in zip(triggers_off_i, trigger_off_event_markers)]

    trigger_i = np.concatenate(triggers_on_i + triggers_off_i)
    trigger_markers = np.concatenate(trigger_on_markers + trigger_off_markers)
    gamepad_lr_events = np.stack([trigger_i, np.zeros_like(trigger_i), trigger_markers], axis=1)

    # create events from exp-marker stream and combine it with gamepad events
    exp_events = np.stack([events_i, np.zeros_like(events_i), events], axis=1)
    all_events = np.concatenate([gamepad_lr_events, exp_events], axis=0)

    # event dictionary - event_name: marker
    event_dict = {**{ename: einfo['marker'] for ename, einfo in exp_cfg['tasks'].items()},
                  **{ename: einfo['marker'] for ename, einfo in exp_cfg['events'].items() if 'marker' in einfo},
                  **{f'{trig}-on': mark for mark, trig in zip(trigger_on_event_markers, ['L2', 'R2'])},
                  **{f'{trig}-off': mark for mark, trig in zip(trigger_off_event_markers, ['L2', 'R2'])}}
    
    # preprocess numpy eeg, before encaptulating into mne raw
    # common median referencing - substract median at each timepoint
    eeg -= np.mean(eeg, axis=0)  # TODO was median
    
    # create raw mne eeg array and add events
    # adding gamepad to eeg channels just complictes things at this point,
    #   as filtering/preprocessing pipeline is separate for the two
    # eeg_gamepad_info = mne.create_info(ch_types=['eeg'] * len(eeg_ch_names) + ['misc'] * len(trigger_lr_i),
    #                                    ch_names=eeg_ch_names + ['L2', 'R2'], sfreq=eeg_sfreq)
    # raw_w_gamepad = mne.io.RawArray(np.concatenate([eeg, gamepad[trigger_lr_i, :]]), eeg_gamepad_info)
    eeg_info = mne.create_info(ch_types='eeg', ch_names=eeg_ch_names, sfreq=eeg_sfreq)
    raw = mne.io.RawArray(eeg, eeg_info, verbose=verbose)
    easycap_montage = mne.channels.make_standard_montage('easycap-M1')
    raw.set_montage(easycap_montage)
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

    # one_big_epoch = filt_raw_eeg._data[np.newaxis, ...]  # hacked the system
    # filt_tfr_eeg = mne.time_frequency.tfr_array_morlet(one_big_epoch, shared_sfreq, freqs, n_cycles, output='power', n_jobs=n_jobs)[0, ...]
    # TODO could just use mne.time_frequency.tfr.cwt
    #   + mne.time_frequency.tfr.morlet without epoch hacking
    # TODO check answers here: https://mne.discourse.group/t/get-epochstfr-object-from-whole-signal-tfr-and-events/6435
    # TODO another way to get epochstfr is to create a dummy numpy array with np.arange, apply epoching by event,
    #   then use the indices to manually epoch tfr and create an EpochsTFR
    # TODO and use fft_or_cwt
    
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
    
    left_success = mne.event.define_target_events(all_events, event_dict['left'], event_dict['L2-on'],
                                                  shared_sfreq, tmin=1e-4, tmax=reaction_tmax, new_id=left_success_marker)
    right_success = mne.event.define_target_events(all_events, event_dict['right'], event_dict['R2-on'],
                                                   shared_sfreq, tmin=1e-4, tmax=reaction_tmax, new_id=right_success_marker)
    lr_success = define_target_events2(all_events, event_dict["left-right"], [event_dict['L2-on'], event_dict['R2-on']],
                                       shared_sfreq, tmin=1e-4, tmax=reaction_tmax, new_id=lr_success_marker)
    # when nothing should be pressed, it just becomes a fucking headache (implemented it myself in the end) - mne is straight dogshit
    nothing_success = define_target_events2(all_events, event_dict["nothing"], [event_dict['L2-on'], event_dict['R2-on']], shared_sfreq,
                                            tmin=1e-4, tmax=exp_cfg['event-duration']['task'][0], new_id=nothing_success_marker, occur=False)
    
    # stats on delay
    if do_plot:
        fig = plt.figure()
        plt.title('button press delay')
        plt.hist(left_success[1], bins=20, label='left', alpha=.5)
        plt.hist(right_success[1], bins=20, label='right', alpha=.5)
        # plt.hist(lr_success[1], bins=10, label='left-right', alpha=.5)  # TODO implement define_target_events2() proper lag array handling
        plt.legend()
        fig.savefig(f'{fig_output_path}/btn_press_delay_{rec_name}.png')
    
    # do the same for the break event
    break_marker = exp_cfg['events']['break']['marker']
    break_success_marker = int(f'{break_marker}0')
    break_success = define_target_events2(all_events, break_marker, [event_dict['L2-off'], event_dict['R2-off']],
                                          shared_sfreq, tmin=1e-4, tmax=reaction_tmax, new_id=break_success_marker)
    
    all_events = np.concatenate([all_events, left_success[0], right_success[0], lr_success[0], nothing_success[0], break_success[0]], axis=0)
    
    # some stats
    num_old = [(all_events[:, 2] == event_dict[ev]).sum() for ev in ['left', 'right', 'left-right', 'nothing']]
    num_succ = [(all_events[:, 2] == ev).sum() for ev in [left_success_marker, right_success_marker, lr_success_marker,
                                                          nothing_success_marker]]
    print('num of events successful/originally:',
          {name: f'{nsucc}/{nold}' for name, nold, nsucc in zip(['left', 'right', 'left-right', 'nothing'], num_old, num_succ)})

    # order events
    all_events = all_events[np.argsort(all_events[:, 0]), :]

    return dict(success_markers={'left': left_success_marker, 'right': right_success_marker,
                                 'left-right': lr_success_marker, 'nothing': nothing_success_marker,
                                 'break': break_success_marker},
                filt_raw_eeg=filt_raw_eeg, gamepad=gamepad, events=all_events, event_dict=event_dict,
                eeg_info=eeg_info, freqs=freqs, eeg_ch_names=eeg_ch_names)


def epoch_on_task(prep_results: dict, exp_cfg_path, freqs, baseline=(None, 0), tfr_mode='cwt',  # cwt | multitaper
                  tfr_baseline_mode='percent', n_cycles=None, n_jobs=4, verbose=False, filter_percentile=None):
    filt_raw_eeg = prep_results['filt_raw_eeg']
    all_events = prep_results['events']

    with open(exp_cfg_path, 'rt') as f:
        exp_cfg = json.load(f)
    
    # epoching
    succ_markers = prep_results['success_markers']

    task_event_ids = {task: marker for task, marker in succ_markers.items()
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
    tfr_epochs_on_task.apply_baseline(baseline, tfr_baseline_mode, verbose=verbose)

    return dict(epochs_on_task=epochs_on_task, tfr_epochs_on_task=tfr_epochs_on_task,
                on_task_times=tfr_epochs_on_task.times, task_event_ids=task_event_ids,
                on_task_events=epochs_on_task.events, on_task_drop_log=epochs_on_task.drop_log)


def epoch_on_pull(prep_results: dict, exp_cfg_path, freqs, baseline=(None, 0), tfr_mode='cwt',
                  tfr_baseline_mode='percent', n_cycles=None, n_jobs=4, verbose=False, filter_percentile=None):
    filt_raw_eeg = prep_results['filt_raw_eeg']
    all_events = prep_results['events']
    event_dict = prep_results['event_dict']

    with open(exp_cfg_path, 'rt') as f:
        exp_cfg = json.load(f)

    # TODO !!! L2 and R2 have overlapping event times (pressed at the same time)
    #   need to separate only-L2 (when instructed for left), only-R2, and both successful pulls
    #   create new events for these in preprocess

    pull_event_ids = dict(L2=event_dict['L2-on'], R2=event_dict['R2-on'])
    epochs_on_pull = mne.Epochs(filt_raw_eeg, all_events, event_id=pull_event_ids, baseline=None, verbose=verbose,
                                tmin=-exp_cfg['event-duration']['baseline'],
                                tmax=exp_cfg['event-duration']['task'][0] - .5)
    epochs_on_pull.apply_baseline(baseline, verbose=verbose)

    if filter_percentile is not None:
        epochs_on_pull = filter_bad_epochs(epochs_on_pull, filter_percentile, copy=False, verbose=verbose)

    tfr_settings = dict(freqs=freqs, n_cycles=n_cycles, return_itc=False, average=False, n_jobs=n_jobs, verbose=verbose)

    if tfr_mode == 'cwt':
        tfr_epochs_on_pull = mne.time_frequency.tfr_morlet(epochs_on_pull, output='power', **tfr_settings)
    elif tfr_mode == 'multitaper':
        tfr_epochs_on_pull = mne.time_frequency.tfr_multitaper(epochs_on_pull, **tfr_settings)
    else:
        raise ValueError('invalid tfr_mode:', tfr_mode)

    tfr_epochs_on_pull.apply_baseline(baseline, tfr_baseline_mode, verbose=verbose)

    return dict(epochs_on_pull=epochs_on_pull, tfr_epochs_on_pull=tfr_epochs_on_pull,
                on_pull_times=tfr_epochs_on_pull.times, pull_event_ids=pull_event_ids,
                on_pull_events=epochs_on_pull.events, on_pull_drop_log=epochs_on_pull.drop_log)


def epoch_on_break(prep_results: dict, exp_cfg_path, freqs, baseline=(None, 0), tfr_mode='cwt',
                   tfr_baseline_mode='percent', n_cycles=None, n_jobs=4, verbose=False, filter_percentile=None):
    filt_raw_eeg = prep_results['filt_raw_eeg']
    all_events = prep_results['events']

    with open(exp_cfg_path, 'rt') as f:
        exp_cfg = json.load(f)

    break_event_id = dict(brk=prep_results['success_markers']['break'])
    epochs_on_break = mne.Epochs(filt_raw_eeg, all_events, event_id=break_event_id, baseline=None, verbose=verbose,
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

    tfr_epochs_on_break.apply_baseline(baseline, tfr_baseline_mode, verbose=verbose)

    return dict(epochs_on_break=epochs_on_break, tfr_epochs_on_break=tfr_epochs_on_break,
                on_break_times=tfr_epochs_on_break.times, break_event_id=break_event_id,
                on_break_events=epochs_on_break.events, on_break_drop_log=epochs_on_break.drop_log)


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

        fig.savefig(f'{output_path}/figures/open-closed_eye.png')
    
    return eyes_open, eyes_closed, eyes_clossed_psd.freqs[peak]


# TODO double-check preproc code
# TODO remove bad epochs - use quality control function in epoching functions


def combined_session_analysis(subject, streams_path, meta_path, output_path, channels=('C3', 'Cz', 'C4'),
                              norm_c34_w_cz=False, verbose=False):
    # load hdf5 + meta data and run gen_erds_plots
    streams_data = h5py.File(streams_path, 'r')

    with open(meta_path, 'rb') as f:
        meta_data = pickle.load(f)

    eeg_info = meta_data['eeg_info']
    freqs = streams_data.attrs['freqs']
    times = streams_data.attrs['on_task_times'][:]

    # epochs = load_epochs_for_subject(output_path, epoch_type='epochs_on_task')
    # events = streams_data['events'][:]
    on_task_events = streams_data['on_task_events'][:]
    # on_task_events_i = np.logical_or.reduce([events[:, 2] == teid for teid in meta_data['task_event_ids'].values()], axis=0)
    # on_task_events = events[on_task_events_i, :]

    # C3-Cz, C4-Cz
    tfr = streams_data['tfr_epochs_on_task'][:]
    if norm_c34_w_cz:
        cz = tfr[:, eeg_info['ch_names'].index('Cz')]
        tfr[:, eeg_info['ch_names'].index('C3')] = 2 * tfr[:, eeg_info['ch_names'].index('C3')] - cz
        tfr[:, eeg_info['ch_names'].index('C4')] = 2 * tfr[:, eeg_info['ch_names'].index('C4')] - cz

    # pick channels, downsample time
    tfr = tfr[..., ::2]
    times = times[::2]
    epochs = mne.time_frequency.EpochsTFR(eeg_info, tfr, times, freqs,
                                          verbose=verbose, events=on_task_events, event_id=meta_data['task_event_ids'])
    # epochs = epochs[:100]
    gen_erds_plots(epochs, subject, meta_data['task_event_ids'], out_folder=f'{output_path}/figures/combined',
                   freqs=freqs, comp_time_freq=True, comp_tf_clusters=False, channels=channels,
                   baseline=meta_data['baseline'], apply_baseline=False, copy=False)

    # tf clustering can only be done when the 'percent' baselining option is used
    # percent scales the tfr data into having negative and positive values, which is then used in the plotting
    #   function to determine significant positive (synchronization) or negative (desync) values
    if meta_data['tfr_baseline_mode'] == 'percent':
        gen_erds_plots(epochs, subject, meta_data['task_event_ids'], out_folder=f'{output_path}/figures/combined_clust',
                       freqs=freqs, comp_time_freq=True, comp_tf_clusters=True, channels=channels,
                       baseline=meta_data['baseline'], apply_baseline=False, copy=False)

    streams_data.close()


def main(
    subject='6808dfab',  # 0717b399 | 6808dfab
    session_ids=range(1, 4),  # range(1, 9) | range(1, 4)
    freqs=np.logspace(np.log(2), np.log(50), num=100, base=np.e),  # linear: np.arange(2, 50, 0.2)
    n_cycles=None,
    tfr_mode='multitaper',  # cwt | multitaper: multitaper is smoother on time
    do_plot=False,
    n_jobs=6,
    verbose=False,
    rerun_proc=False,
    bandpass_freq=(.5, 80),
    notch_freq=(50, 100),
    baseline=(-1, -.05),  # on task; don't use -1.5, remove parts related to the beep; -0.05
    eeg_sfreq=250, gamepad_sfreq=125,  # hardcoded for now
    reaction_tmax=0.5,
    store_per_session_recs=False,
    tfr_baseline_mode='percent',
    filter_percentile=95,
    combined_anal_channels=('C3', 'Cz', 'C4'),
    norm_c34_w_cz=False,
):

    recordings_path = '../recordings'
    exp_cfg_path = 'config/lr_finger/exp_me_l_r_lr_stim-w-dots.json'
    # output_path = f'out/{subject}'
    output_path = f'out_bl{baseline[0]}-{baseline[1]}_tfr-{tfr_mode}-{tfr_baseline_mode}_reac-{reaction_tmax}' \
                  f'_bad-{filter_percentile}_c34-{norm_c34_w_cz}/{subject}'
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
                        # 'epochs_on_pull': {'dtype': 'float32', 'epoch': True, 'get': lambda x: x.get_data()},
                        'epochs_on_break': {'dtype': 'float32', 'epoch': True, 'get': lambda x: x.get_data()},

                        'tfr_epochs_on_task': {'dtype': 'float32', 'epoch': True, 'get': lambda x: x.data},
                        # 'tfr_epochs_on_pull': {'dtype': 'float32', 'epoch': True, 'get': lambda x: x.data},
                        'tfr_epochs_on_break': {'dtype': 'float32', 'epoch': True, 'get': lambda x: x.data},

                        'gamepad': {'dtype': 'float16', 'epoch': False, 'get': lambda x: x},
                        'events': {'dtype': 'int32', 'epoch': True, 'get': lambda x: x},
                        'on_task_events': {'dtype': 'int32', 'epoch': True, 'get': lambda x: x},
                        # 'on_pull_events': {'dtype': 'int32', 'epoch': True, 'get': lambda x: x},
                        'on_break_events': {'dtype': 'int32', 'epoch': True, 'get': lambda x: x}}  # i know, it got out of hand

        # process sessions one-by-one
        num_epochs = []
        meta_names = ['eeg_info', 'event_dict', 'on_task_times', 'task_event_ids', 'break_event_id', 'eeg_ch_names']
        meta_data = {name: None for name in meta_names}
        meta_data['iafs'] = []  # individual alpha freq for each session
        meta_data['baseline'] = baseline
        meta_data['tfr_baseline_mode'] = tfr_baseline_mode
        meta_data['tfr_mode'] = tfr_mode
        meta_data['freqs'] = freqs
        meta_data['filter_percentile'] = filter_percentile

        for i, sid in enumerate(session_ids):

            rec_name = f'{subject}-{sid:03d}'
            fig_output_path = f'{output_path}/figures/{sid:03d}'

            # preprocess session
            sess = preprocess_session(recordings_path, rec_name, subject, sid, exp_cfg_path, eeg_sfreq, gamepad_sfreq,
                                      bandpass_freq, notch_freq, freqs, do_plot, reaction_tmax, n_jobs, verbose,
                                      fig_output_path)

            # create IAF plots
            _, _, iaf = process_eyes_open_closed(sess['filt_raw_eeg'], sess['events'], sess['eeg_info'],
                                                 do_plot=do_plot, output_path=output_path, verbose=verbose)
            meta_data['iafs'].append(iaf)

            # epoch sessions
            task_epochs_sess = epoch_on_task(sess, exp_cfg_path, freqs, baseline, tfr_mode, tfr_baseline_mode,
                                             n_cycles, verbose=verbose, filter_percentile=filter_percentile)
            # TODO see todo in pull_epochs_sess
            # pull_epochs_sess = epoch_on_pull(sess, exp_cfg_path, freqs, baseline, tfr_mode,
            #                                  tfr_baseline_mode, n_cycles, verbose=verbose, filter_percentile=filter_percentile)
            break_epochs_sess = epoch_on_break(sess, exp_cfg_path, freqs, baseline, tfr_mode, tfr_baseline_mode,
                                               n_cycles, verbose=verbose, filter_percentile=filter_percentile)
            sess = dict(**sess, **task_epochs_sess, **break_epochs_sess)  #, **pull_epochs_sess)

            # plots per session
            if do_plot:
                plot_some(sess, fig_output_path, rec_name, freqs, baseline)

            # extract session data and store num epochs for current session
            sess_data = {name: info['get'](sess[name]) for name, info in streams_info.items()}
            num_epochs.append(sess_data['epochs_on_task'].shape[0])

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
                    streams_data[ds_name].resize(streams_data[ds_name].shape[0] + data.shape[0], axis=0)
                    streams_data[ds_name][-data.shape[0]:] = data
                else:  # not epoched stream, create new dataset
                    streams_data.create_dataset(ds_name, dtype=info['dtype'], data=data)

            # store individual session files
            raw_path = f'{output_path}/{subject}_{sid:03d}_raw-eeg.fif'
            epochs_path = f'{output_path}/{subject}_{sid:03d}_epochs_on_task-epo.fif'
            tfr_epochs_path = f'{output_path}/{subject}_{sid:03d}_epochs_on_task-tfr.h5'

            if store_per_session_recs:
                sess['filt_raw_eeg'].save(raw_path, overwrite=True, verbose=verbose)
                sess['epochs_on_task'].save(epochs_path, overwrite=True, verbose=verbose)
                sess['tfr_epochs_on_task'].save(tfr_epochs_path, overwrite=True, verbose=verbose)

        streams_data.attrs['num_epochs'] = num_epochs
        streams_data.attrs['session_ids'] = sum([[sid] * num_epochs[i] for i, sid in enumerate(session_ids)], [])
        streams_data.attrs['on_task_times'] = meta_data['on_task_times']
        streams_data.attrs['freqs'] = meta_data['freqs']

        streams_data.close()
        print('h5 epoch files saved')

        # save metadata
        with open(meta_path, 'wb') as f:
            pickle.dump(meta_data, f)

    # analysis, plots generated from all the sessions of one subject
    combined_session_analysis(subject, streams_path, meta_path, output_path, combined_anal_channels,
                              norm_c34_w_cz, verbose)


# TODO upload new h5 versions


if __name__ == '__main__':

    main(do_plot=False, rerun_proc=True, combined_anal_channels=('C3', 'C4'), norm_c34_w_cz=True)
    # main(do_plot=False, rerun_proc=False, tfr_mode='cwt', freqs=np.logspace(np.log(4), np.log(50), num=100, base=np.e),
    #      combined_anal_channels=('C3', 'C4'), norm_c34_w_cz=True)

    # baseline
    # main(baseline=(-1, -.1), do_plot=True)
    # main(baseline=(-1, -.1), do_plot=True)
    # main(baseline=(-1.5, 0), do_plot=True)
    # main(baseline=(-1.5, -.1), do_plot=True)

    # tfr_mode
    # main(tfr_mode='cwt', do_plot=True, freqs=np.logspace(np.log(4), np.log(50), num=100, base=np.e))

    # tfr_baseline_mode
    # TODO don't try logratio
    # main(tfr_baseline_mode='ratio', do_plot=True)
    # main(tfr_baseline_mode='zscore', do_plot=True)
    # main(tfr_baseline_mode='zlogratio', do_plot=True)

    # reaction_tmax
    # main(reaction_tmax=.8, do_plot=True, rerun_proc=False)
    # main(reaction_tmax=.45, do_plot=True)
    # main(reaction_tmax=.4, do_plot=True)
