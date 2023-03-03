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


def process_session(rec_base_path, subject, session, exp_cfg_path='config/lr_finger/exp_me_l_r_lr_stim-w-dots.json',
                    eeg_sfreq=250, gamepad_sfreq=125, bandpass_freq=(0.5, 80), notch_freq=(50, 100), tfr_mode='cwt',  # cwt | multitaper
                    freqs=np.arange(2, 50, 0.2), n_cycles=None, tfr_baseline_mode='ratio', do_plot=False, reaction_tmax=1.,
                    comp_break_epochs=False, n_jobs=4, verbose=False):
    
    print('-' * 50 +  f'\nPROCESSING: {subject}/{session:03d}\n' + '-' * 50)

    # recording path
    rec_path = f'{rec_base_path}/sub-{subject}/ses-S{session:03d}/eeg/sub-{subject}_ses-S{session:03d}_task-me-l-r-lr_run-001_eeg.xdf'
    rec_name = f'{subject}-{session:03d}'

    # experiment config
    with open(exp_cfg_path, 'rt') as f:
        exp_cfg = json.load(f)

    # channel info
    shared_sfreq = max(eeg_sfreq, gamepad_sfreq)
    eeg_ch_names = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
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

    # TODO check if the same number of on-off happened, pair on-offs, remove ones that can't be paired
    print('trigger on/off times should be equal:')
    print('  L2:', triggers_on_i[0].shape[0], '==', triggers_off_i[0].shape[0])
    print('  R2:', triggers_on_i[1].shape[0], '==', triggers_off_i[1].shape[0])

    # plotting commands for debugging
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

    # event dictionary - event_name: marker
    event_dict = {**{ename: einfo['marker'] for ename, einfo in exp_cfg['tasks'].items()},
                  **{ename: einfo['marker'] for ename, einfo in exp_cfg['events'].items() if 'marker' in einfo},
                  **{f'{trig}-on': mark for mark, trig in zip(trigger_on_event_markers, ['L2', 'R2'])},
                  **{f'{trig}-off': mark for mark, trig in zip(trigger_off_event_markers, ['L2', 'R2'])}}
    
    # preprocess numpy eeg, before encaptulating into mne raw
    # common median referencing - substract compute median at each timepoint
    eeg -= np.median(eeg, axis=0)
    
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
    if do_plot:
        raw.plot_sensors(show_names = True)

    # # add stim channel - not needed, events need to be provided separately to the epoching method anyways
    # stim_info = mne.create_info(['STI'], raw.info['sfreq'], ['stim'])  # separate events = stim channel
    # stim_raw = mne.io.RawArray(np.zeros((1, len(raw.times))), stim_info)
    # raw.add_channels([stim_raw], force_update_info=True)
    # raw.add_events(all_events, stim_channel='STI')
    
    # filter and run tfr on the whole recording (if possible)
    filt_raw_eeg = raw.copy().pick_types(eeg=True).filter(*bandpass_freq, h_trans_bandwidth=5, n_jobs=n_jobs, verbose=verbose) \
        .notch_filter(notch_freq, trans_bandwidth=1, n_jobs=n_jobs, verbose=verbose)
    n_cycles = (np.log(freqs) * 2 + 3).astype(np.int32) if n_cycles is None else n_cycles  # conservatively high freq (and low time) resolution
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
        plt.figure()
        plt.title('button press delay')
        plt.hist(left_success[1], bins=20, label='left', alpha=.5)
        plt.hist(right_success[1], bins=0, label='right', alpha=.5)
        # plt.hist(lr_success[1], bins=10, label='left-right', alpha=.5)  # define_target_events2 fucks lags up
        plt.legend()
        plt.show(block=False)
    
    # do the same for the break event
    break_marker = exp_cfg['events']['break']['marker']
    break_success_marker = int(f'{break_marker}0')
    break_success = define_target_events2(all_events, break_marker, [event_dict['L2-off'], event_dict['R2-off']],
                                          shared_sfreq, tmin=1e-4, tmax=reaction_tmax, new_id=break_success_marker)
    
    all_events = np.concatenate([all_events, left_success[0], right_success[0], lr_success[0], nothing_success[0], break_success[0]], axis=0)
    
    # some stats
    num_old = [(all_events[:, 2] == event_dict[ev]).sum() for ev in ['left', 'right', 'left-right', 'nothing']]
    num_succ = [(all_events[:, 2] == ev).sum() for ev in [left_success_marker, right_success_marker, lr_success_marker, nothing_success_marker]]
    print('num of events successful/originally:',
          {name: f'{nsucc}/{nold}' for name, nold, nsucc in zip(['left', 'right', 'left-right', 'nothing'], num_old, num_succ)})
    
    # epoching
    # TODO add btn press epochs
    baseline = (None, 0)
    task_event_ids = [left_success_marker, right_success_marker, lr_success_marker, nothing_success_marker]
    break_event_id = break_success_marker
    epochs_on_task = mne.Epochs(filt_raw_eeg, all_events, event_id=task_event_ids, baseline=baseline, verbose=verbose,
                                tmin=-exp_cfg['event-duration']['baseline'], tmax=exp_cfg['event-duration']['task'][0])
    epochs_on_task.apply_baseline(baseline, verbose=verbose)
    
    epochs_on_break = None
    if comp_break_epochs:
        epochs_on_break = mne.Epochs(filt_raw_eeg, all_events, event_id=break_event_id, baseline=baseline, verbose=verbose,
                                 tmin=-exp_cfg['event-duration']['task'][0] / 2, tmax=exp_cfg['event-duration']['break'][0])
        epochs_on_break.apply_baseline(baseline, verbose=verbose)
    
    # tfr: cwt or multitaper; multitaper is better if the signal is not that time-locked
    tfr_settings = dict(freqs=freqs, n_cycles=n_cycles, return_itc=False, average=False, n_jobs=n_jobs, verbose=verbose)
    tfr_epochs_on_task = tfr_epochs_on_break = None
    if tfr_mode == 'cwt':
        tfr_epochs_on_task = mne.time_frequency.tfr_morlet(epochs_on_task, output='power', **tfr_settings)
        if comp_break_epochs:
            tfr_epochs_on_break = mne.time_frequency.tfr_morlet(epochs_on_break, output='power', **tfr_settings)
    elif tfr_mode == 'multitaper':
        tfr_epochs_on_task = mne.time_frequency.tfr_multitaper(epochs_on_task, **tfr_settings)
        if comp_break_epochs:
            tfr_epochs_on_break = mne.time_frequency.tfr_multitaper(epochs_on_break, **tfr_settings)
    else:
        raise ValueError('invalid tfr_mode:', tfr_mode)
    
    # tfr epoch prep
    tfr_epochs_on_task.apply_baseline(baseline, tfr_baseline_mode, verbose=verbose)
    if comp_break_epochs:
        tfr_epochs_on_break.apply_baseline(baseline, tfr_baseline_mode, verbose=verbose)

    # some plots
    erds_event_ids = {e: i for i, e in enumerate(task_event_ids)}
    if do_plot:
        filt_raw_eeg.plot(scalings=80, events=all_events, title='all events')
        epochs_on_task.plot(scalings=80, events=all_events, n_epochs=4, title='epochs locked on task')
        epochs_on_break.plot(scalings=80, events=all_events, n_epochs=4, title='epochs locked on break')

        # ERPs
        epochs_on_task.average().plot(scalings=50, title='ERP locked on task')
        epochs_on_break.average().plot(scalings=50, title='ERP locked on break')
        
        # ERPs per task
        epochs_on_task[event_dict['left']].average().plot(scalings=50, window_title='left_trigger')
        epochs_on_task[event_dict['right']].average().plot(scalings=50, window_title='right_trigger')
        epochs_on_task[event_dict['left-right']].average().plot(scalings=50, window_title='left_right_trigger')
        epochs_on_task[event_dict['nothing']].average().plot(scalings=50, window_title='no_trigger')

        gen_erds_plots(epochs_on_task, rec_name, erds_event_ids, out_folder=f'out/{rec_name}/tfr_raw', freqs=freqs,
                    comp_time_freq=True, comp_tf_clusters=False, channels=('C3', 'Cz', 'C4'), baseline=baseline)

        gen_erds_plots(epochs_on_task, rec_name, erds_event_ids, out_folder=f'out/{rec_name}/tfr_clust', freqs=freqs,
                    comp_time_freq=True, comp_tf_clusters=True, channels=('C3', 'Cz', 'C4'), baseline=baseline)
    
    return {'epochs_on_task': epochs_on_task, 'epochs_on_break': epochs_on_break,
            'tfr_epochs_on_task': tfr_epochs_on_task, 'tfr_epochs_on_break': tfr_epochs_on_break,
            'filt_raw_eeg': filt_raw_eeg, 'gamepad': gamepad, 'erds_event_ids': erds_event_ids,
            'events': all_events, 'event_dict': event_dict, 'eeg_info': eeg_info}


if __name__ == '__main__':
    subject = '0717b399'
    session_ids = range(1, 8)
    freqs = np.arange(2, 50, 0.2)
    tfr_mode = 'multitaper'
    do_plot = False
    n_jobs = 4
    verbose = False
    
    recordings_path = '../recordings'
    meta_path = f'out/{subject}/meta.pckl'
    streams_path = f'out/{subject}/{subject}_streams.h5'
    os.makedirs(f'out/{subject}', exist_ok=True)
    
    if not os.path.isfile(epochs_on_task_path):
        
        # prepare h5 dataset: combined raw, epochs, gamepad streams
        streams_data = h5py.File(streams_path, 'w')
        streams_info = {'filt_raw_eeg':       {'dtype': 'float32', 'epoch': False, 'get': lambda x: x.get_data()},
                        'epochs_on_task':     {'dtype': 'float32', 'epoch': True, 'get': lambda x: x.get_data()},
                        'tfr_epochs_on_task': {'dtype': 'float32', 'epoch': True, 'get': lambda x: x.data},
                        'gamepad':            {'dtype': 'float16', 'epoch': False, 'get': lambda x: x}}  # i know, it got out of hand
        
        # process sessions one-by-one
        num_epochs = []
        meta_data = dict(eeg_info=None, erds_event_ids=None, event_dict=None)
        
        for i, sid in enumerate(session_ids):
            sess = process_session(recordings_path, subject, sid, tfr_mode=tfr_mode, n_jobs=n_jobs, do_plot=do_plot, verbose=verbose)
            
            sess_data = {name: info['get'](sess[name]) for name, info in streams_info.items()}
            num_epochs.append(sess_data['epochs_on_task'].shape[0])
            
            # determine size of h5 datasets after first session processing
            # all epochs subsequent of sessions should have the same size (apart from the number of trials)
            if i == 0:
                for name, data in sess_data.items():
                    info = streams_info[name]
                    if info['epoch']:  # only for epoched data, there is a separate dataset created for each non-epoch stream
                        streams_data.create_dataset(name, (0, *data.shape[1:]), dtype=info['dtype'], maxshape=(None, *data.shape[1:]))
                
                # assign meta data
                for meta_name in meta_data.keys():
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
            raw_path = f'out/{subject}/{subject}_{sid:03d}_raw-eeg.fif'
            epochs_path = f'out/{subject}/{subject}_{sid:03d}_epochs_on_task-epo.fif'
            tfr_epochs_path = f'out/{subject}/{subject}_{sid:03d}_epochs_on_task-tfr.h5'
            
            sess['filt_raw_eeg'].save(raw_path, overwrite=True, verbose=verbose)
            sess['epochs_on_task'].save(epochs_path, overwrite=True, verbose=verbose)
            sess['tfr_epochs_on_task'].save(tfr_epochs_path, overwrite=True, verbose=verbose)
        
        streams_data.attrs['num_epochs'] = num_epochs
        streams_data.close()
        print('h5 epoch files saved')
        
        # save metadata
        with open(meta_path, 'wb') as f:
            pickle.dump(meta_data, f)
    
    else:  # already computed
        with open(all_data_path, 'rb') as f:
            sessions = pickle.load(f)  # TODO separately save lightweight meta data
        all_epochs_on_task = mne.read_epochs(f'out/{subject}/epochs_on_task-epo.fif')
        all_tfr_epochs_on_task = mne.time_frequency.read_tfrs(f'out/{subject}/epochs_on_task-tfr.h5')
    
    # TODO load hdf5 and run gen_erds_plots
    
    # freqs = np.arange(2, 50, 0.2)
    # gen_erds_plots(all_tfr_epochs_on_task, subject, erds_event_ids, out_folder=f'out/{subject}', freqs=freqs,
    #                 comp_time_freq=True, comp_tf_clusters=False, channels=('C3', 'Cz', 'C4'), baseline=None)
    # gen_erds_plots(all_tfr_epochs_on_task, subject, erds_event_ids, out_folder=f'out/{subject}', freqs=freqs,
    #                 comp_time_freq=True, comp_tf_clusters=False, channels=('C3', 'Cz', 'C4'), baseline=None)
    
    print(f'data loaded for subject {subject}')