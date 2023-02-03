# lowest latency audio lib
# import sound after setting prefs: https://www.psychopy.org/api/preferences.html
from psychopy import prefs
prefs.hardware['audioLib'] = ['PTB']
prefs.hardware['audioLatencyMode'] = 3
print('PsychoPy preferences:\n', prefs)

from datetime import datetime
import json
import numpy as np
from psychopy import visual, core, sound
from psychopy import logging
import psychtoolbox  # if importing fails, ptb audio backend is not available
from pylsl import StreamInfo, StreamOutlet


def load_event(win, stim_dir, event, event_name):
    vis = None
    if 'img' in event:
        vis = visual.ImageStim(win, image=f'{stim_dir}/{event["img"]}', interpolate=True, name=event_name)
    elif 'txt' in event:
        vis = visual.TextStim(win, text=event['txt'], color='black')
    
    # note: psychopy ptb backend can only play 48kHz sounds
    tone = sound.Sound(**event['tone']) if 'tone' in event else None
    audio = sound.Sound(f'{stim_dir}/{event["audio"]}') if 'audio' in event else None
    marker = [event['marker']] if 'marker' in event else None  # lsl needs a list of markers

    return vis, tone, audio, marker


def fire_event(win, lsl_outlet, vis=None, tone=None, audio=None, marker=None):
    if vis:
        vis.draw()
    if marker:
        win.callOnFlip(lsl_outlet.push_sample, marker)
    if tone or audio:
        next_flip = win.getFutureFlipTime(clock='ptb')
        if tone:
            tone.play(when=next_flip)
        if audio:
            audio.play(when=next_flip)
    
    win.flip()


def experiment():

    # load cfg
    exp_cfg_path = 'config/lr_finger/exp_lr_nothing.json'
    exp_name = exp_cfg_path[exp_cfg_path.rfind('/') + 1:exp_cfg_path.rfind('.')]
    with open(exp_cfg_path, 'rt') as f:
        exp_cfg = json.load(f)

    # init lsl stream
    lsl_info = StreamInfo(name='exp-marker', type='Markers', channel_count=1, channel_format='int32', source_id='exp-script')
    lsl_outlet = StreamOutlet(lsl_info)

    # user input
    input('\nPress enter after LSL recorder has started..')
    ntrials_inp = input(f'Number of trials ({exp_cfg["ntrials"]}): ')
    ntrials = exp_cfg['ntrials'] if len(ntrials_inp) == 0 else int(ntrials_inp)
    event_durations = exp_cfg['event-duration']
    session_duration = event_durations['eyes-open'] + event_durations['eyes-closed'] + event_durations['eyes-move'] + \
                    ntrials * (event_durations['baseline'] + event_durations['task'] + event_durations['break'][1])
    print(f'Starting session with {ntrials} trials taking ~{session_duration / 60:.0f} minutes')

    # setup psychopy logging
    log_path = f'logs/{exp_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M")}.log'
    log_file = logging.LogFile(log_path, level=logging.INFO, filemode='w')

    # init window
    win = visual.Window((800, 600), screen=0, viewPos=(.5, .5), units='pix', fullscr=False, allowGUI=False, color='white')

    # load stims
    event_stims = {ev_name: load_event(win, exp_cfg['stim_dir'], ev, ev_name) for ev_name, ev in exp_cfg['events'].items()}
    task_stims = {ev_name: load_event(win, exp_cfg['stim_dir'], ev, ev_name) for ev_name, ev in exp_cfg['tasks'].items()}

    # start session
    core.wait(exp_cfg['cushion-time'])
    fire_event(win, lsl_outlet, *event_stims['session-beg'])
    core.wait(exp_cfg['cushion-time'])

    # eyes-open
    fire_event(win, lsl_outlet, *event_stims['eyes-open-beg'])
    core.wait(event_durations['eyes-open'])
    fire_event(win, lsl_outlet, *event_stims['eyes-open-end'])
    core.wait(exp_cfg['cushion-time'])

    # eyes-closed
    fire_event(win, lsl_outlet, *event_stims['eyes-closed-beg'])
    core.wait(event_durations['eyes-closed'])
    fire_event(win, lsl_outlet, *event_stims['eyes-closed-end'])
    core.wait(exp_cfg['cushion-time'])

    # eyes-move
    fire_event(win, lsl_outlet, *event_stims['eyes-move-beg'])
    core.wait(event_durations['eyes-move'])
    fire_event(win, lsl_outlet, *event_stims['eyes-move-end'])
    core.wait(exp_cfg['cushion-time'])

    # trials
    task_types = list(exp_cfg['tasks'].keys())
    rnd_tasks = np.random.randint(0, len(task_types), ntrials)
    rnd_break_dur = np.random.uniform(*event_durations['break'], ntrials)

    fire_event(win, lsl_outlet, *event_stims['trials-beg'])
    core.wait(exp_cfg['cushion-time'])

    for trial_i in range(ntrials):

        # baseline
        fire_event(win, lsl_outlet, *event_stims['baseline'])
        core.wait(event_durations['baseline'])

        # task
        task_type = task_types[rnd_tasks[trial_i]]
        fire_event(win, lsl_outlet, *task_stims[task_type])
        core.wait(event_durations['task'])

        # break
        fire_event(win, lsl_outlet, *event_stims['break'])
        core.wait(rnd_break_dur[trial_i])

    fire_event(win, lsl_outlet, *event_stims['session-end'])
    core.wait(exp_cfg['cushion-time'])

    win.close()
    logging.flush()
    print('Log stored at:', log_path)
    core.quit()


if __name__ == '__main__':
    experiment()
