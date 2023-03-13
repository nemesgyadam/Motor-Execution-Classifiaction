# import sound module after setting audio prefs: https://www.psychopy.org/api/preferences.html
from psychopy import prefs
prefs.hardware['audioLib'] = ['PTB']  # lowest latency audio lib
prefs.hardware['audioLatencyMode'] = 3  # real-time af
print('PsychoPy preferences:\n', prefs)

import sys, os
import json
from datetime import datetime
import numpy as np
from pylsl import StreamInfo, StreamOutlet
from psychopy import visual, core, sound
from psychopy import logging
import psychtoolbox  # if importing fails, ptb audio backend is not available
import hashlib  


def load_event(win, stim_dir, event, event_name):
    vis = None
    if 'img' in event:
        vis = visual.ImageStim(win, image=f'{stim_dir}/{event["img"]}', interpolate=True, name=event_name, size=600)
    elif 'txt' in event:
        vis = visual.TextStim(win, text=event['txt'], color='black', height=32)
    
    # note: psychopy ptb audio backend can only play 48kHz sounds
    tone = sound.Sound(**event['tone']) if 'tone' in event else None
    audio = sound.Sound(f'{stim_dir}/{event["audio"]}') if 'audio' in event else None
    marker = [event['marker']] if 'marker' in event else None  # lsl needs a list of markers

    return vis, tone, audio, marker


def fire_event(win: visual.Window, lsl_outlet: StreamOutlet, vis=None, tone=None, audio=None, marker=None):
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
    exp_cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'config/lr_finger/exp_me_l_r_lr_stim-w-dots.json'
    exp_name = exp_cfg_path[exp_cfg_path.rfind('/') + 1:exp_cfg_path.rfind('.')]
    with open(exp_cfg_path, 'rt') as f:
        exp_cfg = json.load(f)

    # init lsl stream
    lsl_info = StreamInfo(name='exp-marker', type='Markers', channel_count=1, channel_format='int32', source_id='exp-script')
    lsl_outlet = StreamOutlet(lsl_info)

    # setup psychopy logging
    os.makedirs('logs', exist_ok=True)
    log_path = f'logs/{exp_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M")}.log'
    log_file = logging.LogFile(log_path, level=logging.INFO, filemode='w')
    cfg_json_str = json.dumps(exp_cfg)
    logging.log(f'CFG:{hashlib.md5(cfg_json_str.encode()).hexdigest()[:8]}:{cfg_json_str}', logging.INFO)
    print('saving logs to:', log_path)

    # set up trials
    input('\n#######################################\n'
          'Fill in and Start LabRecorder then press Enter..'
          '\n#######################################')
    ntrials_inp = input(f'Number of trials ({exp_cfg["ntrials"]}): ')
    ntrials = exp_cfg['ntrials'] if len(ntrials_inp) == 0 else int(ntrials_inp)
    event_durations = exp_cfg['event-duration']
    session_duration = event_durations['eyes-open'] + event_durations['eyes-closed'] + event_durations['eyes-move'] + \
                       ntrials * (event_durations['baseline'] + event_durations['task'][1] + event_durations['break'][1])
    print(f'Starting session with {ntrials} trials taking ~{session_duration / 60:.0f} minutes')
    core.wait(exp_cfg['cushion-time'])

    # init window
    win = visual.Window((1920, 1080), screen=0, viewPos=(.5, .5), units='pix', fullscr=True, allowGUI=False, color='#aaaaaa')  # TODO hide talca pls

    # load stims
    event_stims = {ev_name: load_event(win, exp_cfg['stim_dir'], ev, ev_name) for ev_name, ev in exp_cfg['events'].items()}
    task_stims = {ev_name: load_event(win, exp_cfg['stim_dir'], ev, ev_name) for ev_name, ev in exp_cfg['tasks'].items()}

    # start session
    logging.log(f'Session begins with {ntrials} trials', logging.INFO)
    core.wait(exp_cfg['cushion-time'])
    fire_event(win, lsl_outlet, *event_stims['session-beg'])
    core.wait(exp_cfg['cushion-time'])

    fire_event(win, lsl_outlet, *event_stims['welcome'])
    core.wait(exp_cfg['cushion-time'])

    # eyes-open
    logging.log(f'Open-eye begins', logging.INFO)
    fire_event(win, lsl_outlet, *event_stims['eyes-open-beg'])
    core.wait(event_durations['eyes-open'])
    fire_event(win, lsl_outlet, *event_stims['eyes-open-end'])
    core.wait(exp_cfg['cushion-time'])

    # eyes-closed
    logging.log(f'Closed-eye begins', logging.INFO)
    fire_event(win, lsl_outlet, *event_stims['eyes-closed-beg'])
    core.wait(event_durations['eyes-closed'])
    fire_event(win, lsl_outlet, *event_stims['eyes-closed-end'])
    core.wait(exp_cfg['cushion-time'])

    # eyes-move
    logging.log(f'Eye-movement begins', logging.INFO)
    fire_event(win, lsl_outlet, *event_stims['eyes-move-beg'])
    core.wait(event_durations['eyes-move'])
    fire_event(win, lsl_outlet, *event_stims['eyes-move-end'])
    core.wait(exp_cfg['cushion-time'])

    # trials
    logging.log(f'Trials begins', logging.INFO)
    task_types = list(exp_cfg['tasks'].keys())
    rnd_tasks = np.random.permutation(np.concatenate([np.repeat(i, ntrials // len(task_types) + 1)
                                                      for i in range(len(task_types))]))[:ntrials]
    rnd_task_dur = np.random.uniform(*event_durations['task'], ntrials)
    rnd_break_dur = np.random.uniform(*event_durations['break'], ntrials)

    word_per_sec = 2
    time2read = lambda txt: len(txt.split(' ')) * (1 / word_per_sec) + exp_cfg['cushion-time']
    fire_event(win, lsl_outlet, *event_stims['trials-instruct-1'])
    core.wait(time2read(exp_cfg['events']['trials-instruct-1']['txt']))
    fire_event(win, lsl_outlet, *event_stims['trials-instruct-2'])
    core.wait(20)
    fire_event(win, lsl_outlet, *event_stims['trials-instruct-3'])
    core.wait(time2read(exp_cfg['events']['trials-instruct-3']['txt']))

    fire_event(win, lsl_outlet, *event_stims['trials-beg'])
    core.wait(2 * exp_cfg['cushion-time'])

    for trial_i in range(ntrials):
        print(f'{trial_i} trial begins')

        # baseline
        fire_event(win, lsl_outlet, *event_stims['baseline'])
        core.wait(event_durations['baseline'])

        # task
        task_type = task_types[rnd_tasks[trial_i]]
        logging.log(f'#{trial_i} trial task: {task_type}', logging.INFO)
        fire_event(win, lsl_outlet, *task_stims[task_type])
        core.wait(rnd_task_dur[trial_i])

        # break
        fire_event(win, lsl_outlet, *event_stims['break'])
        core.wait(rnd_break_dur[trial_i])

    # end session
    logging.log(f'Session ends', logging.INFO)
    fire_event(win, lsl_outlet, *event_stims['session-end'])
    core.wait(exp_cfg['cushion-time'])

    win.close()
    logging.flush()
    print('Log stored at:', log_path)
    core.quit()


if __name__ == '__main__':
    experiment()
