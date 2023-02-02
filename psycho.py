import time
import json
import numpy as np
from psychopy import visual, core
from pylsl import StreamInfo, StreamOutlet

# TODO open/closed/moving eyes


def load_stim(win, stim_dir, event, event_name):
    stim = None
    if 'img' in event:
        stim = visual.ImageStim(win, image=f'{stim_dir}/{event["img"]}', interpolate=True, name=event_name)
    elif 'txt' in event:
        stim = visual.TextStim(win, text=event['txt'])
    
    if 'audio' in event:
        pass  # TODO implement audio for baseline start

    marker = [event['marker']] if 'marker' in event else []  # lsl needs a list of markers
    return stim, marker


def fire_event(win, lsl_outlet, stim, marker):
    if stim:
        stim.draw()
    win.callOnFlip(lsl_outlet.push_sample, marker)
    win.flip()


# load cfg
exp_cfg_path = 'config/lr_finger/exp_lr_nothing.json'
with open(exp_cfg_path, 'rt') as f:
    exp_cfg = json.load(f)

# init lsl stream
lsl_info = StreamInfo(name='ext-marker', type='Markers', channel_count=1, channel_format='int32', source_id='exp-script')
lsl_outlet = StreamOutlet(lsl_info)

# init window
win = visual.Window((800, 600), screen=0, viewPos=(.5, .5), units='pix', fullscr=False, allowGUI=False)

# load stims
event_stims = {ev_name: load_stim(win, exp_cfg['stim_dir'], ev, ev_name) for ev_name, ev in exp_cfg['events'].items()}
task_stims = {ev_name: load_stim(win, exp_cfg['stim_dir'], ev, ev_name) for ev_name, ev in exp_cfg['tasks'].items()}

# start session
core.wait(3)
fire_event(win, lsl_outlet, *event_stims['session-beg'])
core.wait(3)

fire_event(win, lsl_outlet, *event_stims['eyes-open-beg'])
core.wait(exp_cfg['event-duration']['eyes-open'])
fire_event(win, lsl_outlet, *event_stims['eyes-open-end'])

core.wait(3)

fire_event(win, lsl_outlet, *event_stims['eyes-closed-beg'])
core.wait(exp_cfg['event-duration']['eyes-closed'])
fire_event(win, lsl_outlet, *event_stims['eyes-closed-end'])

core.wait(3)

fire_event(win, lsl_outlet, *event_stims['eyes-move-beg'])
core.wait(exp_cfg['event-duration']['eyes-move'])
fire_event(win, lsl_outlet, *event_stims['eyes-move-end'])

core.wait(3)

# trials
task_types = list(exp_cfg['tasks'].keys())
rnd_tasks = np.random.randint(0, len(task_types), exp_cfg['ntrials'])
rnd_break_dur = np.random.uniform(*exp_cfg['event-duration']['break'], exp_cfg['ntrials'])

for trial_i in range(exp_cfg['ntrials']):  # TODO or provide ntrials as user input

    # baseline
    fire_event(win, lsl_outlet, *event_stims['baseline'])
    core.wait(exp_cfg['event-duration']['baseline'])

    # task
    task = task_types[rnd_tasks[trial_i]]
    fire_event(win, lsl_outlet, *task_stims[task])
    core.wait(exp_cfg['event-duration']['task'])

    # break
    fire_event(win, lsl_outlet, *task_stims[task])
    core.wait(rnd_break_dur[trial_i])

fire_event(win, lsl_outlet, *event_stims['session-end'])
