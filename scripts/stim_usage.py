""""
Sample script to Stimulus class
"""

import sys

sys.path.append("./")
import time
from utils.stim_utils import Stimulus
from config.arms_offline import config

stim = Stimulus(config.stim_folder,config.classes)
stim.show('Blank')
time.sleep(1)
stim.show('Fixation')
time.sleep(1)

for c in config.classes:
    stim.show(c)
    time.sleep(1)

stim.stop()