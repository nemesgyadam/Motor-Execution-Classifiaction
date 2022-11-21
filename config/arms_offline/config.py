# RECORDING PARAMS
data_path = "sessions"

sample_length = 2   # seconds
stand_by_time = 1   # seconds
classes = ["Rest", "Left", "Right"]

# commands = [
#     "Hold still!", "Raise your  LEFT hand!       <----",
#     "Raise your RIGHT hand!       ---->",
#     "Hold still!",
# ]

commands = [
    "Hold still!",
    "Fist your LEFT hand!       <----",
    "Fist your RIGHT hand!       ---->",
]

# TRAIN PARAMS
batch_size = 120

train_sessions = [('Erno', 'session_06')]
val_sessions = [('Erno', 'session_08')]

resample_to = None

DC_filter = True

bandpass = True
bandpass_freq = [8.0, 32.0] #alpha + beta
order = 4

notch = True
notch_freq = 50.0

#label_smoothing = 0.0

normalize = True
