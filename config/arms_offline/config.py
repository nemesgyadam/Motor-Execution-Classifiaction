# RECORDING PARAMS
data_path = "sessions"

sample_length = 2  # seconds
stand_by_time = 1  # seconds
classes = ["Rest", "Left", "Right"]
stim_folder = 'arrows'

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

relax_time = 10

# TRAIN PARAMS
batch_size = 120

train_sessions = [
    ("Erno", "session_06"),
    ("Erno2", "session_01"),
    ("Erno2", "session_02"),
    ("Erno2", "session_03"),
    ("Erno2", "session_04"),
]

val_sessions = [("Erno", "session_08"), ("Erno2", "session_05")]

resample_to = None

DC_filter = False

bandpass = True
bandpass_freq = [4.0, 20.0]
order = 4

notch = False
notch_freq = 50.0

# label_smoothing = 0.0

normalize = False
