sample_length = 2

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
data_path = "data"
val_sessions = [("Nemes", "session_01"), ("Nemes", "session_06")]

resample_to = 128

DC_filter = True

bandpass = True
bandpass_freq = [8.0, 32.0] #alpha + beta
order = 4

notch = True
notch_freq = 50.0

#label_smoothing = 0.0

normalize = True
