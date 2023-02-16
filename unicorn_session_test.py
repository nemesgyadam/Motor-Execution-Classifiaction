from utils.Unicorn import UnicornWrapper
import time

# unicorn 2 lsl is implemented here: https://gist.github.com/robertoostenveld/6f5f765268847f684585be9e60ecfb67

device = UnicornWrapper()
device.start_session()

while True:
    data = device.get_session_data()

    if data.size > 0:
        ts = data[0, :]
        triggers = data[1, :]
        eeg = data[2:, :]
        print(eeg.dtype)
    time.sleep(1)   
