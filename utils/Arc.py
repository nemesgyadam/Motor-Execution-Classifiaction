"""
Wrapper class for Minrove Arc EEG device.
All access to the device should be done through this class.
"""
import threading
import time
import numpy as np


from brainflow.board_shim import (
    BoardShim,
    BrainFlowInputParams,
    BoardIds,
)

from utils.wifi_utils import list_ssids

arc_channels = [
    "C5",
    "C3",
    "C1",
    "C2",
    "C4",
    "C6",
    "Vbat",
    "Trigger",
    "G00",
    "G01",
    "G10",
    "G11",
    "G20",
    "G21",
    "N",
    "fs",
]
EEG_channels = [0, 1, 2, 3, 4, 5]
LSB = 0.045  # µV


class ArcWrapper:
    def __init__(self):
        """
        Initialize a connection to the Arc device.
        """
        BoardShim.enable_dev_board_logger()
        params = BrainFlowInputParams()
        self.board = BoardShim(BoardIds.MINDROVE_WIFI_BOARD, params)

        self.sample_rate = self.board.get_sampling_rate(16)
        self.frame_length = 100  # 100 samples per request

        self.release()  # Release any potentional previous sessions

        try:
            self.board.prepare_session()
        except Exception as e:
            print(e.__str__())
            print("[ERROR]: Could not connect to Arc device.")
            list_ssids()
            quit()

        print("Device ready (sampling rate: {}hz)".format(self.sample_rate))

    def start_session(self):
        """
        Continously listen to the Arc device.
        """
        buffer_size = 450000
        self.board.start_stream(buffer_size)
        self.session_thread = threading.Thread(
            target=self.session, daemon=True
        )
        self.session_thread.start()
        self.triggers = []
        time.sleep(0.1)

    def session(self):
        """
        This thread responsible for listening to the Arc device.
        It will continously listen to the device for a given amount of minutes.
        And also handles the triggers.
        """

        self.session_buffer = []

        _ = self.board.get_board_data()  # Remove data from ring buffer
        
        self.stop_session = False
        while not self.stop_session:
            time.sleep(self.frame_length / self.sample_rate)
            data = self.board.get_board_data().T  # Removes data from ring buffer
            start_timestamp = time.time_ns()
            EEG_data = data[:, EEG_channels]
            actual_data_received = EEG_data.shape[0]

            time_stamp_stream = self.get_time_stamp_stream(
                start_timestamp, actual_data_received
            )
            trigger_stream = self.get_trigger_stream(time_stamp_stream)
            #TODO concat rosz
            self.session_buffer.append(
                np.vstack((time_stamp_stream, trigger_stream, EEG_data.T))
            )



    def trigger(self, trigger):
        """
        Send a trigger to the buffer.
        """
        self.triggers.append((time.time_ns(), trigger))

    def get_time_stamp_stream(self, start_timestamp, actual_data_received):
        """
        Generate time stamp for frames.
        """
        time_stamps = np.zeros(actual_data_received)
        for i in range(actual_data_received):
            time_between_two_samples = 1000 / self.sample_rate
            delay_ms = time_between_two_samples * (actual_data_received - i)
            delay_ns = delay_ms * 1000000

            time_stamps[i] = (
                start_timestamp - delay_ns
            )  # -start_time for relative time stamps
        return time_stamps

    def get_trigger_stream(self, time_stamps):
        """
        Collect triggers from trigger buffer.
        And add them to main buffer, based on timestamp.
        """
        trigger_stream = np.zeros(len(time_stamps))

        # Iterate over buffer triggers
        # Find closes timestamp to trigger
        for trigger in self.triggers:
            if trigger[0] > time_stamps[0] and trigger[0] < time_stamps[-1]:
                closest_timestamp = np.argmin(np.abs(time_stamps - trigger[0]))
                trigger_stream[closest_timestamp] = trigger[1]
            else:
                if trigger[0] < time_stamps[0]:
                    trigger_stream[0] = trigger[1]
                else:
                    trigger_stream[-1] = trigger[1]
        self.triggers = []  # Clear trigger buffer
        return trigger_stream

    def get_session_data(self):
        """
        Get the complete data from the Arc device.
        """
        return np.hstack(self.session_buffer)

    def listen(self):
        """
        Start listening to the Arc device.
        Stores the data in board object.
        """
        buffer_size = 450000
        self.board.start_stream(buffer_size)

    def get_latest_data(self, n_data_points=1000, EEG_only=True):
        """
        Get the latest data from the Arc device.
        """
        if EEG_only:
            n_channels = len(EEG_channels)
        else:
            n_channels = len(arc_channels)
        data = (
            self.board.get_current_board_data()
        )  # Doesn't remove data from ring buffer
        data = self.board.get_board_data()  # Removes data from ring buffer
        data = data[
            :n_channels, -int(n_data_points) :
        ]  # keep the data of the eeg channels only, and remove data over the trial length
        return data

    def release(self):
        """
        Release the Arc device.
        """
        try:
            self.board.stop_stream()
            self.board.release_session()
        except:
            ...

    def stop(self):
        """
        Stop listening to the Arc device.
        And release all resources.
        """
        self.stop_session = True
        self.session_thread.join()
        self.release()
