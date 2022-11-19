import os
import time
import numpy as np

import brainflow
from brainflow.board_shim import (
    BoardShim,
    BrainFlowInputParams,
    BoardIds,
    BrainFlowError,
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
LSB = [0.045]  # µV


class ArcWrapper:
    def __init__(self):
        """
        Initialize a connection to the Arc device.
        """
        BoardShim.enable_dev_board_logger()
        params = BrainFlowInputParams()
        self.board = BoardShim(BoardIds.MINDROVE_WIFI_BOARD, params)

        try:
            self.board.stop_stream()
            self.board.release_session()
        except:
            ...

        try:
            self.board.prepare_session()
        except Exception as e:
            print(e.__str__())
            print("[ERROR]: Could not connect to Arc device.")
            list_ssids()
            quit()
        self.sample_rate = self.board.get_sampling_rate(16)

        print("Device ready (sampling rate: {}hz)".format(self.sample_rate))

    def listen(self):
        """
        Start listening to the Arc device.
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
        data = self.board.get_board_data()
        data = data[
            :n_channels, -int(n_data_points) :
        ]  # keep the data of the eeg channels only, and remove data over the trial length
        return data

    def stop(self):
        """
        Stop listening to the Arc device.
        """
        try:
            self.board.stop_stream()
            self.board.release_session()
        except:
            ...
