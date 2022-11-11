import numpy as np
import UnicornPy
import time
from utils.time import timeit

#unicorn_channels = "FP1, FFC1, FFC2, FCZ, CPZ, CPP1, CPP2, PZ, AccelX, AccelY, AccelZ, GyroX, GyroY, GyroZ, Battery, Sample"
EEG_channels = list(
    range(
        UnicornPy.EEGConfigIndex, UnicornPy.EEGConfigIndex + UnicornPy.EEGChannelsCount
    )
)  # [0,1,2,3,4,5,6,7]


class UnicornWrapper:
    def __init__(self):
        self.frame_length = 25  # number of samples in between 1 and 25 acquired per acquisition cycle.

        # Get available device serials.
        self.deviceList = UnicornPy.GetAvailableDevices(True)

        if len(self.deviceList) <= 0 or self.deviceList is None:
            raise Exception("No device available.Please pair with a Unicorn first.")

        # Print available device serials.
        print("Available Unicorn devices:")
        i = 0
        for device in self.deviceList:
            print("#%i %s" % (i, device))
            i += 1

        # Request device selection.
        print()
        if len(self.deviceList) == 1:
            self.deviceID = 0
        else:
            self.deviceID = int(input("Select device by ID #"))
            if self.deviceID < 0 or self.deviceID > len(self.deviceList):
                raise IndexError("The selected device ID is not valid.")

        self.testsignale_enabled = False
        self.connect()
        self.total_channels = self.device.GetNumberOfAcquiredChannels()

    def connect(self):
        self.device_name = self.deviceList[self.deviceID]
        print(f"Trying to connect to {self.device_name}")
        try:
            self.device = UnicornPy.Unicorn(self.device_name)
        except UnicornPy.DeviceException as e:
            print(e)
            quit()

        except Exception as e:
            print("An unknown error occured. %s" % e)
        print(f"Connected to {self.device_name}")
        print()

    #@timeit()
    def get_data(self, duration=10):
        try:
            self.device.StopAcquisition()
        except UnicornPy.DeviceException as e:
            ...

        n_data_points = int(duration * UnicornPy.SamplingRate)
        data_buffer = np.zeros((len(EEG_channels), n_data_points))
        try:
            receiveBufferBufferLength = int(self.frame_length * self.total_channels * 4)
            receiveBuffer = bytearray(receiveBufferBufferLength)

            self.device.StartAcquisition(self.testsignale_enabled)

            # Dummy run to get the device ready
            for i in range(40):
                self.device.GetData(
                        self.frame_length, receiveBuffer, receiveBufferBufferLength
                    )

            for i in range(int(n_data_points / self.frame_length)):
                self.device.GetData(
                    self.frame_length, receiveBuffer, receiveBufferBufferLength
                )
                data = np.frombuffer(
                    receiveBuffer,
                    dtype=np.float32,
                    count=self.frame_length * self.total_channels,
                )
                data = np.reshape(data, (self.frame_length, self.total_channels))
                EEG_data = data[:, EEG_channels]
                data_buffer[
                    :, i * self.frame_length : (i + 1) * self.frame_length
                ] = EEG_data.T
            del receiveBuffer
            self.device.StopAcquisition()

        except UnicornPy.DeviceException as e:
            print(e)
            return -1
        except Exception as e:
            print("An unknown error occured. %s" % e)
            return -2
        return data_buffer

    def stop(self):
        try:
            self.device.StopAcquisition()
        except UnicornPy.DeviceException as e:
            ...

        try:
            del self.device
        except Exception as e:
            ...
