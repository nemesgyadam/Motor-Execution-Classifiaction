"""
Wrapper class for gTec Unicorn EEG device.
All access to the device should be done through this class.
"""
import numpy as np
import UnicornPy
import time
import threading
from utils.time import timeit
#Fz, C3, Cz, C4, Pz, PO7, Oz and PO8
unicorn_channels = [
    "FP1",
    "FFC1",
    "FFC2",
    "FCZ",
    "CPZ",
    "CPP1",
    "CPP2",
    "PZ",
    "AccelX",
    "AccelY",
    "AccelZ",
    "GyroX",
    "GyroY",
    "GyroZ",
    "Battery",
    "Sample",
]
EEG_channels = list(
    range(
        UnicornPy.EEGConfigIndex, UnicornPy.EEGConfigIndex + UnicornPy.EEGChannelsCount
    )
)  # [0,1,2,3,4,5,6,7]


class UnicornWrapper:
    def __init__(self):
        """
        Initialize the Unicorn device.
        And connect to it.
        """
        self.frame_length = 25  # value 1 and 25 acquired per acquisition cycle.
        self.testsignale_enabled = False

        self.select_device()
        self.connect()

        self.total_channels = self.device.GetNumberOfAcquiredChannels()
        self.sample_rate = UnicornPy.SamplingRate

    def select_device(self):
        """
        Get available device serials.
        If there is only one device, select it.
        If there is more, choose from the available devices.
        """
        self.deviceList = UnicornPy.GetAvailableDevices(True)

        if len(self.deviceList) <= 0 or self.deviceList is None:
            raise Exception("No device available.Please pair with a Unicorn first.")

        print("Paied Unicorn devices:")
        i = 0
        for device in self.deviceList:
            print("#%i %s" % (i, device))
            i += 1

        print()
        if len(self.deviceList) == 1:
            self.deviceID = 0
        else:
            self.deviceID = int(input("Select device by ID #"))
            if self.deviceID < 0 or self.deviceID > len(self.deviceList):
                raise IndexError("The selected device ID is not valid.")

    def connect(self):
        """
        Connect to the Unicorn device.
        Does not start the session.
        """
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

    def start_session(self):
        """
        Continously listen to the Unicorn device.
        And listen to the triggers.
        """

        self.session_thread = threading.Thread(
            target=self.session, args=(), daemon=True
        )
        self.session_thread.start()
        self.triggers = []
        time.sleep(0.1)

    def session(self):
        """
        Session thread for recording the EEG data,
        during the experiment.
        Also listens to the triggers.
        And collect data in format:
        [(timestamp, trigger, data), ...]
        """
        self.session_buffer = []

        try:
            receiveBufferBufferLength = int(self.frame_length * self.total_channels * 4)
            receiveBuffer = bytearray(receiveBufferBufferLength)

            self.device.StartAcquisition(self.testsignale_enabled)
            self.stop_session = False

            while not self.stop_session:

                start_timestamp = time.time_ns()
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
                actual_data_received = EEG_data.shape[0]

                time_stamp_stream = self.get_time_stamp_stream(
                    start_timestamp, actual_data_received
                )

                trigger_stream = self.get_trigger_stream(time_stamp_stream)

                self.session_buffer.append(
                    np.vstack((time_stamp_stream, trigger_stream, EEG_data.T))
                )

        except UnicornPy.DeviceException as e:
            print(e)
            return -1
        except Exception as e:
            print("An unknown error occured. %s" % e)
            return -2

    def trigger(self, trigger):
        """
        Store a trigger in a queue.
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
            # trigger[0] = timestamp
            # trigger[1] = trigger value
            if trigger[0] > time_stamps[0] and trigger[0] < time_stamps[-1]:
                closest_timestamp = np.argmin(np.abs(time_stamps - trigger[0]))
                trigger_stream[closest_timestamp] = trigger[1]
            else:
                # TODO fix trigger delay!!!
                # print(trigger)
                # print(time_stamps[0], "--->", time_stamps[-1])
                #input("Trigger outside of time stamp range")
                if trigger[0] < time_stamps[0]:
                    trigger_stream[0] = trigger[1]
                else:
                    trigger_stream[-1] = trigger[1]
        self.triggers = []  # Clear trigger buffer
        return trigger_stream

    #@timeit()
    def get_session_data(self):
        """
        Get the complete data from the Arc device.
        """
        return np.hstack(self.session_buffer)

    def release(self):
        """
        Release the device.
        """
        try:
            self.device.StopAcquisition()
        except UnicornPy.DeviceException as e:
            ...
        try:
            del self.device
        except Exception as e:
            ...

    def stop(self):
        """
        Close all threads
        stop connection
        Release the device.
        """
        try:
            self.stop_buffer = True
            self.listener.join()
        except:
            ...
        try:
            self.stop_session = True
            self.session_thread.join()
        except:
            ...

        self.release()
