import glob
import os
import time

import torch
from pynput import keyboard
import numpy as np
from collections import deque
from train_tdom import load_model

import socket
import sys
from pynput import keyboard

try:
    import UnicornPy
except Exception as e:
    UnicornPy = None
    print(e, file=sys.stderr)


class TCPSender:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((ip, port))
        print("Connected to Unity...")

    def send_msg(self, msg):
        self.sock.sendall(msg.encode('utf-8'))


class KeyListener:
    def __init__(self, sender):
        self.sender = sender

    def on_press(self, key):
        try:
            if key.char == 'w':
                self.sender.send_msg('up')
            elif key.char == 'a':
                self.sender.send_msg('left')
            elif key.char == 's':
                self.sender.send_msg('down')
            elif key.char == 'd':
                self.sender.send_msg('right')
        except AttributeError:
            pass

    def on_release(self, key):
        if key == keyboard.Key.esc:
            # Stop listener
            return False


class CircBuff:
    def __init__(self, size, dim, dtype=np.float32) -> None:
        self.size = size
        self.dim = dim
        self.buffer = np.zeros((size, dim), dtype=dtype)
        self.tail = self.count = 0

    def add(self, data: np.ndarray):
        dsize = data.shape[0]

        start = self.tail
        end = (self.tail + dsize) % self.size
        if start < end:
            self.buffer[start:end] = data
        else:
            self.buffer[start:] = data[:self.size - start]
            self.buffer[:end] = data[self.size - start:]

        self.count = min(self.size, self.count + dsize)
        self.tail = end

    def get(self):
        if self.count < self.size:
            return self.buffer[:self.count]
        return np.concatenate([self.buffer[self.tail:], self.buffer[:self.tail]])


if __name__ == "__main__":

    # connect to unicorn
    deviceID = 0
    sfreq = 250
    FrameLength = sfreq // 5
    TestsignaleEnabled = False
    eeg_hist_len = 880
    model_name = 'braindecode_ShallowFBCSPNet_2023-07-18_15-49-57'

    numberOfAcquiredChannels = 17
    receiveBufferBufferLength = 0
    receiveBuffer = None

    try:
        deviceList = UnicornPy.GetAvailableDevices(True)

        if len(deviceList) <= 0 or deviceList is None:
            raise Exception("No device available.Please pair with a Unicorn first.")

        device = UnicornPy.Unicorn(deviceList[deviceID])
        numberOfAcquiredChannels = device.GetNumberOfAcquiredChannels()
        configuration = device.GetConfiguration()

        receiveBufferBufferLength = FrameLength * numberOfAcquiredChannels * 4
        receiveBuffer = bytearray(receiveBufferBufferLength)
        print('CONNECTED TO UNICORN')

    except:
        device = None
        print('COULD NOT CONNECT TO DEVICE, PROCEEDING TO USE THE KEYBOARD')

    # TCP Server configuration
    ip = "127.0.0.1"
    port = 12345

    sender = TCPSender(ip, port)
    listener = KeyListener(sender)

    # Start the key listener
    # if device is None:
        # with keyboard.Listener(on_press=listener.on_press, on_release=listener.on_release) as keyboard_listener:
        #     keyboard_listener.start()

    # load prediction model
    ckpt_fname = os.path.basename(sorted(glob.glob(f'../models/{model_name}/*.ckpt'))[-1])
    model = load_model(f'../models/{model_name}/{ckpt_fname}').to('cuda')

    # setup buffer
    eeg_buffer = CircBuff(eeg_hist_len, numberOfAcquiredChannels)
    device.StartAcquisition(TestsignaleEnabled)

    try:
        while True:
            if device is not None:
                device.GetData(FrameLength, receiveBuffer, receiveBufferBufferLength)
                data = np.frombuffer(receiveBuffer, dtype=np.float32, count=numberOfAcquiredChannels * FrameLength)
                data = np.reshape(data, (FrameLength, numberOfAcquiredChannels))
            else:
                data = np.random.random((numberOfAcquiredChannels, FrameLength))
                time.sleep(FrameLength / sfreq)
            eeg_buffer.add(data)

            print(eeg_buffer.count)

            if eeg_buffer.count == eeg_buffer.size:
                x = torch.as_tensor(eeg_buffer.get(), device='cuda')[None, ...]
                y = model.infer(x)
                print('pred:', y)

    finally:
        device.StopAcquisition()
        del receiveBuffer
        del device
        print("Data acquisition stopped.")
