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
from common import CircBuff
from eeg_analysis import TDomPrepper

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


if __name__ == "__main__":

    # connect to unicorn
    deviceID = 0
    sfreq = 250
    FrameLength = sfreq // 2
    TestsignaleEnabled = False
    model_name = 'braindecode_ShallowFBCSPNet_2023-07-18_18-47-49'  # 'braindecode_EEGResNet_2023-07-19_14-04-36'  # 'braindecode_ShallowFBCSPNet_2023-07-18_18-47-49'
    dev = 'cuda'
    pred_threshold = .7

    eeg_hist_len = 876  # take this from the dataset epoch length
    baseline = (-1., -.05)  # dataset property
    bandpass_freq = (.5, 80)  # dataset property
    notch_freq = (50, 100)  # dataset property
    filter_percentile = 95  # dataset property
    tmin_max = (-.5, 2.)  # epoch specific, check eeg_analysis or the experiment config
    crop_t = (-.2, None)  # should be same as in training script

    # TODO keverd bele open meg closed eye recordingokat trainingsetbe mint nothing

    # TODO preprocessing

    numberOfAcquiredChannels = 17
    receiveBufferBufferLength = 0
    receiveBuffer = None

    eeg_ch_names = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']  # unicorn

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
    #     with keyboard.Listener(on_press=listener.on_press, on_release=listener.on_release) as keyboard_listener:
    #         print('use keyboard')
    #         keyboard_listener.join()

    # load prediction model
    # ckpt_fname = os.path.basename(sorted(glob.glob(f'../models/{model_name}/*.ckpt'))[-1])
    model, cfg = load_model(f'../models/{model_name}', dev)
    chans_i = [eeg_ch_names.index(chan) for chan in cfg['eeg_chans']]

    cls_to_events = {v: k for k, v in cfg['events_to_cls'].items()}
    events_to_msgs = {'left': 'left', 'right': 'right', 'left-right': 'up', 'nothing': 'stay'}

    # setup epoch preprocessing
    crop_len = 550
    prep = TDomPrepper(eeg_hist_len, crop_len, sfreq, cfg['eeg_chans'], bandpass_freq, notch_freq, common=np.mean,
                       tmin_max=tmin_max, crop_t=crop_t, baseline=baseline, filter_percentile=filter_percentile)
    # TODO !!!!!!!!!!!!!!!! TEST IS THIS IN ACTION

    # setup buffer
    eeg_buffer = CircBuff(eeg_hist_len, numberOfAcquiredChannels)
    if device is not None:
        device.StartAcquisition(TestsignaleEnabled)

    try:
        while True:
            if device is not None:
                device.GetData(FrameLength, receiveBuffer, receiveBufferBufferLength)
                data = np.frombuffer(receiveBuffer, dtype=np.float32, count=numberOfAcquiredChannels * FrameLength)
                data = np.reshape(data, (FrameLength, numberOfAcquiredChannels))
            else:
                data = np.random.random((FrameLength, numberOfAcquiredChannels))
                time.sleep(FrameLength / sfreq)
            eeg_buffer.add(data)

            if eeg_buffer.count == eeg_buffer.size:
                eeg = eeg_buffer.get()[..., chans_i]
                epoch = prep(eeg)

                x = torch.as_tensor(epoch, device=dev)[None, ...]
                y = model.infer(x.permute(0, 2, 1))[0]

                y = np.e ** y  # log prob to prob
                highest = np.argmax(y)
                # print(y)
                if y[highest] > pred_threshold:
                    msg = events_to_msgs[cls_to_events[highest]]
                    sender.send_msg(msg)
                    print(f'{y}; sent: {msg}')
            else:
                print(eeg_buffer.count)

    finally:
        if device is not None:
            device.StopAcquisition()
        del receiveBuffer
        del device
        print("Data acquisition stopped.")
