import sys

import mne
import h5py
import pickle

from functools import partial
import torch
import numpy as np
from braindecode.datasets import create_from_mne_raw, create_from_mne_epochs, create_from_X_y
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from braindecode import EEGClassifier
from braindecode.models import *
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from braindecode.preprocessing import exponential_moving_standardize, preprocess, Preprocessor
from copy import deepcopy

import lightning as L
from torch.nn import Linear, CrossEntropyLoss, NLLLoss
from lightning.pytorch.callbacks import Callback
from torch.utils.data.dataset import T_co
from torchvision.transforms import Normalize, Compose, ToTensor
import torchmetrics
import random
import wandb
from pprint import pprint
from itertools import combinations
from typing import Union
from skimage.transform import resize


class EEGTimeDomainDataset(Dataset):  # TODO generalize to epoch types: on_task, on_break, on_pull

    def __init__(self, streams_path, meta_path, cfg):
        streams_data = h5py.File(streams_path, 'r')

        with open(meta_path, 'rb') as f:
            meta_data = pickle.load(f)

        crop_def = (-np.inf, np.inf)
        crop_t = cfg['crop_t']
        crop_t = (crop_def[0] if crop_t[0] is None else crop_t[0], crop_def[1] if crop_t[1] is None else crop_t[1])
        times = streams_data.attrs['on_task_times'][:]
        within_window = (crop_t[0] < times) & (times < crop_t[1])
        on_task_events = streams_data['on_task_events'][:][:, 2]

        eeg_info = meta_data['eeg_info']
        task_event_ids = meta_data['task_event_ids']

        event_id_to_cls = {task_event_ids[ev]: cls for ev, cls in cfg['events_to_cls'].items()}

        # select task relevant epochs
        epochs = streams_data['epochs_on_task'][:]
        relevant_epochs = np.logical_or.reduce([on_task_events == task_event_ids[ev]
                                                for ev in cfg['events_to_cls'].keys()])
        epochs = epochs[relevant_epochs, ...]
        events = on_task_events[relevant_epochs]

        # pick channels
        relevant_chans_i = [eeg_info['ch_names'].index(chan) for chan in cfg['eeg_chans']]
        epochs = epochs[:, relevant_chans_i]

        # load as X, y into braindecode
        mapper = np.vectorize(lambda x: event_id_to_cls[x])
        events_cls = mapper(events)

        assert len(epochs) == len(events_cls)
        self.epochs = epochs[..., within_window]
        self.events_cls = events_cls
        # self.norm = Compose([ToTensor(), Normalize([.5] * epochs.shape[1], [.5] * epochs.shape[1])])

        # standardize by epoch
        means = self.epochs.mean(axis=-1, keepdims=True)
        stds = self.epochs.std(axis=-1, keepdims=True)
        self.epochs = (self.epochs - means) / stds

        # splitting info
        self.session_idx = streams_data['on_task_session_idx'][relevant_epochs]
        self.sessions = meta_data['session_ids']

        streams_data.close()

    def __getitem__(self, index) -> T_co:
        return self.epochs[index], torch.as_tensor(self.events_cls[index], dtype=torch.int64)

    def __len__(self):
        return len(self.epochs)


class EEGTfrDomainDataset(Dataset):

    def __init__(self, streams_path, meta_path, cfg):
        streams_data = h5py.File(streams_path, 'r')

        with open(meta_path, 'rb') as f:
            meta_data = pickle.load(f)

        crop_def = (-np.inf, np.inf)
        crop_t = cfg['crop_t']
        crop_t = (crop_def[0] if crop_t[0] is None else crop_t[0], crop_def[1] if crop_t[1] is None else crop_t[1])

        freqs = streams_data.attrs['freqs']
        times = streams_data.attrs['on_task_times'][:]
        on_task_events = streams_data['on_task_events'][:][:, 2]

        eeg_info = meta_data['eeg_info']
        event_dict = meta_data['event_dict']
        task_event_ids = meta_data['task_event_ids']

        event_id_to_cls = {task_event_ids[ev]: cls for ev, cls in cfg['events_to_cls'].items()}

        # select task relevant epochs  # TODO move this into dataset init
        epochs = streams_data['tfr_epochs_on_task'][:]
        relevant_epochs = np.logical_or.reduce([on_task_events == task_event_ids[ev]
                                                for ev in cfg['events_to_cls'].keys()])
        epochs = epochs[relevant_epochs, ...]
        events = on_task_events[relevant_epochs]

        # remove cz from c3 and c4
        if cfg['rm_cz']:
            cz_i = eeg_info['ch_names'].index('Cz')
            c3_i = eeg_info['ch_names'].index('C3')
            c4_i = eeg_info['ch_names'].index('C4')
            epochs[:, c3_i] = 2 * epochs[:, c3_i] - epochs[:, cz_i]
            epochs[:, c4_i] = 2 * epochs[:, c4_i] - epochs[:, cz_i]

        # pick channels
        relevant_chans_i = [eeg_info['ch_names'].index(chan) for chan in cfg['eeg_chans']]
        epochs = epochs[:, relevant_chans_i]

        # load as X, y into braindecode
        mapper = np.vectorize(lambda x: event_id_to_cls[x])
        events_cls = mapper(events)

        # crop time window
        within_window = (crop_t[0] <= times) & (times <= crop_t[1])
        epochs = epochs[..., within_window]

        # resample
        if cfg['resize_t'] or cfg['resize_f']:
            res_t = cfg['resize_t'] or epochs.shape[-1]
            res_f = cfg['resize_f'] or epochs.shape[-2]
            times = resize(times, (res_t,), preserve_range=True, clip=True)
            freqs = resize(freqs, (res_f,), preserve_range=True, clip=True)
            epochs = resize(epochs, (*epochs.shape[:2], res_f, res_t), preserve_range=True, clip=True)

        # compute erds - contracts the freq channel
        if cfg['erds_bands'] is not None:
            all_band_epochs = []
            for erds_band in cfg['erds_bands']:
                within_freq_window = (erds_band[0] <= freqs) & (freqs <= erds_band[1])
                band_epochs = epochs[:, :, within_freq_window, :].mean(axis=2, keepdims=True)
                all_band_epochs.append(band_epochs)
            epochs = np.concatenate(all_band_epochs, axis=2)
            if cfg['merge_bands_into_chans']:  # bands dim to chan dim
                epochs = epochs.reshape((epochs.shape[0], epochs.shape[1] * len(cfg['erds_bands']), epochs.shape[-1]))

        # debug plots:
        # i=1; ev=events_cls[i]; plt.plot(epochs[i][0], ['--', '-', '--', '--'][ev], label='C3'); plt.plot(epochs[i][1], ['-', '--', '--', '--'][ev], label='C4'); plt.legend(); plt.title(ev); plt.show()

        # alpha left/right/c3/c4; channel order: c3, c4, cz; merge_bands_into_chans=False
        # plt.plot(epochs[events_cls==1, 0, 1, :].mean(axis=0), label='C3-alpha-right'); plt.plot(epochs[events_cls==0, 1, 1, :].mean(axis=0), label='C4-alpha-left');
        # plt.plot(epochs[events_cls==0, 0, 1, :].mean(axis=0), '--', label='C3-alpha-left'); plt.plot(epochs[events_cls==1, 1, 1, :].mean(axis=0), '--', label='C4-alpha-right');
        # plt.legend(); plt.show()


        # by channel (across trials and time) standardization TODO validation normalization leaks into training
        std_over = (0, 2, 3) if cfg['erds_bands'] is None else (0, 2)
        means = epochs.mean(axis=std_over, keepdims=True)
        stds = epochs.std(axis=std_over, keepdims=True)  # self.epochs.std(axis=-1, keepdims=True)
        epochs = (epochs - means) / stds  # normalized already

        assert len(epochs) == len(events_cls)
        self.epochs = epochs
        self.events_cls = events_cls

        # splitting info
        self.session_idx = streams_data['on_task_session_idx'][relevant_epochs]
        self.sessions = meta_data['session_ids']

        streams_data.close()

    def __getitem__(self, index) -> T_co:
        return self.epochs[index], torch.as_tensor(self.events_cls[index], dtype=torch.int64)

    def __len__(self):
        return len(self.epochs)


def rnd_by_epoch_cross_val(data: Dataset, cfg: dict):
    for _ in range(cfg['n_fold']):
        yield random_split(data, [cfg['train_data_ratio'], 1 - cfg['train_data_ratio']])


def by_sess_cross_val(data: Union[EEGTimeDomainDataset, EEGTfrDomainDataset], cfg: dict):
    for val_sessions in map(set, combinations(data.sessions, cfg['k_fold'])):
        yield split_by_session(data, data.session_idx, val_sessions)


def split_by_session(data: Dataset, session_idx, val_sessions):
    val = np.logical_or.reduce([session_idx == vs for vs in val_sessions], axis=0)
    train = ~val
    return Subset(data, np.where(train)[0]), Subset(data, np.where(val)[0])
