import mne
import h5py
import pickle
import sys

from functools import partial
import torch
import numpy as np
from braindecode.datasets import create_from_mne_raw, create_from_mne_epochs, create_from_X_y
from torch.utils.data import DataLoader, Dataset, random_split
from braindecode import EEGClassifier
from braindecode.models import *
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from braindecode.preprocessing import exponential_moving_standardize, preprocess, Preprocessor
from copy import deepcopy
import matplotlib.pyplot as plt
from skimage.transform import resize

import lightning as L
from torch.nn import Linear, CrossEntropyLoss, NLLLoss
from lightning.pytorch.callbacks import Callback
from torch.utils.data.dataset import T_co
from torchvision.transforms import Normalize, Compose, ToTensor
import torchmetrics
import random
import wandb
from pprint import pprint

from braindecode_train_tdom import BrainDecodeClassification, GatherMetrics


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
        session_ids = streams_data.attrs['session_ids'][:]

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

    def __getitem__(self, index) -> T_co:
        return self.epochs[index], torch.as_tensor(self.events_cls[index], dtype=torch.int64)

    def __len__(self):
        return len(self.epochs)


def main(**kwargs):

    cfg = dict(
        subject='0717b399',
        data_ver='out_bl-1--0.05_tfr-multitaper-percent_reac-0.5_bad-95_f-2-80-100',

        # {'left': 0, 'right': 1},  #  {'left': 0, 'right': 1, 'left-right': 2, 'nothing': 3},
        events_to_cls={'left': 0, 'right': 1, 'left-right': 2, 'nothing': 3},
        eeg_chans=['C3', 'C4', 'Cz'],  # ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8'], ['C3', 'Cz', 'C4']
        crop_t=(-.2, None),
        rm_cz=True,
        erds_bands=[(4, 7), (7, 13), (13, 40)],  # None | [(min_hz, max_hz), ...]
        merge_bands_into_chans=True,
        resize_t=None,  # resize time dim
        resize_f=None,  # resize freq dim

        batch_size=8,
        num_workers=0,
        prefetch_factor=2,
        accumulate_grad_batches=1,
        precision=32,
        gradient_clip_val=1,
        loss_fun=NLLLoss,

        model_cls=ShallowFBCSPNet,
        # input_window_samples=100,   # TODO set to epoch len now
        final_conv_length='auto',

        dev='cuda',
        ndev=1,
        multi_dev_strat=None,

        epochs=100,
        init_lr=1e-3,
        train_data_ratio=.85,
    )
    cfg.update(kwargs)
    pprint(cfg)

    # wandb.init(project='eeg-motor-execution', config=cfg)
    mne.set_log_level(False)

    streams_path = f'{cfg["data_ver"]}/{cfg["subject"]}/{cfg["subject"]}_streams.h5'
    meta_path = f'{cfg["data_ver"]}/{cfg["subject"]}/{cfg["subject"]}_meta.pckl'

    # manual data loading  # TODO by-session splitting
    gen = torch.Generator()
    gen.manual_seed(42)
    data = EEGTfrDomainDataset(streams_path, meta_path, cfg)
    train_ds, valid_ds = random_split(data, [cfg['train_data_ratio'], 1 - cfg['train_data_ratio']], generator=gen)

    # init dataloaders
    dl_params = dict(num_workers=cfg['num_workers'], prefetch_factor=cfg['prefetch_factor'],
                     persistent_workers=cfg['num_workers'] > 0, pin_memory=True)

    train_dl = DataLoader(train_ds, cfg['batch_size'], shuffle=True, **dl_params)
    valid_dl = DataLoader(valid_ds, cfg['batch_size'], shuffle=False, **dl_params)

    # init model
    n_classes = len(np.unique(list(cfg['events_to_cls'].values())))

    # https://braindecode.org/stable/api.html#models
    in_chans = len(cfg['eeg_chans']) * len(cfg['erds_bands']) \
        if cfg['erds_bands'] is not None and cfg['merge_bands_into_chans'] else len(cfg['eeg_chans'])
    iws = data[0][0].shape[1]
    model_params = dict(  # what a marvelously fucked up library
        ShallowFBCSPNet=dict(in_chans=in_chans, n_classes=n_classes, input_window_samples=iws, final_conv_length=cfg['final_conv_length']),
        Deep4Net=dict(in_chans=in_chans, n_classes=n_classes, input_window_samples=iws, final_conv_length=cfg['final_conv_length']),
        EEGInception=dict(in_channels=in_chans, n_classes=n_classes, input_window_samples=iws, sfreq=250),
        EEGITNet=dict(in_channels=in_chans, n_classes=n_classes, input_window_samples=iws),
        EEGNetv1=dict(in_chans=in_chans, n_classes=n_classes, input_window_samples=iws, final_conv_length=cfg['final_conv_length']),
        EEGNetv4=dict(in_chans=in_chans, n_classes=n_classes, input_window_samples=iws, final_conv_length=cfg['final_conv_length']),
        HybridNet=dict(in_chans=in_chans, n_classes=n_classes, input_window_samples=iws),
        EEGResNet=dict(in_chans=in_chans, n_classes=n_classes, input_window_samples=iws, n_first_filters=16, final_pool_length=8),
        TIDNet=dict(in_chans=in_chans, n_classes=n_classes, input_window_samples=iws),
    )

    model = cfg['model_cls'](
        **model_params[cfg['model_cls'].__name__],
    )

    # train
    classif = BrainDecodeClassification(model, cfg)
    model_name = f'tfr_braindecode_{model.__class__.__name__}'
    model_fname_template = "{epoch}_{step}_{val_loss:.2f}"

    gather_metrics = GatherMetrics()
    callbacks = [
        ModelCheckpoint(
            f"models/{model_name}",
            model_fname_template,
            monitor="val_loss",
            save_top_k=1,
            save_last=False,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping("val_loss", patience=12),
        gather_metrics,
    ]

    trainer = L.Trainer(
            accelerator=cfg["dev"],
            devices=cfg["ndev"],
            strategy=cfg["multi_dev_strat"],
            max_epochs=cfg["epochs"],
            default_root_dir=f"models/{model_name}",
            callbacks=callbacks,
            benchmark=False,
            accumulate_grad_batches=cfg["accumulate_grad_batches"],
            precision=cfg["precision"],
            gradient_clip_val=cfg["gradient_clip_val"],
        )

    trainer.fit(classif, train_dl, valid_dl)

    min_val_loss = min([m['val_loss'].item() for m in gather_metrics.metrics])
    max_acc = max([m['val_acc'].item() for m in gather_metrics.metrics])
    return dict(min_val_loss=min_val_loss, max_acc=max_acc)


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)

    models_to_try = [
        ShallowFBCSPNet,
        Deep4Net,
        EEGInception,
        EEGITNet,
        EEGNetv1,
        EEGNetv4,
        HybridNet,
        EEGResNet,
        TIDNet
    ]

    # left out: TCN, SleepStager..., USleep, TIDNet
    metricz = {}
    fails = []
    for model in models_to_try:
        try:
            metrics = main(model_cls=model, batch_size=8)
            metricz[model.__name__] = metrics
            print('=' * 80)
            print('=' * 80)
            print(model.__name__, '|', metrics)
            print('=' * 80)
            print('=' * 80)
            pprint(metricz)
        except Exception as e:
            print(e, file=sys.stderr)
            fails.append(model.__name__)

    model_names = list(metricz.keys())
    min_val_loss_i = np.argsort([m['min_val_loss'] for m in metricz.values()])[0]
    max_acc_i = np.argsort([m['max_acc'] for m in metricz.values()])[-1]
    print('best val loss:', model_names[min_val_loss_i], metricz[model_names[min_val_loss_i]])
    print('best val acc:', model_names[max_acc_i], metricz[model_names[max_acc_i]])
    print('fails:', fails, file=sys.stderr)
