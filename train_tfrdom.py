import mne
import h5py
import pickle
import sys
import matplotlib
matplotlib.use('Qt5Agg')  # TODO

from datetime import datetime
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

from data_gen import *
from train_tdom import BrainDecodeClassification, GatherMetrics


def main(**kwargs):

    cfg = dict(
        subject='0717b399',
        data_ver='out_bl-1--0.05_tfr-multitaper-percent_reac-0.6_bad-95_f-2-40-100',

        # {'left': 0, 'right': 1},  #  {'left': 0, 'right': 1, 'left-right': 2, 'nothing': 3},
        events_to_cls={'left': 0, 'right': 1, 'left-right': 2, 'nothing': 3},
        eeg_chans=['C3', 'C4', 'Cz'],  # ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8'], ['C3', 'Cz', 'C4']
        crop_t=(-.2, None),
        rm_cz=True,
        erds_bands=None, #[(4, 7), (7, 13), (13, 40)],  # None | [(min_hz, max_hz), ...]
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
        n_fold=None,  # number of times to randomly re-split train and valid
        leave_k_out=1,  # number of sessions to use as validation in k-fold cross-valid
        reduce_lr_patience=3,
        early_stopping_patience=12,

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

    # manual data loading
    data = EEGTfrDomainDataset(streams_path, meta_path, cfg)

    # RGB spectrograms
    imgs = np.transpose(data.epochs, (0, 2, 3, 1))
    normed_imgs = (imgs - imgs.min(axis=0, keepdims=True)) / \
                  (imgs.max(axis=0, keepdims=True) - imgs.min(axis=0, keepdims=True))

    # # plots
    # cls_to_event = {v: k for k, v in cfg['events_to_cls'].items()}
    # _, axes = plt.subplots(2, 2)
    # axes = axes.reshape(-1)
    # for ev in np.unique(data.events_cls):
    #     m = normed_imgs[data.events_cls == ev].mean(axis=0)
    #     axes[ev].imshow(m)
    #     axes[ev].set_title(f'normed avg {cls_to_event[ev]}')
    # plt.tight_layout()
    # plt.show(block=True)
    #
    # for i, (img, ev) in enumerate(zip(imgs, data.events_cls)):
    #     plt.figure()
    #     img = (img - img.min()) / (img.max() - img.min())
    #     plt.imshow(img)
    #     plt.title(f'normed {cls_to_event[ev]}')
    #     plt.show()

    assert (cfg['n_fold'] is not None) ^ (cfg['leave_k_out'] is not None), 'define n_fold xor leave_k_out'
    ds_split_gen = rnd_by_epoch_cross_val if cfg['n_fold'] is not None else by_sess_cross_val
    print('split generator:', ds_split_gen)

    min_val_losses, max_val_accs = [], []
    # for split_i, (train_ds, valid_ds) in enumerate(ds_split_gen(data, cfg)):  # TODO
    split_i = 0
    train_ds, valid_ds = data.rnd_split_by_session(train_session_idx=np.arange(1, 13), valid_session_idx=np.arange(13, 15))
    if True:  # TODO !!! rm
        print('-' * 80, '\n', f'SPLIT #{split_i:03d}', '\n', '-' * 80)

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

        model = cfg['model_cls'](**model_params[cfg['model_cls'].__name__])

        # train
        classif = BrainDecodeClassification(model, cfg)
        model_name = f'tfr_braindecode_{model.__class__.__name__}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
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

        min_val_losses.append(min_val_loss)
        max_val_accs.append(max_acc)

    return dict(min_val_loss=np.mean(min_val_losses), max_acc=np.mean(max_val_accs),
                min_val_loss_std=np.std(min_val_losses), max_acc_std=np.std(max_val_accs))


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
        # try:
        if True:
            metrics = main(model_cls=model, batch_size=8)
            metricz[model.__name__] = metrics
            print('=' * 80, '\n', '=' * 80)
            print(model.__name__, '|', metrics)
            print('=' * 80, '\n', '=' * 80)
            pprint(metricz)
        # except Exception as e:
        #     print(e, file=sys.stderr)
        #     fails.append(model.__name__)

    model_names = list(metricz.keys())
    min_val_loss_i = np.argsort([m['min_val_loss'] for m in metricz.values()])[0]
    max_acc_i = np.argsort([m['max_acc'] for m in metricz.values()])[-1]
    print('best val loss:', model_names[min_val_loss_i], metricz[model_names[min_val_loss_i]])
    print('best val acc:', model_names[max_acc_i], metricz[model_names[max_acc_i]])
    print('fails:', fails, file=sys.stderr)
