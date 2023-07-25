import glob
import sys
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

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
from torch.nn import Linear, CrossEntropyLoss, NLLLoss, MSELoss
from lightning.pytorch.callbacks import Callback
from torch.utils.data.dataset import T_co
from torchvision.transforms import Normalize, Compose, ToTensor
import torchmetrics
import random
#import wandb
from pprint import pprint
from itertools import combinations
from datetime import datetime
import matplotlib.pyplot as plt

from data_gen import *
from src.models.AdamNet import AdamNet, EEGNet
from train_tdom import BrainDecodeClassification, GatherMetrics, load_model

# TODO training: raw eeg -> rnd slice -> TDomPrepper(common_mean, filt, crop, zscore) -> train
#                filt eeg, zscore -> rnd slice -> TDomPrepper(NOTHING) -> train


def main(init_model=None, fine_tune_subject=None, **kwargs):

    cfg = dict(
        subjects=[],
        # subjects,  # ['1cfd7bfa', '4bc2006e', '4e7cac2d', '8c70c0d3', '0717b399'],
        # data_ver='out_bl-1--0.05_tfr-multitaper-percent_reac-0.5_bad-95_c34-True',  # 2-50 Hz
        data_ver='out_bl-1--0.05_tfr-multitaper-percent_reac-0.7_bad-95_f-2-40-100',
        is_momentart=True,

        # {'left': 0, 'right': 1},  #  {'left': 0, 'right': 1, 'left-right': 2, 'nothing': 3},
        # eeg channels to use ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8'], ['C3', 'C4', 'Cz']
        eeg_chans=['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8'],
        crop_t=(-.2, None),  # the part of the epoch to include
        resample=1.,  # no resample = 1.  # TODO

        batch_size=32,
        num_workers=0,  # dataloader can't pickle pointer to h5
        prefetch_factor=2,
        accumulate_grad_batches=1,
        precision=32,  # 16 | 32
        gradient_clip_val=5,
        loss_fun=MSELoss,
        # number of times to randomly re-split train and valid, metrics are averaged across splits
        n_fold=4,
        # number of sessions to use as validation in k-fold cross-valid - all combinations are computed
        leave_k_out=None,
        reduce_lr_patience=3,
        early_stopping_patience=12,

        model_cls=AdamNet,  # default model

        dev='cuda',
        ndev=1,
        multi_dev_strat=None,

        epochs=100,
        init_lr=5e-3,
        train_data_ratio=.85,  # ratio of training data, the rest is validation
    )
    cfg.update(kwargs)
    pprint(cfg)

    model_params = dict(AdamNet={'out_dim': 2}, EEGNet={'num_classes': 2, 'channels': 8, 'samples': 550})

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # wandb.init(project='eeg-motor-execution', config=cfg)
    mne.set_log_level(False)
    ds_settings = dict(epoch_len=550, ret_likeliest_gamepad=.3, add_chan_to_eeg=True, is_trial_chance=.7)

    if len(cfg['subjects']) == 1:  # fine-tune
        subject = cfg['subjects'][0]
        assert subject == '0717b399'  # TODO session slicing was implemented for this subject
        streams_path = f'{cfg["data_ver"]}/{subject}/{subject}_streams.h5'
        meta_path = f'{cfg["data_ver"]}/{subject}/{subject}_meta.pckl'
        train_ds = MomentaryEEGTimeDomainDataset(streams_path, meta_path, cfg, session_slice=slice(0, 10), **ds_settings)
        valid_ds = MomentaryEEGTimeDomainDataset(streams_path, meta_path, cfg, session_slice=slice(10, 12), **ds_settings)

    else:  # pretrain
        assert fine_tune_subject is not None
        train_datasets = []
        for subject in cfg['subjects']:
            if fine_tune_subject == subject:
                continue  # fine-tune subject as validation
            streams_path = f'{cfg["data_ver"]}/{subject}/{subject}_streams.h5'
            meta_path = f'{cfg["data_ver"]}/{subject}/{subject}_meta.pckl'
            ds = MomentaryEEGTimeDomainDataset(streams_path, meta_path, cfg, **ds_settings)
            train_datasets.append(ds)
        train_ds = CombinedIterableDataset(train_datasets)

        streams_path = f'{cfg["data_ver"]}/{fine_tune_subject}/{fine_tune_subject}_streams.h5'
        meta_path = f'{cfg["data_ver"]}/{fine_tune_subject}/{fine_tune_subject}_meta.pckl'
        valid_ds = MomentaryEEGTimeDomainDataset(streams_path, meta_path, cfg, session_slice=slice(0, 12), **ds_settings)

    # check dataset balance
    # balance = [d[1].max() for d in train_ds]
    # print('label balance:', np.mean(balance))

    assert (cfg['n_fold'] is not None) ^ (cfg['leave_k_out'] is not None), 'define n_fold xor leave_k_out'

    # init dataloaders
    dl_params = dict(num_workers=cfg['num_workers'], prefetch_factor=cfg['prefetch_factor'],
                     persistent_workers=cfg['num_workers'] > 0, pin_memory=True)

    train_dl = DataLoader(train_ds, cfg['batch_size'], shuffle=False, **dl_params)
    valid_dl = DataLoader(valid_ds, cfg['batch_size'], shuffle=False, **dl_params)

    model = cfg['model_cls'](**model_params[cfg['model_cls'].__name__]) if init_model is None else init_model

    # train
    classif = BrainDecodeClassification(model, cfg)
    model_name = f'momentary_{model.__class__.__name__}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    model_fname_template = "{epoch}_{step}_{val_loss:.2f}"

    os.makedirs(f'models/{model_name}', exist_ok=True)
    with open(f'models/{model_name}/cfg.pkl', 'wb') as f:
        pickle.dump({'cfg': cfg, 'model_params': model_params}, f)

    gather_metrics = GatherMetrics()
    callbacks = [
        ModelCheckpoint(
            f'models/{model_name}',
            model_fname_template,
            monitor='val_loss',
            save_top_k=1,
            save_last=False,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval='step'),
        EarlyStopping('val_loss', patience=cfg['early_stopping_patience']),
        gather_metrics,
    ]

    trainer = L.Trainer(
            accelerator=cfg["dev"],
            devices=cfg["ndev"],
            #strategy=cfg["multi_dev_strat"],
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
    max_val_acc = max([m['val_acc'].item() for m in gather_metrics.metrics])

    return model, dict(min_val_loss=min_val_loss, max_val_acc=max_val_acc)


if __name__ == '__main__':
    torch.use_deterministic_algorithms(False)

    models_to_try = [
        AdamNet,
        EEGNet,
    ]

    fine_tune_subject = '0717b399'
    pretrain_subjects = os.listdir('c:\\wut\\asura\\Motor-Execution-Classifiaction\\out_bl-1--0.05_tfr-multitaper-percent_reac-0.7_bad-95_f-2-40-100\\')

    metricz = {}
    for model_cls in models_to_try:
        model, metrics = main(model_cls=model_cls, subjects=pretrain_subjects, fine_tune_subject=fine_tune_subject)
        metricz[model_cls.__name__] = metrics
        print('=' * 80, '\n', '=' * 80)
        print('PRETRAIN')
        print(model_cls.__name__, '|', metrics)
        print('=' * 80, '\n', '=' * 80)
        pprint(metricz)

        model, metrics = main(model_cls=model_cls, subjects=[fine_tune_subject], fine_tune_subject=fine_tune_subject,
                              init_model=model)
        metricz[model_cls.__name__] = metrics
        print('=' * 80, '\n', '=' * 80)
        print('FINETUNE')
        print(model_cls.__name__, '|', metrics)
        print('=' * 80, '\n', '=' * 80)
        pprint(metricz)

    model_names = list(metricz.keys())
    min_val_loss_i = np.argsort([m['min_val_loss'] for m in metricz.values()])[0]
    max_acc_i = np.argsort([m['max_acc'] for m in metricz.values()])[-1]
    print('best val loss:', model_names[min_val_loss_i], metricz[model_names[min_val_loss_i]])
    print('best val acc:', model_names[max_acc_i], metricz[model_names[max_acc_i]])
