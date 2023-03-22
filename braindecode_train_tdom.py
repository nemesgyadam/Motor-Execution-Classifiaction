import mne
import h5py
import pickle

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

import lightning as L
from torch.nn import Linear, CrossEntropyLoss, NLLLoss
from lightning.pytorch.callbacks import Callback
from torch.utils.data.dataset import T_co
from torchvision.transforms import Normalize, Compose, ToTensor
import torchmetrics
import random
import wandb


class GatherMetrics(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        each_me = deepcopy(trainer.callback_metrics)
        self.metrics.append(each_me)


class EEGTimeDomainDataset(Dataset):

    def __init__(self, epochs, events_cls):
        assert len(epochs) == len(events_cls)
        self.epochs = epochs
        self.events_cls = events_cls
        # self.norm = Compose([ToTensor(), Normalize([.5] * epochs.shape[1], [.5] * epochs.shape[1])])

        means = self.epochs.mean(axis=-1, keepdims=True)
        stds = self.epochs.std(axis=-1, keepdims=True)
        self.epochs = (self.epochs - means) / stds  # normalized already

    def __getitem__(self, index) -> T_co:
        return self.epochs[index], torch.as_tensor(self.events_cls[index], dtype=torch.int64)

    def __len__(self):
        return len(self.epochs)


class BrainDecodeClassification(L.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.loss_fun = cfg['loss_fun']()
        self.accuracy = torchmetrics.Accuracy('multiclass', num_classes=len(cfg['events_to_cls']))

        self.model.requires_grad_(True)
        print(self.model)

    def training_step(self, batch, batch_idx):
        x, y = batch
        yy = self.model(x)

        # window is smaller than the epoch
        if len(yy.shape) == 3:
            yy = yy.mean(dim=-1)

        loss = self.loss_fun(yy, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yy = self.model(x)
        if len(yy.shape) == 3:
            yy = yy.mean(dim=-1)

        loss = self.loss_fun(yy, y)
        acc = self.accuracy(yy, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg['init_lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, verbose=True, factor=0.2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


def main(**kwargs):
    # load streams
    cfg = dict(
        subject='0717b399',
        data_ver='out_bl-1--0.05_tfr-multitaper-percent_reac-0.5_bad-95_c34-True',

        # {'left': 0, 'right': 1},  #  {'left': 0, 'right': 1, 'left-right': 2, 'nothing': 3},
        events_to_cls={'left': 0, 'right': 1, 'left-right': 2, 'nothing': 3},
        eeg_chans=['C3', 'C4', 'Cz'],  # ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
        prep_std_params=dict(factor_new=1e-3, init_block_size=500),

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
    print(cfg)

    # wandb.init(project='eeg-motor-execution', config=cfg)
    mne.set_log_level(False)

    streams_path = f'{cfg["data_ver"]}/{cfg["subject"]}/{cfg["subject"]}_streams.h5'
    meta_path = f'{cfg["data_ver"]}/{cfg["subject"]}/{cfg["subject"]}_meta.pckl'
    streams_data = h5py.File(streams_path, 'r')

    with open(meta_path, 'rb') as f:
        meta_data = pickle.load(f)

    freqs = streams_data.attrs['freqs']
    times = streams_data.attrs['on_task_times'][:]
    on_task_events = streams_data['on_task_events'][:][:, 2]
    session_ids = streams_data.attrs['session_ids'][:]

    eeg_info = meta_data['eeg_info']
    event_dict = meta_data['event_dict']
    task_event_ids = meta_data['task_event_ids']

    event_id_to_cls = {task_event_ids[ev]: cls for ev, cls in cfg['events_to_cls'].items()}

    # select task relevant epochs  # TODO move this into dataset init
    epochs = streams_data['epochs_on_task'][:]
    relevant_epochs = np.logical_or.reduce([on_task_events == task_event_ids[ev]
                                            for ev in cfg['events_to_cls'].keys()])
    epochs = epochs[relevant_epochs, ...]
    events = on_task_events[relevant_epochs]

    # pick channels
    relevant_chans_i = [eeg_info['ch_names'].index(chan) for chan in cfg['eeg_chans']]
    epochs = epochs[:, relevant_chans_i, :]

    # load as X, y into braindecode
    mapper = np.vectorize(lambda x: event_id_to_cls[x])
    events_cls = mapper(events)

    # manual data loading  # TODO by-session splitting
    data = EEGTimeDomainDataset(epochs, events_cls)
    train_ds, valid_ds = random_split(data, [cfg['train_data_ratio'], 1 - cfg['train_data_ratio']])

    # init dataloaders
    dl_params = dict(num_workers=cfg['num_workers'], prefetch_factor=cfg['prefetch_factor'],
                     persistent_workers=cfg['num_workers'] > 0, pin_memory=True)

    train_dl = DataLoader(train_ds, cfg['batch_size'], shuffle=True, **dl_params)
    valid_dl = DataLoader(valid_ds, cfg['batch_size'], shuffle=False, **dl_params)

    # init model
    n_classes = len(np.unique(list(cfg['events_to_cls'].values())))

    iws = data[0][0].shape[1]
    model_params = dict(  # what a marvelously fucked up library
        ShallowFBCSPNet=dict(in_chans=len(cfg['eeg_chans']), n_classes=n_classes, input_window_samples=iws, final_conv_length=cfg['final_conv_length']),
        Deep4Net=dict(in_chans=len(cfg['eeg_chans']), n_classes=n_classes, input_window_samples=iws, final_conv_length=cfg['final_conv_length']),
        EEGInception=dict(in_channels=len(cfg['eeg_chans']), n_classes=n_classes, input_window_samples=iws, sfreq=250),
        EEGITNet=dict(in_channels=len(cfg['eeg_chans']), n_classes=n_classes, input_window_samples=iws),
        EEGNetv1=dict(in_chans=len(cfg['eeg_chans']), n_classes=n_classes, input_window_samples=iws, final_conv_length=cfg['final_conv_length']),
        EEGNetv4=dict(in_chans=len(cfg['eeg_chans']), n_classes=n_classes, input_window_samples=iws, final_conv_length=cfg['final_conv_length']),
        HybridNet=dict(in_chans=len(cfg['eeg_chans']), n_classes=n_classes, input_window_samples=iws),
        EEGResNet=dict(in_chans=len(cfg['eeg_chans']), n_classes=n_classes, input_window_samples=iws, n_first_filters=16, final_pool_length=8),
        TIDNet=dict(in_chans=len(cfg['eeg_chans']), n_classes=n_classes, input_window_samples=iws),
    )

    model = cfg['model_cls'](
        **model_params[cfg['model_cls'].__name__],
    )

    # train
    classif = BrainDecodeClassification(model, cfg)
    model_name = f'braindecode_{model.__class__.__name__}'
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
    for model in models_to_try:
        metrics = main(model_cls=model, batch_size=4)
        metricz[model.__name__] = metrics
        print('='*80)
        print('=' * 80)
        print(model.__name__, '|', metrics)
        print('=' * 80)
        print('=' * 80)
        print(metricz)

    model_names = list(metricz.keys())
    min_val_loss_i = np.argsort([m['min_val_loss'] for m in metricz.values()])[0]
    max_acc_i = np.argsort([m['max_acc'] for m in metricz.values()])[-1]
    print('best val loss:', model_names[min_val_loss_i], metricz[model_names[min_val_loss_i]])
    print('best val acc:', model_names[max_acc_i], metricz[model_names[max_acc_i]])
