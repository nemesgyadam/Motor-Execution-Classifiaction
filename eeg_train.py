import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co
import h5py
import pickle
from typing import List, Dict, Tuple
import wandb
import lightning as L
from scipy.interpolate import interp1d
from skimage.transform import resize
from torchvision.transforms import Compose, ToTensor
from torch.nn import Linear, Softmax, CrossEntropyLoss
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights


class EEGTfrEpochs(Dataset):

    def __init__(self, h5_path: str, event_id_cls_map: Dict[int, int], sessions: List[int], chans: List[int],
                 epoch_t_interval: Tuple[float, float] = None, epoch_type='tfr_epochs_on_task', load_to_mem=True,
                 transform=lambda x: x, match_t_and_freq_dim=None):

        self.data = h5py.File(h5_path, 'r')
        self.epochs = self.data[epoch_type]
        self.event_id_cls_map = event_id_cls_map
        self.chans = chans
        self.transform = transform
        self.match_t_and_freq_dim = match_t_and_freq_dim  # TODO just pass cfg

        # extract needed epoch ids
        self.num_epochs = self.data.attrs['num_epochs']
        cum_epochs = np.cumsum(self.num_epochs)
        sess_begs = np.concatenate([[0], cum_epochs[:-1]])[sessions]
        sess_ends = cum_epochs[sessions]

        self.epoch_idx = np.concatenate([np.arange(sbeg, send) for sbeg, send in zip(sess_begs, sess_ends)])
        self.events = self.data['events'][:, 2][self.epoch_idx]

        # get relevant events
        self.events = self.data['on_task_events'][:, 2][self.epoch_idx]
        include_events = np.logical_or.reduce([self.events == evid for evid in event_id_cls_map.keys()], axis=0)
        self.events = self.events[include_events]
        self.epoch_idx = self.epoch_idx[include_events]

        # define start and end of epoch
        self.times = self.data.attrs['on_task_times']
        epoch_t_interval = (0, self.times[-1]) if epoch_t_interval is None else epoch_t_interval
        self.epoch_t_slice = slice(np.argmin(np.abs(self.times - epoch_t_interval[0])),
                                   np.argmin(np.abs(self.times - epoch_t_interval[1])))
        self.times = self.times[self.epoch_t_slice]

        # load data to memory
        self.epochs_in_mem = None
        if load_to_mem:
            # can't have two indexing vectors, too fancy
            self.epochs_in_mem = np.stack([self.epochs[self.epoch_idx, chan, :, self.epoch_t_slice]
                                           for chan in chans], axis=1)
            self.epochs = None
            self.data.close()
            self.data = None

    def __getitem__(self, index) -> T_co:
        x = self.epochs_in_mem[index, ...] if self.epochs_in_mem is not None \
            else self.epochs[self.epoch_idx[index], self.chans, :, self.epoch_t_slice]
        y = self.event_id_cls_map[self.events[index]]

        # TODO move this to init when epochs in mem
        if self.match_t_and_freq_dim:
            # fx = interp1d(self.times, x, kind='linear', axis=-1)
            # fx(self.times[::])
            x = np.transpose(x, (1, 2, 0))  # channel last
            match_dim = self.match_t_and_freq_dim(x.shape[:2])  # TODO could be max
            x = resize(x, (match_dim, match_dim), preserve_range=True, order=2)
            # x = np.transpose(x, (2, 0, 1))  # ToTensor does the transpose

        return self.transform(x), y

    def __len__(self):
        return len(self.epoch_idx)

    def close(self):
        try:
            self.data.close()
        except:
            pass


class TfrClassification(L.LightningModule):
    def __init__(self, model_cls, model_weights, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = model_cls(weights=model_weights)
        self.model.fc = Linear(self.model.fc.in_features, len(cfg['event_name_cls_map'])).to('cuda')  # TODO rm to cuda
        # self.softmax = Softmax(dim=2)
        self.loss_fun = CrossEntropyLoss()

        self.model.requires_grad_(True)
        self.model.fc.requires_grad_(True)
        print(self.model)

    def training_step(self, batch, batch_idx):
        x, y = batch
        yy = self.model(x)
        loss = self.loss_fun(yy, y)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch
            yy = self.model(x)
            loss = self.loss_fun(yy, y)
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


# TODO augmentations (torchvision.transforms)


ds_root_path = 'out/'
subject = '0717b399'

cfg = dict(
    # data params
    train_sessions=list(range(0, 6)),  # 0-indexing
    valid_sessions=list(range(6, 7)),
    event_name_cls_map=dict(left=0, right=1),
    include_chans_name=['C3', 'Cz', 'C4'],
    epoch_t_interval=None,  # if None, (0, np.inf)
    epoch_type='tfr_epochs_on_task',
    load_to_mem=True,
    match_t_and_freq_dim=max,  # None | min | max | ...

    # model/training params
    epochs=100,
    batch_size=8,
    accumulate_grad_batches=1,
    dev='cuda',
    ndev=1,
    multi_dev_strat=None,
    precision=32,
    gradient_clip_val=None,
    num_workers=4,
    prefetch_factor=4,
)


class norm:
    def __call__(self, x):
        return (x - x.min()) / (x.max() - x.min())


if __name__ == '__main__':

    model_cls = resnet18
    model_weights = ResNet18_Weights.DEFAULT

    transform = Compose([norm(), ToTensor(), model_weights.transforms()])

    with open(f'{ds_root_path}/{subject}/{subject}_meta.pckl', 'rb') as f:
        meta = pickle.load(f)

    task_event_ids = meta['task_event_ids']
    chan_names = meta['eeg_ch_names']
    print(chan_names)
    print(task_event_ids)

    include_chans_idx = [chan_names.index(name) for name in cfg['include_chans_name']]
    event_id_cls_map = {task_event_ids[evname]: cls for evname, cls in cfg['event_name_cls_map'].items()}

    train_ds = EEGTfrEpochs(f'{ds_root_path}/{subject}/{subject}_streams.h5', event_id_cls_map,
                            cfg['train_sessions'], include_chans_idx, cfg['epoch_t_interval'],
                            cfg['epoch_type'], cfg['load_to_mem'], transform, cfg['match_t_and_freq_dim'])
    valid_ds = EEGTfrEpochs(f'{ds_root_path}/{subject}/{subject}_streams.h5', event_id_cls_map,
                            cfg['valid_sessions'], include_chans_idx, cfg['epoch_t_interval'],
                            cfg['epoch_type'], cfg['load_to_mem'], transform, cfg['match_t_and_freq_dim'])

    loader_cfg = dict(num_workers=cfg['num_workers'], prefetch_factor=cfg['prefetch_factor'], pin_memory=True,
                      persistent_workers=cfg['num_workers'] > 0, batch_size=cfg['batch_size'])

    train_dl = DataLoader(train_ds, shuffle=True, **loader_cfg)
    valid_dl = DataLoader(valid_ds, shuffle=False, **loader_cfg)

    # model
    model_name = 'resnetbaby'
    model_fname_template = '{epoch}_{step}_{val_loss:.2f}'
    classif = TfrClassification(model_cls, model_weights, cfg)

    callbacks = [ModelCheckpoint(f'models/{model_name}', model_fname_template, monitor='val_loss',
                                 save_top_k=1, save_last=False, verbose=True),
                 LearningRateMonitor(logging_interval='step'),
                 EarlyStopping('val_loss', patience=10)]

    trainer = L.Trainer(accelerator=cfg['dev'], devices=cfg['ndev'], strategy=cfg['multi_dev_strat'],
                        max_epochs=cfg['epochs'], default_root_dir=f'models/{model_name}', callbacks=callbacks,
                        benchmark=False, accumulate_grad_batches=cfg['accumulate_grad_batches'],
                        precision=cfg['precision'], gradient_clip_val=cfg['gradient_clip_val'])

    trainer.fit(classif, train_dl, valid_dl)

    train_ds.close()
    valid_ds.close()
