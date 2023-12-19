import glob
import sys
import os
from tqdm import tqdm

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

import warnings
from docstring_inheritance.docstring_inheritors.bases.inheritor import DocstringInheritanceWarning
warnings.filterwarnings("ignore", category=DocstringInheritanceWarning)

import mne
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from braindecode.models import *
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from copy import deepcopy
import lightning as L
from torch.nn import  NLLLoss
from lightning.pytorch.callbacks import Callback
import random
from pprint import pprint
from datetime import datetime
import matplotlib.pyplot as plt

import argparse
from data_gen import *


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Settings for training")
    parser.add_argument("--mode", type=str, help="Single or multi subject", choices=["single","multi"], default="single")
    parser.add_argument("--onset", type=str, help="Cue or pull onset", choices=["task", "pull"], default="task")
    return parser.parse_args()


class GatherMetrics(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        each_me = deepcopy(trainer.callback_metrics)
        self.metrics.append(each_me)


class BrainDecodeClassification(L.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.loss_fun = cfg['loss_fun']()
        self.accuracy = None
        self.accum_conf = None
        self.val_preds = None
        self.val_targets = None

        self.num_classes = len(cfg['events_to_cls']) if 'events_to_cls' in cfg else 4  # nothing/L/R/LR
        self.accuracy = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes)
        self.confusion = torchmetrics.ConfusionMatrix('multiclass', num_classes=self.num_classes)

        print(self.model)

    def training_step(self, batch, batch_idx):
        x, y = batch
        yy = self.model(x)

        # window is smaller than the epoch
        if len(yy.shape) == 3:
            yy = yy.mean(dim=-1)
            # TODO !!! it's possible that there is no relevant info in the first/last/middle window, so taking the mean here could ruin
            # TODO HAVE A LAST WEIGHTING LAYER IMPLEMENTED
            # TODO but this never runs with the models and data we have loaded rn

        loss = self.loss_fun(yy, y)
        return loss

    @staticmethod
    def _regr_to_classif(y):
        acc_y_in_0 = (y[:, 0] < .5) & (y[:, 1] < .5)
        acc_y_in_1 = (y[:, 0] >= .5) & (y[:, 1] < .5)
        acc_y_in_2 = (y[:, 0] < .5) & (y[:, 1] >= .5)
        acc_y_in_3 = (y[:, 0] >= .5) & (y[:, 1] >= .5)
        acc_y_in = torch.zeros(y.shape[0], dtype=y.dtype, device=y.device)
        acc_y_in[acc_y_in_0] = 0
        acc_y_in[acc_y_in_1] = 1
        acc_y_in[acc_y_in_2] = 2
        acc_y_in[acc_y_in_3] = 3
        return acc_y_in

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yy = self.model(x)
        if len(yy.shape) == 3:
            yy = yy.mean(dim=-1)

        loss = self.loss_fun(yy, y)
        self.log("val_loss", loss, prog_bar=True)
        if self.cfg['is_momentary']:
            y_regr = self._regr_to_classif(y)
            yy_regr = self._regr_to_classif(yy)
            acc = self.accuracy(yy_regr, y_regr)
            conf = self.confusion(yy_regr, y_regr)
        else:
            acc = self.accuracy(yy, y)
            conf = self.confusion(yy, y)
        self.log("val_acc", acc, prog_bar=True)

        if batch_idx == 0:
            self.accum_conf = conf
            self.val_preds = [yy]
            self.val_targets = [y]
        else:
            self.accum_conf += conf
            self.val_preds.append(yy)
            self.val_targets.append(y)

    def infer(self, x):
        with torch.no_grad():
            yy = self.model(x)
            if len(yy.shape) == 3:
                yy = yy.mean(dim=-1)
            return yy.cpu().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg['init_lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.cfg['reduce_lr_patience'], verbose=True, factor=0.0001,mode="max"
        )
        return {
            "optimizer": optimizer,
            #"lr_scheduler": scheduler,
            "monitor": "val_acc",
        }


class Ensemble(torch.nn.Module):
    def __init__(self, models: List[torch.nn.Module], classifs: List[BrainDecodeClassification],
                 accuracies: np.ndarray):
        super().__init__()
        self.models = models
        self.classifs = classifs
        self.accuracies = accuracies.reshape((-1, 1, 1))

    def forward(self, x):
        votes = np.stack([torch.exp(c(x)) for c in self.models], axis=0)  # models x batch x classes
        votes = (votes * self.accuracies).sum(axis=0)  # batch x classes
        votes = torch.argmax(votes, dim=1)
        return votes

    def infer(self, x):
        votes = np.stack([np.exp(c.infer(x)) for c in self.classifs], axis=0)  # models x batch x classes
        votes = (votes * self.accuracies).sum(axis=0)  # batch x classes
        votes = np.argmax(votes, axis=1)
        return votes


def load_model(path_to_dir, device='cuda'):
    ckpt_fname = sorted(glob.glob(f'{path_to_dir}/*.ckpt'))[-1]

    with open(f'{path_to_dir}/cfg.pkl', 'rb') as f:
        tmp_ = pickle.load(f)
    cfg, model_params = tmp_['cfg'], tmp_['model_params']

    model = cfg['model_cls'](**model_params[cfg['model_cls'].__name__])
    classif = BrainDecodeClassification.load_from_checkpoint(ckpt_fname, map_location=device, model=model, cfg=cfg)
    return classif, cfg


def ensemble_validation(models: List[torch.nn.Module], classifs: List[BrainDecodeClassification],
                        data_loader: DataLoader, accuracies: np.ndarray):
    accuracies = accuracies.reshape((-1, 1, 1))
    accuracy = torchmetrics.Accuracy('multiclass', num_classes=classifs[0].num_classes)
    confusion = torchmetrics.ConfusionMatrix('multiclass', num_classes=classifs[0].num_classes)
    votez, ys = [], []

    for x, y in data_loader:
        votes = np.stack([np.exp(c.infer(x)) for c in classifs], axis=0)  # models x batch x classes
        votes = (votes * accuracies).sum(axis=0)  # batch x classes
        votes = np.argmax(votes, axis=1)
        votez.append(votes)
        ys.append(y)

    votez = torch.as_tensor(np.concatenate(votez))
    ys = torch.as_tensor(np.concatenate(ys))

    acc = accuracy(votez, ys).numpy()
    conf = confusion(votez, ys).numpy()
    print(f'ENSEMBLE MODELING: accuracy: {acc}; confusion: {conf}')


def main(cfg, param_updates, **kwargs):
    cfg = deepcopy(cfg)
    cfg.update(kwargs)
    pprint(cfg)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    mne.set_log_level(False)
    datasets = []
    for subject in tqdm(cfg["subjects"], 'load subjects'):
        streams_path = f'{cfg["data_ver"]}/{subject}/{subject}_streams.h5'
        meta_path = f'{cfg["data_ver"]}/{subject}/{subject}_meta.pckl'

        data = EEGTimeDomainDataset(streams_path, meta_path, cfg)
        datasets.append(data)

    min_val_losses, max_val_accs = [], []

    if cfg["mode"] == "single":  # for subject 0717b399, session 13 and 14 are motor imaginary
        train_ds, valid_ds = data.rnd_split_by_session(train_session_idx=np.arange(1, 11), valid_session_idx=np.arange(11, 13))
    
    elif cfg["mode"] == "multi":
        train_ds, valid_ds = split_multi_subject_by_session(datasets)

    # init dataloaders
    dl_params = dict(num_workers=cfg['num_workers'], prefetch_factor=cfg['prefetch_factor'],
                     persistent_workers=cfg['num_workers'] > 0, pin_memory=True)

    train_dl = DataLoader(train_ds, cfg['batch_size'], shuffle=True, **dl_params)
    valid_dl = DataLoader(valid_ds, cfg['batch_size'], shuffle=False, **dl_params)

    # init model
    n_classes = len(np.unique(list(cfg['events_to_cls'].values())))

    # each model has its own parameter set, braindecode is fucked up, this per model parametrization is necessary
    # https://braindecode.org/stable/api.html#models
    iws = data[0][0].shape[1]
    model_params = dict(  # what a marvelously fucked up library
        ShallowFBCSPNet=dict(n_chans=len(cfg['eeg_chans']), n_outputs=n_classes, n_times=iws,
                             final_conv_length=cfg['final_conv_length'], n_filters_time=40,
                             filter_time_length=25, n_filters_spat=40, pool_time_length=75, pool_time_stride=15),
        Deep4Net=dict(n_chans=len(cfg['eeg_chans']), n_outputs=n_classes, n_times=iws,
                      final_conv_length=cfg['final_conv_length'], n_filters_time=25, n_filters_spat=25,
                      filter_time_length=10, n_filters_2=50, filter_length_2=10, n_filters_3=100,
                      filter_length_3=10, n_filters_4=200, filter_length_4=10),
        EEGInception=dict(n_chans=len(cfg['eeg_chans']), n_outputs=n_classes, n_times=iws, sfreq=data.sfreq,
                          n_filters=8),
        EEGITNet=dict(n_chans=len(cfg['eeg_chans']), n_outputs=n_classes, n_times=iws),
        EEGNetv1=dict(n_chans=len(cfg['eeg_chans']), n_outputs=n_classes, n_times=iws,
                      final_conv_length=cfg['final_conv_length']),
        EEGNetv4=dict(n_chans=len(cfg['eeg_chans']), n_outputs=n_classes, n_times=iws,
                      final_conv_length=cfg['final_conv_length'], F1=8, D=2, F2=16, kernel_length=64, third_kernel_size=(8, 4)),
        HybridNet=dict(n_chans=len(cfg['eeg_chans']), n_outputs=n_classes, n_times=iws),
        EEGResNet=dict(n_chans=len(cfg['eeg_chans']), n_outputs=n_classes, n_times=iws,
                       n_first_filters=16, final_pool_length=8, n_layers_per_block=2),
        TIDNet=dict(n_chans=len(cfg['eeg_chans']), n_outputs=n_classes, n_times=iws,
                    s_growth=24, t_filters=32, temp_layers=2, spat_layers=2, bottleneck=3),
        AdamNet={'out_dim': n_classes, 'softmax': True, 'add_channel': True},
        EEGNet={'num_classes': n_classes, 'channels': 8, 'samples': int(550 * cfg['resample']), 'softmax': True,
                **dict(dropout_rate=0.3, kernel_length=256, num_filters1=64, depth_multiplier=8, num_filters2=128)}
    )

    for model_name, params in param_updates.items():
        model_params[model_name].update(params)

    model = cfg['model_cls'](**model_params[cfg['model_cls'].__name__])

    # train
    classif = BrainDecodeClassification(model, cfg)
    model_name = f'braindecode_{args.mode}_{model.__class__.__name__}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    model_fname_template = "{epoch}_{step}_{val_loss:.2f}"

    os.makedirs(f'models/{model_name}', exist_ok=True)
    with open(f'models/{model_name}/cfg.pkl', 'wb') as f:
        pickle.dump({'cfg': cfg, 'model_params': model_params}, f)

    gather_metrics = GatherMetrics()
    model_checkpoint = ModelCheckpoint(
        f'checkpoints/{model_name}',
        model_fname_template,
        monitor='val_acc',
        save_top_k=1,
        save_last=False,
        verbose=True,
        mode="max")

    callbacks = [
        model_checkpoint,
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
            enable_progress_bar=False,
            logger=CSVLogger(f"checkpoints/{model_name}", name="training_logs"),
        )

    trainer.fit(classif, train_dl, valid_dl)

    min_val_loss = min([m['val_loss'].item() for m in gather_metrics.metrics])
    max_acc = max([m['val_acc'].item() for m in gather_metrics.metrics])

    min_val_losses.append(min_val_loss)
    max_val_accs.append(max_acc)

    # torchmetrics.ConfusionMatrix('multiclass', num_classes=4).update(classif.accum_conf).plot()

    metrics = dict(min_val_loss=np.mean(min_val_losses), max_acc=np.mean(max_val_accs),
                   min_val_loss_std=np.std(min_val_losses), max_acc_std=np.std(max_val_accs),
                   confusion=classif.accum_conf)
    model = BrainDecodeClassification.load_from_checkpoint(model_checkpoint.best_model_path,
                                                           model=cfg['model_cls'](**model_params[cfg['model_cls'].__name__]),
                                                           cfg=cfg).model
    return model, classif, trainer, metrics


if __name__ == "__main__":

    args = parse_args()
    preproc_path = 'out_bl-1--0.05_tfr-multitaper-logratio_reac-0.6_bad-95_f-2-40-100'
    subjects = os.listdir(preproc_path)

    subjects = ["0717b399"] if args.mode == "single" else subjects
    events_to_cls = {'left-pull': 0, 'right-pull': 1, 'left-right-pull': 2} if args.onset == "pull" \
        else {'left': 0, 'right': 1, 'left-right': 2, 'nothing': 3}

    default_cfg = dict(
        mode=args.mode,
        subjects=subjects,   # ['0717b399','1cfd7bfa', '4bc2006e', '4e7cac2d', '8c70c0d3', '0717b399'],
        data_ver='out_bl-1--0.05_tfr-multitaper-logratio_reac-0.6_bad-95_f-2-40-100',
        is_momentary=False,
        events_to_cls=events_to_cls,
        eeg_chans=['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8'],
        crop_t=(-.2, None),  # the part of the epoch to include
        resample=1.,  # no resample = 1.  # TODO
        stream_pred=False,
        part=args.onset,

        batch_size=16,
        num_workers=8,
        prefetch_factor=2,
        accumulate_grad_batches=1,
        precision=32,  # 16 | 32
        gradient_clip_val=1,
        loss_fun=NLLLoss,
        # number of times to randomly re-split train and valid, metrics are averaged across splits
        n_fold=None,
        # number of sessions to use as validation in k-fold cross-valid - all combinations are computed
        leave_k_out=None,
        reduce_lr_patience=5,
        early_stopping_patience=12,

        model_cls=ShallowFBCSPNet,  # default model
        # number of samples to include in each window of the decoder - now set to the full recording length
        # input_window_samples=100,
        final_conv_length='auto',

        dev='cuda',
        ndev=1,
        multi_dev_strat=None,

        epochs=100,
        init_lr=1e-3,
        train_data_ratio=.85,  # ratio of training data, the rest is validation
    )

    torch.use_deterministic_algorithms(False)

    models_to_try = [
        ShallowFBCSPNet,
        EEGNetv4,
        EEGResNet,
        EEGInception,
        EEGITNet,
        Deep4Net,
        TIDNet,
        HybridNet,
    ]

    model_params_updates = [
        dict(ShallowFBCSPNet=dict(n_filters_time=40, filter_time_length=25, n_filters_spat=40,
                                  pool_time_length=75, pool_time_stride=15),
             EEGResNet=dict(n_first_filters=16, final_pool_length=8, n_layers_per_block=2),
        )
    ]

    for update_i, updates in enumerate(model_params_updates):
        metricz = {}
        fails = []
        models, classifs, accuracies = [], [], []

        for model_cls in models_to_try:

            model, classif, trainer, metrics = main(default_cfg, updates, model_cls=model_cls, batch_size=8)
            models.append(model)
            classifs.append(classif)
            metricz[model_cls.__name__] = metrics
            accuracies.append(metrics['max_acc'])
            print('=' * 80, '\n', '=' * 80)
            print(update_i, model_cls.__name__, '|', metrics)
            print('=' * 80, '\n', '=' * 80)
            pprint(metricz)

            plt.figure()
            plt.imshow(metrics['confusion'].cpu().numpy())
            plt.title(f'{update_i}/{model_cls.__name__}')

        model_names = list(metricz.keys())
        min_val_loss_i = np.argsort([m['min_val_loss'] for m in metricz.values()])[0]
        max_acc_i = np.argsort([m['max_acc'] for m in metricz.values()])[-1]
        print(f'{update_i} best val loss:', model_names[min_val_loss_i], metricz[model_names[min_val_loss_i]])
        print(f'{update_i} best val acc:', model_names[max_acc_i], metricz[model_names[max_acc_i]])
        print(f'{update_i} fails:', fails, file=sys.stderr)

        plt.figure()
        plt.imshow(metricz[model_names[max_acc_i]]['confusion'].cpu().numpy())

        # ensemble
        subject = '0717b399'
        streams_path = f'{default_cfg["data_ver"]}/{subject}/{subject}_streams.h5'
        meta_path = f'{default_cfg["data_ver"]}/{subject}/{subject}_meta.pckl'
        data = EEGTimeDomainDataset(streams_path, meta_path, default_cfg, args.onset)

        dl_params = dict(num_workers=default_cfg['num_workers'], prefetch_factor=default_cfg['prefetch_factor'],
                         persistent_workers=default_cfg['num_workers'] > 0, pin_memory=True)
        train_ds, valid_ds = data.rnd_split_by_session(train_session_idx=np.arange(1, 11),
                                                       valid_session_idx=np.arange(11, 13))
        valid_dl = DataLoader(valid_ds, default_cfg['batch_size'], shuffle=False, **dl_params)

        ensemble_validation(models, classifs, valid_dl, np.array(accuracies))
