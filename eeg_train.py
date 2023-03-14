import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co
import h5py
import pickle

import lightning as L

from skimage.transform import resize
from torchvision.transforms import Compose, ToTensor, Resize

from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)

from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

from src.EEGTfrEpochs import EEGTfrEpochs
from src.models.TfrClassification import TfrClassification
#from config.train.lr_finger import cfg
from config.train.lr_finger_tfr import cfg


# ds_root_path = 'out/'
# subject = '0717b399'

ds_root_path = "/home/Data/LeftRightFinger_Stim_ME/preproc"
subject = "0717b399"


class norm:  # TODO per-sample
    def __call__(self, x):
        return (x - x.min()) / (x.max() - x.min())


if __name__ == "__main__":
    model_cls = resnet18
    model_weights = ResNet18_Weights.DEFAULT

    transform = Compose(
        [ToTensor(), Resize(224)]
    )  # norm(), ToTensor(),  model_weights.transforms()])  # TODO norm(),

    with open(f"{ds_root_path}/{subject}/{subject}_meta.pckl", "rb") as f:
        meta = pickle.load(f)

   
    chan_names = meta["eeg_ch_names"]
    task_event_ids = meta["task_event_ids"]
    print('Available EEG Channels:', chan_names)
    print('Event IDs:', task_event_ids)

    include_chans_idx = [chan_names.index(name) for name in cfg["include_chans_name"]]
    event_id_cls_map = {
        task_event_ids[evname]: cls for evname, cls in cfg["event_name_cls_map"].items()
    }

    train_ds = EEGTfrEpochs(
        f"{ds_root_path}/{subject}/{subject}_streams.h5",
        event_id_cls_map,
        cfg["train_sessions"],
        include_chans_idx,
        cfg["epoch_t_interval"],
        cfg["epoch_type"],
        cfg["load_to_mem"],
        transform,
        cfg["match_t_and_freq_dim"],
    )
    valid_ds = EEGTfrEpochs(
        f"{ds_root_path}/{subject}/{subject}_streams.h5",
        event_id_cls_map,
        cfg["valid_sessions"],
        include_chans_idx,
        cfg["epoch_t_interval"],
        cfg["epoch_type"],
        cfg["load_to_mem"],
        transform,
        cfg["match_t_and_freq_dim"],
    )

    loader_cfg = dict(
        num_workers=cfg["num_workers"],
        prefetch_factor=cfg["prefetch_factor"],
        pin_memory=True,
        persistent_workers=cfg["num_workers"] > 0,
        batch_size=cfg["batch_size"],
    )

    train_dl = DataLoader(train_ds, shuffle=True, **loader_cfg)
    valid_dl = DataLoader(valid_ds, shuffle=False, **loader_cfg)

   
    # model
    model_name = "resnetbaby"
    model_fname_template = "{epoch}_{step}_{val_loss:.2f}"
    classif = TfrClassification(model_cls, model_weights, cfg)

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
        EarlyStopping("val_loss", patience=10),
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

    train_ds.close()
    valid_ds.close()
