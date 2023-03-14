import numpy as np

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import h5py
from typing import List, Dict, Tuple
from skimage.transform import resize


class EEGTfrEpochs(Dataset):
    def __init__(
        self,
        h5_path: str,
        event_id_cls_map: Dict[int, int],
        sessions: List[int],
        chans: List[int],
        epoch_t_interval: Tuple[float, float] = None,
        epoch_type="tfr_epochs_on_task",
        load_to_mem=True,
        transform=lambda x: x,
        match_t_and_freq_dim=None,
    ):
        assert match_t_and_freq_dim is None or (
            match_t_and_freq_dim is not None and epoch_type == "tfr_epochs_on_task"
        )

        self.data = h5py.File(h5_path, "r")
        self.epochs = self.data[epoch_type]
        self.event_id_cls_map = event_id_cls_map
        self.chans = chans
        self.transform = transform
        self.match_t_and_freq_dim = match_t_and_freq_dim  # TODO just pass cfg

        # extract needed epoch ids
        self.num_epochs = self.data.attrs["num_epochs"]
        cum_epochs = np.cumsum(self.num_epochs)
        sess_begs = np.concatenate([[0], cum_epochs[:-1]])[sessions]
        sess_ends = cum_epochs[sessions]

        self.epoch_idx = np.concatenate(
            [np.arange(sbeg, send) for sbeg, send in zip(sess_begs, sess_ends)]
        )
        self.events = self.data["events"][:, 2][self.epoch_idx]

        # get relevant events
        self.events = self.data["on_task_events"][:, 2][self.epoch_idx]
        include_events = np.logical_or.reduce(
            [self.events == evid for evid in event_id_cls_map.keys()], axis=0
        )
        self.events = self.events[include_events]
        self.epoch_idx = self.epoch_idx[include_events]

        # define start and end of epoch
        self.times = self.data.attrs["on_task_times"]
        epoch_t_interval = (
            (0, self.times[-1]) if epoch_t_interval is None else epoch_t_interval
        )
        self.epoch_t_slice = slice(
            np.argmin(np.abs(self.times - epoch_t_interval[0])),
            np.argmin(np.abs(self.times - epoch_t_interval[1])),
        )
        self.times = self.times[self.epoch_t_slice]

        # load data to memory
        self.epochs_in_mem = None
        if load_to_mem:
            # can't have two indexing vectors, too fancy
            self.epochs_in_mem = np.stack(
                [
                    self.epochs[self.epoch_idx, chan, :, self.epoch_t_slice]
                    for chan in chans
                ],
                axis=1,
            )
            self.epochs = None
            self.data.close()
            self.data = None

    def __getitem__(self, index) -> T_co:
        x = (
            self.epochs_in_mem[index, ...]
            if self.epochs_in_mem is not None
            else self.epochs[self.epoch_idx[index], self.chans, ..., self.epoch_t_slice]
        )
        y = self.event_id_cls_map[self.events[index]]

        # TODO move this to init when epochs in mem
        if self.match_t_and_freq_dim:
            x = np.transpose(x, (1, 2, 0))  # channel last
            match_dim = self.match_t_and_freq_dim(x.shape[:2])
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
