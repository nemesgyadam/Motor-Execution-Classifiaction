# from: https://github.com/arayabrain/speech-decoding/blob/main/models.py
# paper: https://arxiv.org/pdf/2208.12266.pdf

import sys
import numpy as np
import torch
import torch.nn as nn
from time import time
import torch.nn.functional as F
from termcolor import cprint
from einops import rearrange

import yaml
import mne
import numpy as np
import torch


# from: https://github.com/arayabrain/speech-decoding/blob/main/configs/config.yaml
DEF_CFG = {
    'D1': 270,
    'D2': 320,
    'F': 512,
    'K': 32,
    'batch_size': 64,
    'd_drop': 0.3,
    'dataset': 'Brennan2018',
    'epochs': 3500,
    'hydra': {'job': {'chdir': True}},
    'init_temperature': 5.1,
    'lr': '3e-4',
    'lr_exp_gamma': 0.99,
    'lr_multistep_mlstns': [0.4, 0.6, 0.8, 0.9],
    'lr_scheduler': 'multistep',
    'lr_step_gamma': 0.5,
    'lr_step_numsteps': 5,
    'num_workers': 6,
    'preprocs': {'audio_resample_rate': 16000,
                 'baseline_len_sec': 0.5,
                 'brain_filter_high': 60,
                 'brain_filter_low': 1.0,
                 'brain_resample_rate': 120,
                 'clamp': True,
                 'clamp_lim': 20,
                 'last4layers': True,
                 'lowpass_filter_width': 128,
                 'seq_len_sec': 3,
                 'shift_brain': True,
                 'shift_len': 150,
                 'subject_wise': True},
    'rebuild_dataset': False,
    'reduction': 'mean',
    'reproducible': False,
    'updates': 1200,
    'use_sampler': True,
    'use_wandb': False,
    'wandb': {'entity': 'nightdude', 'project': 'speech_decoding'},
    'wav2vec_model': 'facebook/wav2vec2-large-xlsr-53'
}


def ch_locations_2d(info):
    # dataset_name, root_dir = args.dataset, args.root_dir
    #
    # if dataset_name == "Brennan2018":
    #     montage = mne.channels.make_standard_montage("easycap-M10")
    #     info = mne.create_info(ch_names=montage.ch_names, sfreq=512., ch_types="eeg")
    #     info.set_montage(montage)
    #
    #     layout = mne.channels.find_layout(info, ch_type="eeg")
    #
    #     loc = layout.pos[:, :2]  # ( 61, 2 )
    #     # Channel 29 was broken in Brennan 2018
    #     loc = np.delete(loc, 28, axis=0)  # ( 60, 2 )
    #
    # elif dataset_name == "Gwilliams2022":
    #     bids_path = mne_bids.BIDSPath(
    #         subject='01',
    #         session='0',
    #         task='0',
    #         datatype="meg",
    #         root=f'{root_dir}/data/Gwilliams2022/',
    #     )
    #     raw = mne_bids.read_raw_bids(bids_path)
    #
    #     layout = mne.channels.find_layout(raw.info, ch_type="meg")
    #
    #     loc = layout.pos[:, :2]
    #
    # else:
    #     raise ValueError()'

    layout = mne.channels.find_layout(info, ch_type="eeg")
    loc = layout.pos[:, :2]

    # min-max normalization
    loc = (loc - loc.min(axis=0)) / (loc.max(axis=0) - loc.min(axis=0))

    # NOTE: "In practice, as a_j is periodic, we scale down (x,y) to keep a margin of 0.1 on each side."
    loc = loc * 0.8 + 0.1

    return torch.from_numpy(loc.astype(np.float32))


class SpatialAttention(nn.Module):
    """Same as SpatialAttentionVer2, but a little more concise"""

    def __init__(self, cfg):
        super(SpatialAttention, self).__init__()

        # vectorize of k's and l's
        a = []
        for k in range(cfg['K']):
            for l in range(cfg['K']):
                a.append((k, l))
        a = torch.tensor(a)
        k, l = a[:, 0], a[:, 1]

        # vectorize x- and y-positions of the sensors
        loc = ch_locations_2d(cfg['eeg_info'])
        x, y = loc[:, 0], loc[:, 1]

        # make a complex-valued parameter, reshape k,l into one dimension
        self.z = nn.Parameter(torch.rand(size=(cfg['D1'], cfg['K'] ** 2), dtype=torch.cfloat))#.to(device)

        # NOTE: pre-compute the values of cos and sin (they depend on k, l, x and y which repeat)
        phi = 2 * torch.pi * (torch.einsum('k,x->kx', k, x) + torch.einsum('l,y->ly', l, y))  # torch.Size([1024, 60]))
        self.cos = torch.cos(phi).to(cfg['dev'])
        self.sin = torch.sin(phi).to(cfg['dev'])

        self.spatial_dropout = SpatialDropout(loc, cfg['d_drop'])

    def forward(self, X):

        # NOTE: do hadamard product and and sum over l and m (i.e. m, which is l X m)
        re = torch.einsum('jm, me -> je', self.z.real, self.cos)  # torch.Size([270, 60])
        im = torch.einsum('jm, me -> je', self.z.imag, self.sin)
        a = re + im  # essentially (unnormalized) weights with which to mix input channels into ouput channels
        # ( D1, num_channels )

        # NOTE: to get the softmax spatial attention weights over input electrodes,
        # we don't compute exp, etc (as in the eq. 5), we take softmax instead:
        SA_wts = F.softmax(a, dim=-1)  # each row sums to 1
        # ( D1, num_channels )

        # NOTE: drop some channels within a d_drop of the sampled channel
        dropped_X = self.spatial_dropout(X)

        # NOTE: each output is a diff weighted sum over each input channel
        return torch.einsum('oi,bit->bot', SA_wts, dropped_X)


class SpatialDropout(nn.Module):
    """Using same drop center for all samples in batch"""

    def __init__(self, loc, d_drop, dev):
        super(SpatialDropout, self).__init__()
        self.loc = loc  # ( num_channels, 2 )
        self.d_drop = d_drop
        self.num_channels = loc.shape[0]
        self.dev = dev

    def forward(self, X):  # ( B, num_channels, seq_len )
        assert X.shape[1] == self.num_channels

        if self.training:
            drop_center = self.loc[np.random.randint(self.num_channels)]  # ( 2, )
            distances = (self.loc - drop_center).norm(dim=-1)  # ( num_channels, )
            mask = torch.where(distances < self.d_drop, 0., 1.).to(self.dev)  # ( num_channels, )
            return torch.einsum('c,bct->bct', mask, X)
        else:
            return X


class SubjectBlock(nn.Module):

    def __init__(self, cfg):
        super(SubjectBlock, self).__init__()

        self.num_subjects = cfg['num_subjects']
        self.D1 = cfg['D1']
        self.K = cfg['K']
        self.spatial_attention = SpatialAttention(cfg)
        self.conv = nn.Conv1d(in_channels=self.D1, out_channels=self.D1, kernel_size=1, stride=1)
        self.subject_layer = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.D1,
                out_channels=self.D1,
                kernel_size=1,
                bias=False,
                stride=1,
                #device=device,
            ) for _ in range(self.num_subjects)
        ])

    def forward(self, X, subject_idxs):
        X = self.spatial_attention(X)  # ( B, 270, 256 )
        X = self.conv(X)  # ( B, 270, 256 )
        X = torch.cat([self.subject_layer[i](x.unsqueeze(dim=0)) for i, x in zip(subject_idxs, X)])  # ( B, 270, 256 )
        return X


class SubjectBlock_proto(nn.Module):

    def __init__(self, cfg):
        super(SubjectBlock_proto, self).__init__()

        self.num_subjects = cfg['num_subjects']
        self.D1 = cfg['D1']
        self.K = cfg['K']
        self.spatial_attention = SpatialAttention(cfg)
        self.conv = nn.Conv1d(in_channels=self.D1, out_channels=self.D1, kernel_size=1, stride=1)

        # NOTE: The below implementations are equivalent to learning a matrix:
        self.subject_matrix = nn.Parameter(torch.rand(self.num_subjects, self.D1, self.D1))
        # self.subject_layer = [
        #     nn.Conv1d(in_channels=self.D1, out_channels=self.D1, kernel_size=1, stride=1, device=device)
        #     for _ in range(self.num_subjects)
        # ]

    def forward(self, X, subject_idxs):
        X = self.spatial_attention(X)  # ( B, 270, 256 )
        X = self.conv(X)  # ( B, 270, 256 )

        # NOTE to Sensho: this has caused problems. I slighly changed it here. Hope it doesn't break anything for you
        _subject_idxs = subject_idxs.tolist()
        X = self.subject_matrix[_subject_idxs] @ X  # ( 270, 270 ) @ ( B , 270, 256 ) -> ( B, 270, 256 )
        # _X = []
        # for i, x in enumerate(X):  # x: ( 270, 256 )
        #     x = self.subject_layer[subject_idxs[i]](x.unsqueeze(0))  # ( 1, 270, 256 )
        #     _X.append(x.squeeze())
        # X = torch.stack(_X)

        return X  # ( B, 270, 256 )


class ConvBlock(nn.Module):

    def __init__(self, k, D1, D2):
        super(ConvBlock, self).__init__()

        self.k = k
        self.D2 = D2
        self.in_channels = D1 if k == 0 else D2

        self.conv0 = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.D2,
            kernel_size=3,
            padding='same',
            dilation=2**((2 * k) % 5),
        )
        self.batchnorm0 = nn.BatchNorm1d(num_features=self.D2)
        self.conv1 = nn.Conv1d(
            in_channels=self.D2,
            out_channels=self.D2,
            kernel_size=3,
            padding='same',
            dilation=2**((2 * k + 1) % 5),
        )
        self.batchnorm1 = nn.BatchNorm1d(num_features=self.D2)
        self.conv2 = nn.Conv1d(
            in_channels=self.D2,
            out_channels=2 * self.D2,
            kernel_size=3,
            padding='same',
            dilation=2,  #FIXME: The text doesn't say this, but the picture shows dilation=2
        )

    def forward(self, X):
        if self.k == 0:
            X = self.conv0(X)
        else:
            X = self.conv0(X) + X  # skip connection

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        X = self.conv2(X)
        X = F.glu(X, dim=-2)

        return X  # ( B, 320, 256 )


class BrainEncoder(nn.Module):

    def __init__(self, cfg):
        super(BrainEncoder, self).__init__()

        self.num_subjects = cfg['num_subjects']
        self.D1 = cfg['D1']
        self.D2 = cfg['D2']
        self.F = cfg['F'] if not cfg['last4layers'] else 1024
        self.K = cfg['K']

        # self.subject_block = SubjectBlock(cfg)
        self.subject_block = SubjectBlock_proto(cfg)
        cprint("USING THE NEW (PROTO) IMPLEMENTATION OF THE SUBJECT BLOCK", 'red', 'on_blue', attrs=['bold'])

        self.conv_blocks = nn.Sequential()
        for k in range(5):
            self.conv_blocks.add_module(f"conv{k}", ConvBlock(k, self.D1, self.D2))

        self.conv_final1 = nn.Conv1d(
            in_channels=self.D2,
            out_channels=2 * self.D2,
            kernel_size=1,
        )
        self.conv_final2 = nn.Conv1d(
            in_channels=2 * self.D2,
            out_channels=self.F,
            kernel_size=1,
        )

    def forward(self, X, subject_idxs):
        X = self.subject_block(X, subject_idxs)
        X = self.conv_blocks(X)
        X = F.gelu(self.conv_final1(X))
        X = F.gelu(self.conv_final2(X))
        return X
