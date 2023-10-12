import torch
from .transformer_block import BasicTransformerBlock
from torch import nn, Tensor
from torch import functional as F


class SqueezeExcitation1d(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
            self,
            input_channels: int,
            squeeze_channels: int,
            activation=torch.nn.ReLU,
            scale_activation=torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc1 = torch.nn.Conv1d(input_channels, squeeze_channels, kernel_size=1)
        self.fc2 = torch.nn.Conv1d(squeeze_channels, input_channels, kernel_size=1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input


class ErdsBandsNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.chans = len(cfg['eeg_chans']) * len(cfg['erds_bands'])
        self.growth = 2
        self.nlayers = 4
        self.growth_until = 5
        self.kernels = [21] + [3] * (self.nlayers - 1)
        self.strides = [ 7] + [2] * (self.nlayers - 1)
        self.alpha_droput = .1
        self.squeeze_factor = 4
        self.out_dim = len(cfg['events_to_cls'])
        assert len(self.kernels) == len(self.strides) == self.nlayers

        self.layers = self._get_chan_enc()
        self.flatten = nn.LSTM()  # nn.Flatten()
        self.lin_out = nn.LazyLinear(self.out_dim)
        self.out_act = nn.LogSoftmax(dim=-1)

    def _get_chan_enc(self):
        layers = []
        in_chans = self.chans
        for l in range(self.nlayers):
            out_chans = int(in_chans * self.growth) if l < self.growth_until else in_chans
            layer = self._get_conv_block(l, in_chans, out_chans)
            layers.extend(layer)
            in_chans = out_chans
        return nn.ModuleList(layers)

    def _get_conv_block(self, layer_i, in_chans, out_chans):  # cleaner code
        # from: https://stackoverflow.com/questions/59285058/batch-normalization-layer-for-cnn-lstm
        return [nn.Conv1d(in_chans, out_chans, self.kernels[layer_i], self.strides[layer_i], bias=False),
                nn.BatchNorm1d(out_chans),
                nn.SELU(),
                nn.FeatureAlphaDropout(self.alpha_droput),
                SqueezeExcitation1d(out_chans, out_chans // self.squeeze_factor)]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.lin_out(x)
        x = self.out_act(x)
        return x


if __name__ == '__main__':
    block = BasicTransformerBlock(dim=64, num_attention_heads=8, attention_head_dim=128, dropout=.5,
                                  cross_attention_dim=256)

    eeg = torch.zeros((4, 8, 256))
    subj_emb = torch.zeros((4, 1, 64))

    y = block(hidden_states=subj_emb, encoder_hidden_states=eeg)

    print(y.shape)


    torch.nn.LazyConv2d()
    torch.nn.LazyBatchNorm2d()
