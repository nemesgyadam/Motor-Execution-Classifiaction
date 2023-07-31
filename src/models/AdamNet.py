import torch
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncodingUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size1=(1, 3), kernel_size2=(3, 1), padding=1, stride=2):
        super(EncodingUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size1, padding=padding, stride=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size2, padding=padding, stride=2)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return nn.functional.relu(x)


class AdamNet(nn.Module):  # TODO use it
    def __init__(self, out_dim, softmax=False, add_channel=False):
        super(AdamNet, self).__init__()
        self.add_channel = add_channel
        self.encoder1 = EncodingUnit(1, 16)
        self.encoder2 = EncodingUnit(16, 32)
        self.encoder3 = EncodingUnit(32, 64)
        self.out_lin = nn.Linear(640, out_dim)  # TODO hardcoded
        if softmax:
            self.out_act = nn.Softmax()
        else:
            self.out_act = nn.Sigmoid()

    def forward(self, x):
        if self.add_channel:
            x = torch.unsqueeze(x, 1)

        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        # print(x.shape)
        x = x.reshape(x.shape[0], -1)
        x = self.out_act(self.out_lin(x))
        return x


class EEGNet(nn.Module):
    def __init__(self, num_classes: int = 4, channels: int = 22, samples: int = 401,
                 dropout_rate: float = 0.5, kernel_length: int = 64, num_filters1: int = 16,
                 depth_multiplier: int = 2, num_filters2: int = 32, norm_rate: float = 0.25, softmax=False, add_channel=False) -> None:
        super(EEGNet, self).__init__()

        self.channels = channels
        self.samples = samples
        self.add_channel = add_channel

        # First convolutional block
        # Temporal convolutional to learn frequency filters
        self.conv1 = nn.Conv2d(1, num_filters1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters1)

        # Depthwise convolutional block
        # Connected to each feature map individually, to learn frequency-specific spatial filters
        self.dw_conv1 = nn.Conv2d(num_filters1, num_filters1 * depth_multiplier, (channels, 1), groups=num_filters1,
                                  bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters1 * depth_multiplier)
        self.activation = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Separable convolutional block
        # Learns a temporal summary for each feature map individually,
        # followed by a pointwise convolution, which learns how to optimally mix the feature maps together
        self.sep_conv1 = nn.Conv2d(num_filters1 * depth_multiplier, num_filters1 * depth_multiplier, (1, 16),
                                   groups=num_filters1 * depth_multiplier, padding=(0, 8), bias=False)
        self.conv2 = nn.Conv2d(num_filters1 * depth_multiplier, num_filters2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_filters2)
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Fully connected layer
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(num_filters2 * (samples // 32), num_classes)

        if softmax:
            self.out_activation = nn.Softmax()
        else:
            self.out_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1, self.channels, self.samples)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dw_conv1(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)

        x = self.sep_conv1(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.dense(x)
        x = self.out_activation(x)

        return x


if __name__ == "__main__":
    model = AdamNet(2)
    batch_size = 4
    n = 550
    x = torch.randn(batch_size, 1, 8, n)
    output = model(x)

    print(output.shape)
