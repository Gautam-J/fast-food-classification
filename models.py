from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0
from torchinfo import summary


@dataclass
class ModelConfig:
    img_size: int = 224
    n_channels: int = 3
    n_classes: int = 5
    batch_size: int = 32


class Net(nn.Module):

    def __init__(self, config: ModelConfig):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(config.n_channels, 32,
                               kernel_size=7,
                               stride=1,
                               padding='valid')
        self.conv2 = nn.Conv2d(32, 32,
                               kernel_size=3,
                               stride=1,
                               padding='valid')
        self.conv3 = nn.Conv2d(32, 64,
                               kernel_size=3,
                               stride=1,
                               padding='valid')
        self.conv4 = nn.Conv2d(64, 64,
                               kernel_size=3,
                               stride=1,
                               padding='valid')
        self.conv5 = nn.Conv2d(64, 64,
                               kernel_size=3,
                               stride=1,
                               padding='valid')
        self.conv6 = nn.Conv2d(64, 64,
                               kernel_size=3,
                               stride=1,
                               padding='valid')

        self.maxpool = nn.MaxPool2d(kernel_size=2,
                                    stride=2)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout2d(0.2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.out = nn.Linear(64, config.n_classes)

    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = self.maxpool(x)
        x = F.silu(self.conv2(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        x = F.silu(self.conv3(x))
        x = self.maxpool(x)
        x = F.silu(self.conv4(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        x = F.silu(self.conv5(x))
        x = self.maxpool(x)
        x = F.silu(self.conv6(x))
        x = self.global_avgpool(x)
        x = self.flatten(x)
        output = self.out(x)
        return output


if __name__ == "__main__":
    config = ModelConfig()
    # model = Net(config)
    model = efficientnet_b0()
    for p in model.parameters():
        p.requires_grad = False
    model.classifier[1] = nn.Linear(1280, config.n_classes)

    summary(model, (config.batch_size, config.n_channels,
                    config.img_size, config.img_size))
