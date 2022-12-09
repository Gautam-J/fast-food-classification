from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


@dataclass
class ModelConfig:
    img_size: int = 28
    n_channels: int = 1
    n_classes: int = 5
    batch_size: int = 32


class Net(nn.Module):

    def __init__(self, config: ModelConfig):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(config.n_channels, 6,
                               kernel_size=5,
                               stride=1,
                               padding='valid')
        self.conv2 = nn.Conv2d(6, 16,
                               kernel_size=5,
                               stride=1,
                               padding='valid')
        self.conv3 = nn.Conv2d(16, 120,
                               kernel_size=5,
                               stride=1,
                               padding='valid')

        self.avgpool = nn.AvgPool2d(kernel_size=2,
                                    stride=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(120, 84)
        self.out = nn.Linear(84, config.n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.avgpool(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool(x)
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        output = self.out(x)
        return output


if __name__ == "__main__":
    model = Net(ModelConfig())
    summary(model, (32, 1, 32, 32))
