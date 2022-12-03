import os
import glob
import random
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision.io import read_image, ImageReadMode

from typing import Iterable, Optional

plt.style.use('seaborn')


@dataclass
class ModelConfig:
    img_size: int = 32
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


def get_transforms(config: ModelConfig) -> transforms.Compose:
    """Returns a compose of torchvision transforms."""

    transform = [
        transforms.ToTensor(),
        transforms.Resize((config.img_size, config.img_size)),
        transforms.Pad(2, fill=0),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]

    if config.n_channels == 1:
        transform.append(transforms.Grayscale())

    return transforms.Compose(transform)


def save_classification_report(y_true: Iterable,
                               y_pred: Iterable,
                               class_labels: list[str],
                               directory: str = '.') -> None:
    """Computes the classification report and saves it as a heatmap.
    Parameters
    ----------
    y_true : ndarray
        The true labels.
    y_pred : ndarray
        The predicted labels.
    class_labels : list[str] or list[int]
        A list of labels to be used in the heatmap for better readability.
    directory : str, default='.'
        Path to directory where the generated heatmap will be saved.
    Notes
    -----
    The directory passed should already exist. By default, the plot will
    be saved in the current working directory.
    """

    report = classification_report(y_true, y_pred, target_names=class_labels,
                                   output_dict=True)

    df = pd.DataFrame(report).T
    cr = sns.heatmap(df, annot=True, cmap='coolwarm', cbar=False)

    os.makedirs(directory, exist_ok=True)

    cr.yaxis.set_ticklabels(cr.yaxis.get_ticklabels(),
                            rotation=0, ha='right', fontsize=10)
    cr.xaxis.set_ticklabels(cr.xaxis.get_ticklabels(),
                            rotation=45, ha='right', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{directory}/classification_report.png')
    plt.close()


def save_confusion_matrix(y_true: Iterable,
                          y_pred: Iterable,
                          class_labels: list[str],
                          directory: str = '.') -> None:
    """Computes and saves the normalized confusion matrix.
    Parameters
    ----------
    y_true : ndarray
        The true labels.
    y_pred : ndarray
        The predicted labels.
    class_labels : list[str] or list[int]
        A list of labels to be used in the heatmap for better readability. This
        should be the same as in `y_true` and `y_pred`.
    directory : str, default='.'
        Path to directory where the generated heatmap will be saved.
    Note:
    -----
    The directory passed should already exist. By default, the plot will
    be saved in the current working directory.
    """

    matrix = confusion_matrix(y_true, y_pred, labels=class_labels,
                              normalize='true')

    df = pd.DataFrame(matrix, index=class_labels, columns=class_labels)
    hm = sns.heatmap(df, annot=True, cmap='coolwarm', cbar=False)

    os.makedirs(directory, exist_ok=True)

    hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(),
                            rotation=0, ha='right', fontsize=10)
    hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(),
                            rotation=45, ha='right', fontsize=10)

    hm.set_xlabel('Predicted Label')
    hm.set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig(f'{directory}/confusion_matrix.png')
    plt.close()


def get_logger(name: str,
               file_path: Optional[str] = None,
               formatter: Optional[logging.Formatter] = None,
               level: int = logging.DEBUG) -> logging.Logger:
    """Set up a python logger to log results.
    Parameters
    ----------
    name : str
        Name of the logger.
    formatter : logging.Formatter, default=None
        A custom formatter for the logger to output. If None, a default
        formatter of format `"%Y-%m-%d %H:%M:%S LEVEL MESSAGE"` is used.
    file_path : str, default=None
        File path to record logs. Must end with a readable extension. If None,
        the logs are not logged in any file, and are logged only to `stdout`.
    level : LEVEL or int, default=logging.DEBUG (10)
        Base level to log. Any level lower than this level will not be logged.
    Returns
    -------
    logger : logging.Logger
        A logger with formatters and handlers attached to it.
    Notes
    -----
    If passing a directory name along with the log file, make sure the
    directory exists.
    """

    if formatter is None:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)

    if file_path is not None:
        fileHandler = logging.FileHandler(file_path)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    return logger


def set_rng_seed(seed: int = 42) -> None:
    """Sets a fixed seed for random number generators for built-in random
    module, numpy, and pytorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def view_images_in_directory(path: str,
                             n_images: int = 25,
                             n_rows: int = 5,
                             img_size: int = 224,
                             save_path: Optional[str] = None) -> None:
    """Displays random images in a grid from a given directory."""

    label = path.split('/')[-1]
    paths = [img_path for img_path in glob.glob(f'{path}/*')]

    random_indices = np.random.randint(0, len(paths), size=(n_images,))
    paths = [paths[i] for i in random_indices]

    imgs = [TVF.resize(read_image(path, mode=ImageReadMode.RGB), size=[img_size, img_size]) for path in paths]
    grid = make_grid(imgs, nrow=n_rows)
    grid = grid.permute(1, 2, 0)  # channels_first to channels_last

    plt.figure(figsize=(8, 8))
    plt.title(label)
    plt.imshow(grid)
    plt.tight_layout()
    plt.grid(False)
    plt.axis('off')

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


@torch.no_grad()
def verify_init_loss(config: ModelConfig,
                     data: tuple[torch.Tensor, torch.Tensor],
                     device: torch.device) -> tuple[float, float]:
    """Returns initial loss of model and expected loss."""

    model = Net(config)
    model.to(device)
    model.eval()

    x, y = data
    x = x.to(device)
    y = y.to(device)

    pred = model(x)
    init_loss = F.cross_entropy(pred, y).item()
    expected_loss = -torch.log(1/torch.tensor(config.n_classes)).item()

    return init_loss, expected_loss


def get_dataloader(path: str, config: ModelConfig, n_workers: int,
                   shuffle: bool = True) -> DataLoader:
    """Returns a torchvision dataloader that iters over batched data(x, y)."""

    transform = get_transforms(config)
    dataset = ImageFolder(path, transform=transform)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=config.batch_size,
                            num_workers=n_workers,
                            shuffle=shuffle)

    return dataloader
