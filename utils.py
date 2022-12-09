import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from models import ModelConfig

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from typing import Optional

plt.style.use('seaborn')


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
