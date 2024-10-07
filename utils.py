import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from models import ModelConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder

from typing import Optional, Callable, TypeVar
from typing_extensions import ParamSpec

plt.style.use('seaborn')
R = TypeVar("R")  # return value of inner function
P = ParamSpec("P")  # parameters of inner function


def measure_time(func: Callable[P, R]) -> Callable[P, R]:
    """Print time taken ({}m {}s) to execute the wrapper function.

    Parameters
    ----------
    func : Callable[P, R]
        The function to be wrapped.

    Returns
    -------
    Callable[P, R]
        The input function, wrapped with this decorator.
    """
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start_time = time.time()
        result = func(*args, **kwargs)
        run_time = round(time.time() - start_time)
        m, s = divmod(run_time, 60)
        print('-' * 80)
        print(f"'{func.__name__}' took {m:.0f}m {s:.2f}s to execute.")
        print('-' * 80)
        return result
    return wrapper


def plot_predictions(model: nn.Module,
                     config: ModelConfig,
                     device: torch.device,
                     path: str,
                     n_images: int = 25) -> None:
    """Get predictions for random subset of testing dataset."""

    transform = get_transforms(config)
    dataset = ImageFolder('./data/test', transform=transform)
    random_dataset = random.sample(dataset.imgs, n_images)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    data = []

    with torch.no_grad():
        model.eval()
        for x, y in random_dataset:
            img = Image.open(x)
            img_t = transform(img).to(device)
            img_t = torch.unsqueeze(img_t, dim=0)
            pred = model(img_t)
            probs = F.softmax(pred, dim=1)
            pred_idx = torch.argmax(pred, dim=1).item()
            pred_class = idx_to_class[pred_idx]
            y_class = idx_to_class[y]
            data.append((x, pred_class, probs[0][pred_idx].item(), y_class))

    nrows = int(n_images**0.5)
    plt.figure(figsize=(10, 10))
    for i in range(1, n_images + 1):
        plt.subplot(nrows, nrows, i)
        img = Image.open(data[i - 1][0])
        img = img.resize((200, 200))
        plt.title(f'{data[i-1][3]} | {data[i-1][1]} - {data[i-1][2]:.2%}', size=8)
        plt.imshow(img)
        plt.grid()
        plt.axis('off')

    plt.suptitle('Label | Prediction - Confidence Score')
    plt.tight_layout()
    plt.savefig(f'{path}/predictions.png')


def save_model(model: nn.Module, path: str) -> None:
    """Saves the pytorch model, along with its weight."""
    torch.save(model, f'{path}/model.pt')
    torch.save(model.state_dict(), f'{path}/weights.pt')


def get_transforms(config: ModelConfig) -> transforms.Compose:
    """Returns a compose of torchvision transforms."""

    transform = [
        transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    if config.n_channels == 1:
        transform.append(transforms.Grayscale())

    return transforms.Compose(transform)


def get_logger(name: str,
               io_stream: StringIO = None,
               file_path: str = None,
               formatter: logging.Formatter = None,
               base_level: int = logging.DEBUG,
               file_level: int = logging.INFO,
               io_level: int = logging.ERROR) -> logging.Logger:
    """Set up a python logger to log results.

    Parameters
    ----------
    name : str
        Name of the logger.
    io_stream : StringIO, default=None
        StringIO object to capture logs inside the string.
    formatter : logging.Formatter, default=None
        A custom formatter for the logger to output. If None, a default
        formatter of format `"%Y-%m-%d %H:%M:%S LEVEL MESSAGE"` is used.
    file_path : str, default=None
        File path to record logs. Must end with a readable extension. If None,
        the logs are not logged in any file, and are logged only to `stdout`.
    base_level : LEVEL or int, default=logging.DEBUG (10)
        Base level to log. Any level lower than this level will not be logged
        by any of the handlers.
    file_level: LEVEL or int, default=logging.INFO (20)
        Level to log in file. Any level lower than this will not be logged
        by the file handler.
    io_level: LEVEL or int, default=logging.ERROR (40)
        Level to log in io stream. Any level lower than this will not be logged
        by the IO stream handler.

    Returns
    -------
    logger : logging.Logger
        A logger with formatters and handlers attached to it.

    Notes
    -----
    If passing a directory name along with the log file, make sure the
    directory exists.

    Example for setting up IO Stream:
    >>> import io
    >>> ios = io.StringIO()
    >>> logger = get_logger('my_logger', io_stream=ios)
    >>> contents = ios.getvalue()
    >>> ios.close()
    >>> print(contents)
    """

    try:
        if formatter is None:
            formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s",
                                          datefmt='%Y-%m-%d %H:%M:%S%z')

        logger = logging.getLogger(name)
        logger.setLevel(base_level)

        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        logger.addHandler(streamHandler)

        if io_stream is not None:
            ioStreamHandler = logging.StreamHandler(io_stream)
            ioStreamHandler.setFormatter(formatter)
            ioStreamHandler.setLevel(io_level)
            logger.addHandler(ioStreamHandler)

        if file_path is not None:
            fileHandler = logging.FileHandler(file_path)
            fileHandler.setFormatter(formatter)
            fileHandler.setLevel(file_level)
            logger.addHandler(fileHandler)

        return logger
    except Exception as e:
        print('Error while getting logger:', e)


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
