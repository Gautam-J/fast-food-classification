import glob
import random
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import torch
from torchvision.utils import make_grid
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.functional as TVF

from typing import Iterable, List, Optional

plt.style.use('seaborn')


def save_classification_report(y_true: Iterable,
                               y_pred: Iterable,
                               class_labels: List[str],
                               directory: str = '.'):
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

    cr.yaxis.set_ticklabels(cr.yaxis.get_ticklabels(),
                            rotation=0, ha='right', fontsize=10)
    cr.xaxis.set_ticklabels(cr.xaxis.get_ticklabels(),
                            rotation=45, ha='right', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{directory}/classification_report.png')
    plt.close()


def save_confusion_matrix(y_true: Iterable,
                          y_pred: Iterable,
                          class_labels: List[str],
                          directory: str = '.'):
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
               log_file: Optional[str] = None,
               formatter: Optional[logging.Formatter] = None,
               level: int = logging.DEBUG):
    """Set up a python logger to log results.
    Parameters
    ----------
    name : str
        Name of the logger.
    formatter : logging.Formatter, default=None
        A custom formatter for the logger to output. If None, a default
        formatter of format `"%Y-%m-%d %H:%M:%S LEVEL MESSAGE"` is used.
    log_file : str, default=None
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

    if log_file is not None:
        fileHandler = logging.FileHandler(log_file)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    return logger


def set_rng_seed(seed: int = 42):
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

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


if __name__ == "__main__":
    pass
