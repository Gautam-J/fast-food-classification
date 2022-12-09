import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from torchvision.utils import make_grid
from torchvision.io import read_image, ImageReadMode

from typing import Iterable, Optional


def plot_learning_curve(train_loss: list[float],
                        test_loss: list[float],
                        train_acc: list[float],
                        test_acc: list[float],
                        path: str) -> None:
    """Plots and saves the learning curve of the given training history."""

    plt.figure(figsize=(8, 8))
    plt.subplot(211)
    plt.plot(train_acc, label='train_accuracy')
    plt.plot(test_acc, label='test_accuracy')
    plt.title(f'Train: {train_acc[-1]:.2%} | Test: {test_acc[-1]:.2%}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.subplot(212)
    plt.plot(train_loss, label='train_loss')
    plt.plot(test_loss, label='test_loss')
    plt.title(f'Train: {train_loss[-1]:.4f} | Test: {test_loss[-1]:.4f}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig(f'{path}/learning_curve.png')
    plt.close()


def plot_overfit_test(lossi: list[float],
                      path: str,
                      log_transform: bool = True) -> None:
    """Plots the rolling average of loss values."""

    ylabel = 'Loss'
    if log_transform:
        lossi = torch.log10(torch.tensor(lossi)).numpy()
        ylabel = r'$\log_{10}$ Loss'

    plt.plot(lossi)
    plt.xlabel('Iterations')
    plt.ylabel(ylabel)
    plt.title('Overfit on single batch data')
    plt.savefig(f'{path}/overfit_single_batch.png')
    plt.close()


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
    sns.heatmap(df, annot=True, cmap='coolwarm', cbar=False)

    os.makedirs(directory, exist_ok=True)

    # TODO: check working of rotation of ticks
    # cr.yaxis.set_ticklabels(cr.yaxis.get_ticklabels(),
    #                         rotation=0, ha='right', fontsize=10)
    # cr.xaxis.set_ticklabels(cr.xaxis.get_ticklabels(),
    #                         rotation=45, ha='right', fontsize=10)

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, ha='right', fontsize=10)

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
    sns.heatmap(df, annot=True, cmap='coolwarm', cbar=False)

    os.makedirs(directory, exist_ok=True)

    # hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(),
    #                         rotation=0, ha='right', fontsize=10)
    # hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(),
    #                         rotation=45, ha='right', fontsize=10)

    # hm.set_xlabel('Predicted Label')
    # hm.set_ylabel('True Label')

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, ha='right', fontsize=10)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.tight_layout()
    plt.savefig(f'{directory}/confusion_matrix.png')
    plt.close()


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

    imgs = [TVF.resize(read_image(path, mode=ImageReadMode.RGB), size=[
                       img_size, img_size]) for path in paths]
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
    plt.close()


def get_rolling_average(values: list[float],
                        window_size: int = 50) -> list[float]:
    """Returns a list of rolling/sliding mean values."""

    t_values = torch.tensor(values)
    rolling_avg = F.avg_pool1d(t_values.view(
        1, -1), kernel_size=window_size, stride=1)

    return rolling_avg.view(-1,).numpy()


def plot_input_independent_test(lossi_zero: list[float],
                                lossi: list[float],
                                path: str,
                                window_size: int = 32) -> None:
    """Plots the loss curve for both zeroed and real inputs."""

    lossi_zero = get_rolling_average(lossi_zero, window_size)
    lossi = get_rolling_average(lossi, window_size)

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(lossi_zero)
    plt.title('Zeroed input')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.subplot(122)
    plt.plot(lossi)
    plt.title('Real input')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.suptitle('Input Independent Baseline Test')
    plt.savefig(f'{path}/input_independent_test.png')
    plt.close()
