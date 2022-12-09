import os
import time
import json
import torch
import argparse
import warnings

import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from models import Net, ModelConfig
from visualizations import plot_input_independent_test, plot_overfit_test, plot_learning_curve
from utils import get_dataloader, set_rng_seed, get_logger, save_model, plot_predictions
from stages import (
    verify_init_loss,
    overfit_single_batch,
    input_independent_baseline,
    chart_dependency_backprop
)

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fast food classification')
    parser.add_argument('--work_dir', '-o', type=str,
                        default='out', help="output working directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random number generator seed")
    parser.add_argument('--device', type=str, default='mps',
                        help="device to use for computation (cpu, cuda, mps)")
    parser.add_argument('--n_workers', type=int, default=8,
                        help="workers to use for loading data")
    parser.add_argument('--n_epochs', type=int, default=2,
                        help="number of epochs to train")
    parser.add_argument('--n_iters', type=int, default=500,
                        help="number of iteratiosn to train a single batch")
    parser.add_argument('--learning_rate', '-l', type=float,
                        default=1e-3, help="learning rate")
    parser.add_argument('--weight_decay', '-w', type=float,
                        default=0., help="weight decay")
    parser.add_argument('--init_loss', action='store_true',
                        help='Flag to test init loss')
    parser.add_argument('--ip_ind', action='store_true',
                        help='Flag to test input independent baseline')
    parser.add_argument('--overfit_single', action='store_true',
                        help='Flag to overfit single batch of data')
    parser.add_argument('--chart_backprop', action='store_true',
                        help='Flag to chart dependencies using backprop')
    opts = parser.parse_args()

    os.makedirs(opts.work_dir, exist_ok=True)
    logger = get_logger('ffc', file_path=f'{opts.work_dir}/temp.log')
    logger.info(f'{"NEW RUN":=^50}')

    config = ModelConfig()

    with open(f'{opts.work_dir}/opts.json', 'w') as f:
        f.write(json.dumps(vars(opts), indent=4))
    with open(f'{opts.work_dir}/config.json', 'w') as f:
        f.write(json.dumps(config.__dict__, indent=4))

    set_rng_seed(opts.seed)
    logger.info(f'RNG seed set as {opts.seed}')

    device = torch.device(opts.device)
    logger.info(f'Using {device} device')

    if opts.init_loss:
        logger.info('Starting init loss test...')
        init_loss, expected_loss = verify_init_loss(config, device)
        logger.critical(
            f'Initial loss: {init_loss:.4f} Expected loss: {expected_loss:.4f}')

    if opts.ip_ind:
        logger.info('Starting input-independent baseline test...')
        lossi_zero, lossi = input_independent_baseline(config, device, logger,
                                                       opts.learning_rate,
                                                       opts.n_workers)
        logger.critical(
            f'Zeroed input loss after 1 epoch: {lossi_zero[-1]:.4f}')
        logger.critical(f'Real input loss after 1 epoch: {lossi[-1]:.4f}')

        plot_input_independent_test(lossi_zero, lossi, path=opts.work_dir)
        logger.info('Saved input-independent baseline test plot')

    if opts.overfit_single:
        logger.info('Starting overfit on single batch data')
        lossi = overfit_single_batch(config, device, logger,
                                     opts.learning_rate,
                                     opts.n_workers,
                                     opts.n_iters)
        logger.critical(
            f'Loss after {opts.n_iters} iterations on {config.batch_size} inputs: {lossi[-1]:.4f}')

        plot_overfit_test(lossi, opts.work_dir, log_transform=True)
        logger.info('Saved overfit on single batch test plot')

    if opts.chart_backprop:
        logger.info('Starting dependencies test using backprop')
        chart_dependency_backprop(config, device)
        logger.critical('Dependecies verified using backprop')

    logger.info('Training model with given config and opts...')
    writer = SummaryWriter(comment=opts.work_dir)
    acci = []
    lossi = []
    train_acce = []
    test_acce = []
    train_losse = []
    test_losse = []

    model = Net(config)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(),
                     lr=opts.learning_rate,
                     weight_decay=opts.weight_decay)

    train_dataloader = get_dataloader("./data/train", config, opts.n_workers)
    test_dataloader = get_dataloader("./data/test", config, opts.n_workers, shuffle=False)

    for k in range(opts.n_epochs):
        logger.debug(f'Starting epoch {k}...')
        t0 = time.time()

        model.train()
        for i, data in enumerate(train_dataloader):
            x, y = data
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            acc = torch.sum(torch.argmax(pred, dim=1) == y) / pred.shape[0]
            lossi.append(loss.item())
            acci.append(acc.item())

            writer.add_scalar('Loss/train_batch', loss.item(), k * len(train_dataloader) + i)
            writer.add_scalar('Accuracy/train_batch', acc.item(), k * len(train_dataloader) + i)

            if i % 20 == 0:
                logger.debug(f'Epoch[{k}/{opts.n_epochs}] Iter[{i}/{len(train_dataloader)}] Loss: {loss.item():.4f} Acc: {acc.item():.2%}')

        t1 = time.time()

        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            total_acc = 0.0
            for x, y in test_dataloader:
                x, y = x.to(device), y.to(device)
                pred = model(x)

                total_loss += loss_fn(pred, y).item()
                total_acc += (torch.sum(torch.argmax(pred, dim=1) == y) / pred.shape[0]).item()

            test_loss = total_loss / len(test_dataloader)
            test_acc = total_acc / len(test_dataloader)
            test_acce.append(test_acc)
            test_losse.append(test_loss)

            writer.add_scalar('Loss/test_epoch', test_loss, k)
            writer.add_scalar('Accuracy/test_epoch', test_acc, k)

            total_loss = 0.0
            total_acc = 0.0
            for x, y in train_dataloader:
                x, y = x.to(device), y.to(device)
                pred = model(x)

                total_loss += loss_fn(pred, y).item()
                total_acc += (torch.sum(torch.argmax(pred, dim=1) == y) / pred.shape[0]).item()

            train_loss = total_loss / len(train_dataloader)
            train_acc = total_acc / len(train_dataloader)
            train_acce.append(train_acc)
            train_losse.append(train_loss)

            writer.add_scalar('Loss/train_epoch', train_loss, k)
            writer.add_scalar('Accuracy/train_epoch', train_acc, k)

        t2 = time.time()

        logger.debug('=' * 50)
        logger.critical(f'End of epoch {k} Time Taken: {t1 - t0:.2f} + {t2 - t1:.2f} = {t2 - t0:.2f}s')
        logger.critical(f'Training loss: {train_loss:.4f} Training acc: {train_acc:.2%}')
        logger.critical(f'Testing loss: {test_loss:.4f} Testing acc: {test_acc:.2%}')
        logger.debug('=' * 50)

    writer.close()

    save_model(model, opts.work_dir)
    logger.info("Model saved successfully!")

    plot_learning_curve(train_losse, test_losse, train_acce, test_acce, opts.work_dir)
    plot_predictions(model, config, device, opts.work_dir)

    logger.info('Experiment finished successfully!')
