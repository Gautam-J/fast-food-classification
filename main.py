import os
import json
import argparse

import torch

from utils import ModelConfig
from utils import (
    set_rng_seed,
    get_logger,
    verify_init_loss,
    overfit_single_batch,
    input_independent_baseline,
    plot_input_independent_test,
    plot_overfit_test,
    chart_dependency_backprop
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fast food classification')
    parser.add_argument('--work_dir', '-o', type=str, default='./out', help="output working directory")
    parser.add_argument('--seed', type=int, default=42, help="random number generator seed")
    parser.add_argument('--device', type=str, default='mps', help="device to use for computation (cpu, cuda, mps)")
    parser.add_argument('--n_workers', type=int, default=8, help="workers to use for loading data")
    parser.add_argument('--n_epochs', type=int, default=2, help="number of epochs to train")
    parser.add_argument('--learning_rate', '-l', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--weight_decay', '-w', type=float, default=0., help="weight decay")
    parser.add_argument('--init_loss', action='store_true', help='Flag to test init loss')
    parser.add_argument('--ip_ind', action='store_true', help='Flag to test input independent baseline')
    parser.add_argument('--overfit_single', action='store_true', help='Flag to overfit single batch of data')
    parser.add_argument('--chart_backprop', action='store_true', help='Flag to chart dependencies using backprop')
    opts = parser.parse_args()

    os.makedirs(opts.work_dir, exist_ok=True)
    logger = get_logger('ffc', file_path=f'{opts.work_dir}/temp.log')
    logger.info('====================NEW RUN====================')

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
        logger.critical(f'Initial loss: {init_loss:.4f} Expected loss: {expected_loss:.4f}')

    if opts.ip_ind:
        logger.info('Starting input-independent baseline test...')
        lossi_zero, lossi = input_independent_baseline(config, device, logger,
                                                       opts.learning_rate,
                                                       opts.n_workers)
        logger.critical(f'Zeroed input loss after 1 epoch: {lossi_zero[-1]:.4f}')
        logger.critical(f'Real input loss after 1 epoch: {lossi[-1]:.4f}')

        plot_input_independent_test(lossi_zero, lossi, path=opts.work_dir)
        logger.info('Saved input-independent baseline test plot')

    if opts.overfit_single:
        logger.info('Starting overfit on single batch data')
        lossi = overfit_single_batch(config, device, logger,
                                     opts.learning_rate,
                                     opts.n_workers,
                                     opts.n_epochs)
        logger.critical(f'Loss after {opts.n_epochs} iterations on {config.batch_size} inputs: {lossi[-1]:.4f}')

        plot_overfit_test(lossi, opts.work_dir, log_transform=True)
        logger.info('Saved overfit on single batch test plot')

    if opts.chart_backprop:
        logger.info('Starting dependencies test using backprop')
        chart_dependency_backprop(config, device)
        logger.info('Dependecies verified')
