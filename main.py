import os
import json
import argparse

import torch
from torchinfo import summary

from utils import ModelConfig, Net
from utils import (
    set_rng_seed,
    get_logger,
    verify_init_loss,
    get_dataloader
)

parser = argparse.ArgumentParser(description='Fast food classification')
parser.add_argument('--work_dir', '-o', type=str, default='./out', help="output working directory")
parser.add_argument('--seed', type=int, default=42, help="random number generator seed")
parser.add_argument('--baseline', action='store_true', help="run for getting baseline")
parser.add_argument('--device', type=str, default='mps', help="device to use for computation (cpu, cuda, mps)")
parser.add_argument('--n_workers', type=int, default=8, help="workers to use for loading data")
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

model = Net(config)
print(summary(model, input_size=(config.batch_size, config.n_channels,
                                 config.img_size, config.img_size), verbose=0))

x = torch.randn((config.batch_size, config.n_channels,
                 config.img_size, config.img_size),
                device=device)
y = torch.randint(config.n_classes, size=(config.batch_size,), device=device)

init_loss, expected_loss = verify_init_loss(config, (x, y), device)
logger.critical(f'Initial loss: {init_loss:.4f} Expected loss: {expected_loss:.4f}')

train_loader = get_dataloader('./data/train', config, opts.n_workers)
test_loader = get_dataloader('./data/test', config, opts.n_workers, shuffle=False)
