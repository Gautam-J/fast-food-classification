import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from models import ModelConfig, Net


def overfit_single_batch(config: ModelConfig,
                         device: torch.device,
                         logger: logging.Logger,
                         lr: float,
                         n_workers: int,
                         n_epochs: int) -> list[float]:
    """Returns loss values afer overfitting on a single batch of data."""

    train_loader = get_dataloader('./data/train', config, n_workers)
    batch_iter = iter(train_loader)
    current_batch = next(batch_iter)
    data = [x for x in current_batch]
    x, y = data
    x, y = x.to(device), y.to(device)

    model = Net(config)
    model.to(device)
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    lossi = []

    for k in range(n_epochs):
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        lossi.append(loss.item())

        if k % 10 == 0:
            logger.debug(f'Epoch: {k} Loss: {loss.item():.4f}')

    return lossi


def input_independent_baseline(config: ModelConfig,
                               device: torch.device,
                               logger: logging.Logger,
                               lr: float,
                               n_workers: int) -> tuple[list[float],
                                                        list[float]]:
    """Returns the loss values per batch for zeroed input and normal input."""
    # TODO: Refactor this function to avoid repeated code.

    logger.info('Training with zeroed input...')

    model = Net(config)
    model.to(device)
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    train_loader = get_dataloader('./data/train', config, n_workers)
    lossi_zero = []

    for i, data in enumerate(train_loader):
        x, y = data
        x, y = torch.zeros_like(x).to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        lossi_zero.append(loss.item())

        if i % 10 == 0:
            logger.debug(f'Iteration: {i} Loss: {loss.item():.4f}')

    logger.info('Training with real input...')

    model = Net(config)
    model.to(device)
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    train_loader = get_dataloader('./data/train', config, n_workers)
    lossi = []

    for i, data in enumerate(train_loader):
        x, y = data
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        lossi.append(loss.item())

        if i % 10 == 0:
            logger.debug(f'Iteration: {i} Loss: {loss.item():.4f}')

    return lossi_zero, lossi


@torch.no_grad()
def verify_init_loss(config: ModelConfig,
                     device: torch.device) -> tuple[float, float]:
    """Returns initial loss of model and expected loss."""

    model = Net(config)
    model.to(device)
    model.eval()

    x = torch.randn((config.batch_size, config.n_channels,
                     config.img_size + 4, config.img_size + 4),
                    device=device)
    y = torch.randint(config.n_classes, size=(config.batch_size,),
                      device=device)

    pred = model(x)
    init_loss = F.cross_entropy(pred, y).item()
    expected_loss = -torch.log(1/torch.tensor(config.n_classes)).item()

    return init_loss, expected_loss


def chart_dependency_backprop(config: ModelConfig,
                              device: torch.device) -> None:
    """Charts dependencies using backpropagation."""

    model = Net(config)
    model.to(device)
    model.train()

    x = torch.randn((config.batch_size, 1, config.img_size + 4,
                     config.img_size + 4), device=device)
    x.requires_grad = True

    out = model(x)
    loss = out[2].sum()
    loss.backward()

    assert x.grad is not None
    for i in range(config.batch_size):
        if i != 2:
            assert (x.grad[i] == 0).all()
        else:
            assert (x.grad[i] != 0).any()
