from statistics import mean
import torch
from .utils import convert

schedulers = {
    'after_epoch': {
        'LambdaLR',
        'MultiplicativeLR',
        'StepLR',
        'MultiStepLR',
        'ExponentialLR',
        'CosineAnnealingLR',
    },
    'after_batch': {
        'CyclicLR',
        'OneCycleLR',
    },
    'special_cases': {
        'ReduceLROnPlateau',
        'CosineAnnealingWarmRestarts',
    },
}


class BaseTrainer():

    def __init__(
            self,
            model,
            criterion,
            optimizer,
            scheduler=None,
            device=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.train_callback = None
        self.valid_callback = None

        self.epoch = 0

    @property
    def scheduler_name(self):
        return type(self.scheduler).__name__

    def after_train(self, func):
        if callable(func):
            self.train_callback = func
        else:
            raise Exception(f"Not a callable object: {func}")
        return func

    def after_valid(self, func):
        if callable(func):
            self.valid_callback = func
        else:
            raise Exception(f"Not a callable object: {func}")
        return func

    def train_batch(self, i, x, y):
        ŷ = self.model(x)
        ℓ = self.criterion(ŷ, y)

        self.optimizer.zero_grad()
        ℓ.backward()
        self.optimizer.step()

        if self.train_callback is not None:
            callback_kwargs = {
                'i': i,
                'input': x,
                'output': ŷ,
                'target': y,
                'loss': ℓ,
            }
            self.train_callback(**{
                k: v for k, v in callback_kwargs.items()
                if k in self.train_callback.__code__.co_varnames
            })

        return ℓ.item()

    def valid_batch(self, i, x, y):
        ŷ = self.model(x)
        ℓ = self.criterion(ŷ, y)

        if self.valid_callback is not None:
            callback_kwargs = {
                'i': i,
                'input': x,
                'output': ŷ,
                'target': y,
                'loss': ℓ,
            }
            self.valid_callback(**{
                k: v for k, v in callback_kwargs.items()
                if k in self.valid_callback.__code__.co_varnames
            })

        return ℓ.item()

    def train(self, dataloader):
        self.model.train()

        train_losses = []
        n_iters = len(dataloader)

        for i, (x, y) in enumerate(dataloader):
            if self.device is not None:
                x = convert(x, self.device)
                y = convert(y, self.device)

            ℓ = self.train_batch(i, x, y)
            train_losses.append(ℓ)

            if self.scheduler_name in schedulers['after_batch']:
                self.scheduler.step()
            elif self.scheduler_name == 'CosineAnnealingWarmRestarts':
                self.scheduler.step(self.epoch + i / n_iters)

        if self.scheduler_name in schedulers['after_epoch']:
            self.scheduler.step()

        self.epoch += 1
        return mean(train_losses)

    @torch.no_grad()
    def valid(self, dataloader):
        self.model.eval()

        losses = []

        for i, (x, y) in enumerate(dataloader):
            if self.device is not None:
                x = convert(x, self.device)
                y = convert(y, self.device)

            ℓ = self.valid_batch(i, x, y)
            losses.append(ℓ)

        valid_loss = mean(losses)

        if self.scheduler_name == 'ReduceLROnPlateau':
            self.scheduler.step(valid_loss)

        return valid_loss
