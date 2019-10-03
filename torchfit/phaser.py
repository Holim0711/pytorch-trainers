import torch
from .utils import *


class Phaser():

    def __init__(self, model, criterion, optimizer, device=None):
        if device is not None:
            model.to(device)
            criterion.to(device)
            optimizer.load_state_dict(optimizer.state_dict())

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.callbacks = {
            'train': None,
            'valid': None,
        }

    def train(self, dataloader):
        self.model.train()

        for x, y in dataloader:
            x = _convert_data(x, self.device)
            y = _convert_data(y, self.device)

            ŷ = self.model(x)
            l = self.criterion(ŷ, y)

            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()

            if self.callbacks['train'] is not None:
                self.callbacks['train'](pred=ŷ, true=y, loss=l)

    @torch.no_grad()
    def valid(self, dataloader):
        self.model.eval()

        for x, y in dataloader:
            x = _convert_data(x, self.device)
            y = _convert_data(y, self.device)

            ŷ = self.model(x)
            l = self.criterion(ŷ, y)

            if self.callbacks['valid'] is not None:
                self.callbacks['valid'](pred=ŷ, true=y, loss=l)

    def after(self, phase):
        if phase not in self.callbacks:
            raise ValueError(f'phase should be in {set(self.callbacks)}.')
        def register(func):
            _check_named_params(func, ['pred', 'true', 'loss'])
            self.callbacks[phase] = func
        return register
