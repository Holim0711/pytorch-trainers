import torch
from .utils import _convert


class Phaser():

    def __init__(self, model, criterion, optimizer, device=None,
                 convert_x=None, convert_y=None):
        if device is not None:
            model.to(device)
            criterion.to(device)
            optimizer.load_state_dict(optimizer.state_dict())

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.convert_x = convert_x if convert_x is not None else _convert
        self.convert_y = convert_y if convert_y is not None else _convert

        self.callbacks = {
            'train': [],
            'valid': [],
        }

    def after(self, phase):
        if phase not in self.callbacks:
            raise ValueError(f'invalid phase name: {phase}')
        def register(func):
            self.callbacks[phase].append(func)
        return register

    def train(self, dataloader):
        self.model.train()

        for x, y in dataloader:
            x = self.convert_x(x, device=self.device)
            y = self.convert_y(y, device=self.device)

            ŷ = self.model(x)
            l = self.criterion(ŷ, y)

            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()

            for func in self.callbacks['train']:
                func(input=x, true=y, pred=ŷ, loss=l)

    @torch.no_grad()
    def valid(self, dataloader):
        self.model.eval()

        for x, y in dataloader:
            x = self.convert_x(x, device=self.device)
            y = self.convert_y(y, device=self.device)

            ŷ = self.model(x)
            l = self.criterion(ŷ, y)

            for func in self.callbacks['valid']:
                func(input=x, true=y, pred=ŷ, loss=l)
