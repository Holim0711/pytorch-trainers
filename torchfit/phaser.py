import torch
from .utils import _convert


class Phaser():

    def __init__(self, model, criterion, optimizer, device=None,
                 multi_batch=1, multi_batch_reduction='mean'):
        if device is not None:
            model.to(device)
            criterion.to(device)
            optimizer.load_state_dict(optimizer.state_dict())

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.multi_batch = int(multi_batch)
        self.multi_batch_reduction = multi_batch_reduction

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

    def finalize(self)
        self.optimizer.step()
        self.optimizer.zero_grad()
        for func in self.callbacks['train']:
            func(input=x, true=y, pred=ŷ, loss=l)

    def train(self, dataloader):
        self.model.train()

        n_batch = len(dataloader)
        n_remain = n_batch % self.multi_batch

        for i, (x, y) in enumerate(dataloader):
            x = _convert(x, device=self.device)
            y = _convert(y, device=self.device)

            ŷ = self.model(x)
            l = self.criterion(ŷ, y)

            if self.multi_batch > 1 and self.multi_batch_reduction == 'mean':
                l /= self.multi_batch if i < n_batch - n_remain else n_remain

            l.backward()

            if (i + 1) % self.multi_batch == 0:
                self.finalize()

        if n_remain != 0:
            if dataloader.drop_last:
                self.optimizer.zero_grad()
            else:
                self.finalize()

    @torch.no_grad()
    def valid(self, dataloader):
        self.model.eval()

        for x, y in dataloader:
            x = _convert(x, device=self.device)
            y = _convert(y, device=self.device)

            ŷ = self.model(x)
            l = self.criterion(ŷ, y)

            for func in self.callbacks['valid']:
                func(input=x, true=y, pred=ŷ, loss=l)
