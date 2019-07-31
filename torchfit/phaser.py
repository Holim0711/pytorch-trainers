import torch
from inspect import getargspec


def _check_params(func, params):
    argspec = getargspec(func)
    for x in params:
        if x not in argspec.args:
            raise Exception(f"There's no {x} in this function.")


class Phaser():

    def __init__(self, model, criterion, optimizer, device='cpu'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device(device)

        self._after_train_batch_func = None
        self._after_train_epoch_func = None
        self._after_valid_batch_func = None
        self._after_valid_epoch_func = None

    def train(self, dataloader):
        self.model.train()
        loss_sum = 0

        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            ŷ = self.model(x)
            loss = self.criterion(ŷ, y)
            loss_sum += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self._after_train_batch_func:
                self._after_train_batch_func(pred=ŷ, true=y)

        if self._after_train_epoch_func:
            self._after_train_epoch_func(loss=loss_sum/len(dataloader))

    @torch.no_grad()
    def valid(self, dataloader):
        self.model.eval()
        loss_sum = 0

        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            ŷ = self.model(x)
            loss = self.criterion(ŷ, y)
            loss_sum += loss.item()

            if self._after_valid_batch_func:
                self._after_valid_batch_func(pred=ŷ, true=y)

        if self._after_valid_epoch_func:
            self._after_valid_epoch_func(loss=loss_sum/len(dataloader))

    def run(self, train_dataloader, valid_dataloader):
        self.train(train_dataloader)
        self.valid(valid_dataloader)

    def after_train_batch(self, func):
        _check_params(func, ['pred', 'true'])
        self._after_train_batch_func = func
        return func

    def after_train_epoch(self, func):
        _check_params(func, ['loss'])
        self._after_train_epoch_func = func
        return func

    def after_valid_batch(self, func):
        _check_params(func, ['pred', 'true'])
        self._after_valid_batch_func = func
        return func

    def after_valid_epoch(self, func):
        _check_params(func, ['loss'])
        self._after_valid_epoch_func = func
        return func
