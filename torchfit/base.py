import torch
from tqdm import tqdm


class BasePhaser():

    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.callback_train = None
        self.callback_valid = None
        self.callback_params = None

    def after_train(self, func):
        self.callback_train = func
        return func

    def after_valid(self, func):
        self.callback_valid = func
        return func

    def _train(self, x, y):
        ŷ = self.model(x)
        l = self.criterion(ŷ, y)
        l.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.callback_params['input'] = x
        self.callback_params['pred'] = ŷ
        self.callback_params['true'] = y
        self.callback_params['loss'] = l

    def _valid(self, x, y):
        ŷ = self.model(x)
        l = self.criterion(ŷ, y)

        self.callback_params['input'] = x
        self.callback_params['pred'] = ŷ
        self.callback_params['true'] = y
        self.callback_params['loss'] = l

    def train(self, dataloader):
        self.model.train()

        self.callback_params = {}

        for x, y in tqdm(dataloader):
            self._train(x, y)

            if self.callback_train:
                self.callback_train(**self.callback_params)

            self.callback_params.clear()

    @torch.no_grad()
    def valid(self, dataloader):
        self.model.eval()

        self.callback_params = {}

        for x, y in tqdm(dataloader):
            self._valid(x, y)

            if self.callback_valid:
                self.callback_valid(**self.callback_params)

            self.callback_params.clear()
