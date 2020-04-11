import torch


class BaseTrainer():

    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.train_callback = None
        self.valid_callback = None

    def after_train(self, func):
        self.train_callback = func
        return func

    def after_valid(self, func):
        self.valid_callback = func
        return func

    def _train(self, i, x, y):
        ŷ = self.model(x)
        ℓ = self.criterion(ŷ, y)

        self.optimizer.zero_grad()
        ℓ.backward()
        self.optimizer.step()

        if self.train_callback is not None:
            self.train_callback(i=i, input=x, output=ŷ, target=y, loss=ℓ)

    def _valid(self, i, x, y):
        ŷ = self.model(x)
        ℓ = self.criterion(ŷ, y)

        if self.valid_callback is not None:
            self.valid_callback(i=i, input=x, output=ŷ, target=y, loss=ℓ)

    def train(self, dataloader):
        self.model.train()
        for i, (x, y) in enumerate(dataloader):
            self._train(i, x, y)

    @torch.no_grad()
    def valid(self, dataloader):
        self.model.eval()
        for i, (x, y) in enumerate(dataloader):
            self._valid(i, x, y)
