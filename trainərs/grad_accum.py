from .base import BaseTrainer
from collections import defaultdict


class GradAccumTrainer(BaseTrainer):

    def __init__(self, model, criterion, optimizer, multi_batch, device=None):
        super().__init__(model, criterion, optimizer, device=device)
        self.multi_batch = int(multi_batch)
        self.train_callback_params = defaultdict(list)

    def _train_last_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.train_callback is not None:
            self.train_callback(**self.train_callback_params)
            self.train_callback_params.clear()

    def _train(self, i, x, y):
        if i < self._num_batches - self._num_remains:
            multi_batch = self.multi_batch
        else:
            multi_batch = self._num_remains

        ŷ = self.model(x)
        ℓ = self.criterion(ŷ, y) / multi_batch

        ℓ.backward()

        self.train_callback_params['i'].append(i)
        self.train_callback_params['input'].append(x)
        self.train_callback_params['output'].append(ŷ)
        self.train_callback_params['target'].append(y)
        self.train_callback_params['loss'].append(ℓ)

        if (i + 1) % self.multi_batch == 0:
            self._train_last_step()

    def train(self, dataloader, drop_last=False):
        self._num_batches = len(dataloader)
        self._num_remains = self._num_batches % self.multi_batch

        self.optimizer.zero_grad()
        self.train_callback_params.clear()

        super().train(dataloader)

        if self.num_remains:
            if drop_last:
                self.optimizer.zero_grad()
            else:
                self._train_last_step()
