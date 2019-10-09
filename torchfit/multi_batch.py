from .base import BasePhaser
from .utils import iterate


class MultiBatchPhaser(BasePhaser):

    def __init__(self, model, loss, optim, device=None,
                 multi_batch=1, multi_batch_reduction='mean'):

        if multi_batch_reduction not in {'mean', 'sum'}:
            raise ValueError(f"invalid method: {multi_batch_reduction}")

        super().__init__(model, loss, optim, device)
        self.multi_batch = int(multi_batch)
        self.multi_batch_reduction = multi_batch_reduction

    def train(self, dataloader):
        self.model.train()

        n_batch = len(dataloader)
        n_remain = n_batch % self.multi_batch

        for i, (x, y) in enumerate(iterate(dataloader)):
            ŷ = self.model(x)
            l = self.loss(ŷ, y)

            if self.multi_batch > 1 and self.multi_batch_reduction == 'mean':
                l /= self.multi_batch if i < n_batch - n_remain else n_remain

            l.backward()

            if (i + 1) % self.multi_batch == 0:
                self.optim.step()
                self.optim.zero_grad()
                self.callback['train'](input=x, true=y, pred=ŷ, loss=l)

        if n_remain != 0:
            if dataloader.drop_last:
                self.optim.zero_grad()
            else:
                self.optim.step()
                self.optim.zero_grad()
                self.callback['train'](input=x, true=y, pred=ŷ, loss=l)
