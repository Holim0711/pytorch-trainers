import torch
from collections import defaultdict
from .utils import Optimizer, iterate


class BasePhaser():

    def __init__(self, model, loss, optim, device=None, verbose=True):
        if not isinstance(optim, Optimizer):
            optim = Optimizer(optim)
        if not isinstance(device, torch.device):
            device = torch.device(device)

        self.model = model
        self.loss = loss
        self.optim = optim
        self.device = device
        self.verbose = verbose

        if device is not None:
            self.model.to(device)
            self.loss.to(device)
            self.optim.reload_state_dict()

        self.callback = defaultdict(lambda **x: None)

    def after(self, phase):
        if phase not in {'train', 'valid'}:
            raise ValueError(f"invalid phase: {phase}")
        def register(func):
            self.callback[phase] = func
        return register

    def train(self, dataloader):
        self.model.train()
        for x, y in iterate(dataloader, self.device, self.verbose):
            ŷ = self.model(x)
            l = self.loss(ŷ, y)
            l.backward()

            self.optim.step()
            self.optim.zero_grad()
            self.callback['train'](input=x, true=y, pred=ŷ, loss=l)

    @torch.no_grad()
    def valid(self, dataloader):
        self.model.eval()
        for x, y in iterate(dataloader, self.device, self.verbose):
            ŷ = self.model(x)
            l = self.loss(ŷ, y)
            self.callback['valid'](input=x, true=y, pred=ŷ, loss=l)
