import torch
from .utils import _convert, mgda_frank_wolfe_solver


class PhaserMGDA():

    def __init__(self, model_s, model_t, criterion, optimizer, device=None,
                 convert_x=None, convert_y=None):
        if device is not None:
            model_s.to(device)
            model_t.to(device)
            criterion.to(device)
            optimizer.load_state_dict(optimizer.state_dict())

        self.model_s = model_s
        self.model_t = model_t
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
        self.model_s.train()
        self.model_t.train()

        for x, y in dataloader:
            x = self.convert_x(x, device=self.device)
            y = self.convert_y(y, device=self.device)

            z = self.model_s(x)
            ź = z.detach().requires_grad_()

            ŷ = self.model_t(ź)
            l = self.criterion(ŷ, y)

            Δź = [None] * len(l)
            for t, l_t in enumerate(l):
                l_t.backward()
                Δź[t] = ź.grad.flatten()
                ź.grad = None
            Δź = torch.stack(Δź)

            α = mgda_frank_wolfe_solver(Δź).to(self.device)

            ŷ = self.model_t(z)
            l = self.criterion(ŷ, y)

            self.optimizer.zero_grad()
            torch.stack(l).dot(α).backward()
            self.optimizer.step()

            for func in self.callbacks['train']:
                func(input=x, true=y, pred=ŷ, loss=l, scale=α)

    @torch.no_grad()
    def valid(self, dataloader):
        self.model_s.eval()
        self.model_t.eval()

        for x, y in dataloader:
            x = self.convert_x(x, device=self.device)
            y = self.convert_y(y, device=self.device)

            ŷ = self.model_t(self.model_s(x))
            l = self.criterion(ŷ, y)

            for func in self.callbacks['valid']:
                func(input=x, true=y, pred=ŷ, loss=l)
