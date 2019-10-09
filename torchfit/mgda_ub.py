import torch
from .base import BasePhaser
from .utils import Optimizer, iterate, mgda_frank_wolfe_solver


class MgdaUbPhaser(BasePhaser):

    def __init__(self, model_enc, model_dec,
                 loss, optim_enc, optim_dec, device=None):

        model = torch.nn.Sequential(model_enc, model_dec)
        optim = Optimizer(optim_enc, optim_dec)

        super().__init__(model, loss, optim, device)
        self.model_enc = model_enc
        self.model_dec = model_dec

    def train(self, dataloader):
        self.model.train()

        for x, y in iterate(dataloader):
            z = self.model_enc(x)
            ź = z.detach().requires_grad_()

            ŷ = self.model_dec(ź)
            l = self.loss(ŷ, y)

            Δź = []
            for l_t in l:
                l_t.backward(retain_graph=True)
                Δź.append(ź.grad.flatten())
                ź.grad = None
            Δź = torch.stack(Δź)

            α = mgda_frank_wolfe_solver(Δź).to(self.device)

            ŷ = self.model_dec(z)
            l = self.loss(ŷ, y)

            self.optim.zero_grad()
            torch.stack(l).dot(α).backward()
            self.optim.step()

            self.callback['train'](input=x, true=y, pred=ŷ, loss=l, scale=α)
