import torch
from ..base import BaseTrainer
from .fw_solver import frank_wolfe_solver


class MGDATrainer(BaseTrainer):

    def __init__(self, encoder, decoder, criterion, optimizer, device=None):
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.train_callback = None
        self.valid_callback = None

    def _train(self, i, x, y):
        z = self.encoder(x)
        ź = z.detach().requires_grad_()

        ŷ = self.decoder(ź)
        ℓ = self.criterion(ŷ, y)

        Δź = []
        for ℓ_t in ℓ:
            ℓ_t.backward(retain_graph=True)
            Δź.append(ź.grad.flatten())
            ź.grad = None
        Δź = torch.stack(Δź)

        α = frank_wolfe_solver(Δź).to(self.device)

        ŷ = self.decoder(z)
        ℓ = self.criterion(ŷ, y)

        self.optimizer.zero_grad()
        torch.stack(ℓ).dot(α).backward()
        self.optimizer.step()

        if self.train_callback is not None:
            self.train_callback(
                i=i, input=x, output=ŷ, target=y, loss=ℓ, scale=α)
