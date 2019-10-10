from torch.optim import Optimizer as torch_optim


class Optimizer(list):

    def __init__(self, *args):
        if any(not isinstance(x, torch_optim) for x in args):
            raise ValueError("Not a torch.optim.optimizer")
        super().__init__(args)

    def zero_grad(self):
        for x in self:
            x.zero_grad()

    def step(self):
        for x in self:
            x.step()

    def reload_state_dict(self):
        for x in self:
            x.load_state_dict(x.state_dict())
