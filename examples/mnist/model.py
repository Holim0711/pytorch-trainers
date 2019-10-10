import torch.nn as nn
import torch.nn.functional as nf
from torch.nn import NLLLoss as Loss
from torch.optim import SGD as Optim
from torchvision.transforms import Compose, ToTensor, Normalize


__all__ = [
    'cfg',
    'Transform',
    'Model',
    'Loss',
    'Optim',
]


cfg = {
    'train_batch_size': 64,
    'valid_batch_size': 1000,
    'learning_rate': 1e-2,
    'model': {},
    'loss': {},
    'optim': {},
}


class Transform(Compose):
    def __init__(self):
        super().__init__([
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
        ])


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = nf.relu(nf.max_pool2d(self.conv1(x), 2))
        x = nf.relu(nf.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = nf.relu(self.fc1(x))
        x = nf.dropout(x, training=self.training)
        x = self.fc2(x)
        return nf.log_softmax(x, dim=-1)