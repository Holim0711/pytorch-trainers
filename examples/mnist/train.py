import torch
from model import *
from metric import Metric
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from configparser import ConfigParser

cfg = ConfigParser()
cfg.read('config.ini')
cfg_model = dict((k, eval(v)) for k, v in cfg['Model'].items())
cfg_loss = dict((k, eval(v)) for k, v in cfg['Loss'].items())
cfg_optim = dict((k, eval(v)) for k, v in cfg['Optim'].items())
cfg = dict((k, eval(v)) for k, v in cfg['Hyper'].items())

torch.manual_seed(cfg['random_seed'])

model = Model(**cfg_model)

loss = Loss(**cfg_loss)

optim = Optim(model.parameters(), lr=cfg['learning_rate'], **cfg_optim)

trainset = MNIST(".", train=True, transform=Transform(), download=True)
validset = MNIST(".", train=False, transform=Transform(), download=False)

trainloader = DataLoader(trainset, cfg['train_batch_size'], shuffle=True)
validloader = DataLoader(validset, cfg['valid_batch_size'], shuffle=False)

metric = Metric()
train_loss = []
valid_loss = []


from torchfit import BasePhaser
phaser = BasePhaser(model, loss, optim, device='cuda')

@phaser.after('train')
def after_train(input, pred, true, loss):
    train_loss.append(loss.item())

@phaser.after('valid')
def after_valid(input, pred, true, loss):
    valid_loss.append(loss.item())
    metric.update(pred, true)


from statistics import mean
from torch.utils.tensorboard import SummaryWriter


with SummaryWriter() as writer:
    config_string = '| param | value |\n| ----- | ----- |\n'
    config_string += '\n'.join(f'| {k} | {str(v)} |' for k, v in cfg.items())
    writer.add_text('configuration', config_string)
    writer.add_text('transform/train', str(trainloader.dataset.transform))
    writer.add_text('transform/valid', str(validloader.dataset.transform))
    writer.add_graph(model, next(iter(validloader))[0].to('cuda'))

    for epoch in range(20):
        phaser.train(trainloader)
        writer.add_scalar('loss/train', mean(train_loss), epoch)

        phaser.valid(validloader)
        writer.add_scalar('loss/valid', mean(valid_loss), epoch)

        acc, fig = metric.output()
        writer.add_scalar(f'accuracy', acc, epoch)
        writer.add_figure(f'confusion matrix', fig, epoch)
