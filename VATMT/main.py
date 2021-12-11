import argparse
from torchvision import datasets, transforms
import torch.optim as optim
from model import *
from utils import *
import os

from Datasets import data
import checkpoint

import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.datasets

from Datasets.data import NO_LABEL
import datetime
import models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# batch_size = 32
batch_size = 256
eval_batch_size = 100
unlabeled_batch_size = 128
num_labeled = 4000
num_valid = 1000
num_iter_per_epoch = 400
eval_freq = 5
lr = 0.001
cuda_device = "0"


global global_step
global_step = 0

parser = argparse.ArgumentParser()
parser.add_argument('--BN', default=True, help='Use Batch Normalization? ')
parser.add_argument('--dataset', default='cifar10', help='cifar10 | svhn')
parser.add_argument('--dataroot', default='cifar10', help='path to dataset')
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--num_epochs', type=int, default=120)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--epsilon', type=float, default=2.5)
parser.add_argument('--top_bn', type=bool, default=True)
parser.add_argument('--method', default='vat')
parser.add_argument('--sntg', default=False, help='Use SNTG loss?')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--labeled-batch-size', default=100, type=int,#100
                        metavar='N', help="labeled examples per minibatch (default: no constrain)")                        
parser.add_argument('--model',default='convlarge', help='Basically using Convlarge for all experiments')


opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

def tocuda(x):
    if opt.use_cuda:
        return x.cuda()
    return x

def d_mse(p1,p2):
    loss = nn.MSELoss()
    mse_loss = loss(p1, p2)
    return mse_loss

def _l2_normalize(d):

    d = d.numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)

def entropy_loss(ul_y):
    p = F.softmax(ul_y, dim=1)
    return -(p*F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)   

def vat_loss(model2, ema_model2, ul_x, ul_y, xi=1e-6, eps=2.5, num_iters=1):

    # find r_adv

    d = torch.Tensor(ul_x.size()).normal_()
    for i in range(num_iters):
        d = xi *_l2_normalize(d)
        d = Variable(d.cuda(), requires_grad=True)
        y_hat = ema_model2(ul_x + d)
        mse_loss = d_mse(ul_y.detach(), y_hat)
        mse_loss.backward()

        d = d.grad.data.clone().cpu()
        model2.zero_grad()

    d = _l2_normalize(d)
    d = Variable(d.cuda())
    r_adv = eps *d
    # compute lds
    y_hat = ema_model2(ul_x + r_adv.detach())
    mse_loss = d_mse(ul_y.detach(), y_hat)

    return mse_loss

def train(model, model2, ema_model, ema_model2, x, y, ul_x, optimizer1, optimizer2):
    global global_step
    model.train()
    ema_model.train()
    model2.train()
    ema_model2.train()

    ce = nn.CrossEntropyLoss()
    y_pred = ema_model(x) #TODO: model(x)
    ce_loss = ce(y_pred, y)

    ul_y = ema_model(ul_x) #TODO: model(x)
    mse_loss = vat_loss(model2, ema_model2, ul_x, ul_y, eps=opt.epsilon)
    loss = mse_loss + ce_loss
    if opt.method == 'vatent':
        loss += entropy_loss(ul_y)

    optimizer1.zero_grad()
    optimizer2.zero_grad()
    loss.backward() # update student model
    optimizer1.step()
    optimizer2.step()
    global_step += 1
    
    # update teacher model
    update_ema_variables(model, ema_model, opt.ema_decay, global_step)
    update_ema_variables(model2, ema_model2, opt.ema_decay, global_step)

    return mse_loss, ce_loss


def eval(model, x, y):

    y_pred = model(x)
    prob, idx = torch.max(y_pred, dim=1)
    return torch.eq(idx, y).float().mean()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)


if opt.dataset == 'svhn':
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root=opt.dataroot, split='train', download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root=opt.dataroot, split='test', download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ])),
        batch_size=eval_batch_size, shuffle=True)

elif opt.dataset == 'cifar10':
    num_labeled = 4000
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=opt.dataroot, train=True, download=True,
                      transform=transforms.Compose([
                          data.RandomTranslateWithReflect(4),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=opt.dataroot, train=False, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ])),
        batch_size=eval_batch_size, shuffle=True)

else:
    raise NotImplementedError

train_data = []
train_target = []

for (data, target) in train_loader:
    train_data.append(data)
    train_target.append(target)

train_data = torch.cat(train_data, dim=0)
train_target = torch.cat(train_target, dim=0)

valid_data, train_data = train_data[:num_valid, ], train_data[num_valid:, ]
valid_target, train_target = train_target[:num_valid], train_target[num_valid:, ]

labeled_train, labeled_target = train_data[:num_labeled, ], train_target[:num_labeled, ]
unlabeled_train = train_data[num_labeled:, ]

#  Intializing the model
model = models.__dict__[opt.model](opt, data=None).cuda()
ema_model = models.__dict__[opt.model](opt,nograd = True, data=None).cuda()
model.apply(weights_init)
ema_model.apply(weights_init)
optimizer1 = optim.Adam(model.parameters(), lr=lr)
# optimizer = optim.Adam(ema_model.parameters(), lr=lr)

model2 = models.__dict__[opt.model](opt, data=None).cuda()
ema_model2 = models.__dict__[opt.model](opt,nograd = True, data=None).cuda()
model2.apply(weights_init)
ema_model2.apply(weights_init)
optimizer2 = optim.Adam(model2.parameters(), lr=lr)
# optimizer = optim.Adam(ema_model2.parameters(), lr=lr)

# Attempts to restore the latest checkpoint if exists
print('Loading model...')
model, start_epoch = checkpoint.restore_checkpoint(model, 'checkpoints/{}'.format(opt.dataset))

# train the network
for epoch in range(start_epoch, opt.num_epochs):

    if epoch > opt.epoch_decay_start:
        decayed_lr = (opt.num_epochs - epoch) * lr / (opt.num_epochs - opt.epoch_decay_start)
        optimizer1.lr = decayed_lr
        optimizer1.betas = (0.5, 0.999)        
        optimizer2.lr = decayed_lr
        optimizer2.betas = (0.5, 0.999)

    for i in range(num_iter_per_epoch):

        batch_indices = torch.LongTensor(np.random.choice(labeled_train.size()[0], batch_size, replace=False))
        x = labeled_train[batch_indices]
        y = labeled_target[batch_indices]
        batch_indices_unlabeled = torch.LongTensor(np.random.choice(unlabeled_train.size()[0], unlabeled_batch_size, replace=False))
        ul_x = unlabeled_train[batch_indices_unlabeled]

        v_loss, ce_loss = train(model.train(),model2.train(), ema_model.train(), ema_model2.train(), Variable(tocuda(x)), Variable(tocuda(y)), Variable(tocuda(ul_x)),
                                optimizer1, optimizer2)

        if i % 100 == 0:
            # print(v_loss.item(), ce_loss.item())
            # print("Epoch :", epoch, "Iter :", i, "VAT Loss :", v_loss.data[0], "CE Loss :", ce_loss.data[0])
            print("Epoch :", epoch+1, "Iter :", i, "VAT Loss :", v_loss.item(), "CE Loss :", ce_loss.item())

    if epoch % eval_freq == 0 or epoch + 1 == opt.num_epochs:

        batch_indices = torch.LongTensor(np.random.choice(labeled_train.size()[0], batch_size, replace=False))
        x = labeled_train[batch_indices]
        y = labeled_target[batch_indices]
        train_accuracy = eval(model.eval(), Variable(tocuda(x)), Variable(tocuda(y)))
        # print("Train accuracy :", train_accuracy.data[0])
        print("Train accuracy :", train_accuracy.item())

        for (data, target) in test_loader:
            test_accuracy = eval(model.eval(), Variable(tocuda(data)), Variable(tocuda(target)))
            # print("Test accuracy :", test_accuracy.data[0])
            print("Test accuracy :", test_accuracy.item())
            break

    # Save checkpoint
    checkpoint.save_checkpoint(model, epoch+1, 'checkpoints/{}'.format(opt.dataset))


test_accuracy = 0.0
counter = 0
for (data, target) in test_loader:
    n = data.size()[0]
    acc = eval(model.eval(), Variable(tocuda(data)), Variable(tocuda(target)))
    test_accuracy += n*acc
    counter += n

# print("Full test accuracy :", test_accuracy.data[0]/counter)
print("Full test accuracy :", test_accuracy.item()/counter)