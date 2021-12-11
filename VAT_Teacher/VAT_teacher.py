import argparse
from torchvision import datasets, transforms
import torch.optim as optim
# from model import *
# from VAT_MT_util import *
import os

from Datasets import data
import checkpoint

import os
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
# from misc.utils import *
# from tensorboardX import SummaryWriter
import datetime
# from parameters import get_parameters
import models

# from misc import ramps
from Datasets import data
# from models import losses

import torchvision.transforms as transforms


# batch_size = 32
batch_size = 256
eval_batch_size = 100
unlabeled_batch_size = 128
num_labeled = 1000
num_valid = 1000
num_iter_per_epoch = 400
eval_freq = 5
lr = 0.001
cuda_device = "0"
global global_step
global_step = 0


parser = argparse.ArgumentParser()
parser.add_argument('--BN', default=True, help='Use Batch Normalization? ')
parser.add_argument('--dataset', required=True, help='cifar10 | svhn')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--num_epochs', type=int, default=120)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--epsilon', type=float, default=2.5)
parser.add_argument('--top_bn', type=bool, default=True)
parser.add_argument('--method', default='vat')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--labeled-batch-size', default=100, type=int,#100
                        metavar='N', help="labeled examples per minibatch (default: no constrain)")                        
parser.add_argument('--model',default='convlarge', help='Basically using Convlarge for all experiments')


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

def tocuda(x):
    if args.use_cuda:
        return x.cuda()
    return x

def _l2_normalize(d):

    d = d.numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)

def fint_vap_x(ul_x, xi=1e-6, eps=2.5, num_iters=1):

    # find x+pertubation

    d = torch.Tensor(ul_x.size()).normal_()
    for i in range(num_iters):
        d = xi *_l2_normalize(d)
        d = Variable(d.cuda(), requires_grad=True)
        y_hat = ema_model(ul_x + d)
        d = d.grad.data.clone().cpu()
        ema_model.zero_grad()

    d = _l2_normalize(d)
    d = Variable(d.cuda())
    r_adv = eps *d
    # compute lds
    y_hat = ema_model(ul_x + r_adv.detach())
    return y_hat


def entropy_loss(ul_y):
    p = F.softmax(ul_y, dim=1)
    return -(p*F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)


def d_mse(p1,p2):
    loss = nn.MSELoss()
    mse_loss = loss(p1, p2)
    return mse_loss

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)   

def train(x, y, optimizer):
    # model: find student_model output
    # ema_model: find teacher_model output
    # y: baseline y
    # y_pred: emaout1, which is the student model output of baseline x
    global global_step
    # Ls: cross entropy loss
    ce = nn.CrossEntropyLoss()
    emaout1 = model(x)
    ce_loss = ce(emaout1, y) 

    # find x + perterbation
    x_vap = fint_vap_x(x)
    emaout2 = model2(x_vap)

     # get teacher model output
    out1 = ema_model(emaout1)
    out2 = ema_model2(emaout2)
    # d_mse
    d_loss = d_mse(out1, out2)
    loss = d_loss + ce_loss
    if args.method == 'vatent':
        loss += entropy_loss(out2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    global_step += 1
    update_ema_variables(model, ema_model, args.ema_decay, global_step)
    update_ema_variables(model2, ema_model2, args.ema_decay, global_step)
    return d_loss, ce_loss


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


if args.dataset == 'svhn':
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root=args.dataroot, split='train', download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root=args.dataroot, split='test', download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ])),
        batch_size=eval_batch_size, shuffle=True)

elif args.dataset == 'cifar10':
    num_labeled = 4000
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=args.dataroot, train=True, download=True,
                      transform=transforms.Compose([
                          data.RandomTranslateWithReflect(4),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=args.dataroot, train=False, download=True,
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
model = models.__dict__[args.model](args, data=None).cuda()
ema_model = models.__dict__[args.model](args,nograd = True, data=None).cuda()
model.apply(weights_init)
ema_model.apply(weights_init)
optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = optim.Adam(ema_model.parameters(), lr=lr)

model2 = models.__dict__[args.model](args, data=None).cuda()
ema_model2 = models.__dict__[args.model](args,nograd = True, data=None).cuda()
model2.apply(weights_init)
ema_model2.apply(weights_init)
optimizer = optim.Adam(model2.parameters(), lr=lr)
optimizer = optim.Adam(ema_model2.parameters(), lr=lr)

# Attempts to restore the latest checkpoint if exists
print('Loading model...')
model, start_epoch = checkpoint.restore_checkpoint(model, 'checkpoints/{}'.format(args.dataset))

# train the network
for epoch in range(start_epoch, args.num_epochs):

    if epoch > args.epoch_decay_start:
        decayed_lr = (args.num_epochs - epoch) * lr / (args.num_epochs - args.epoch_decay_start)
        optimizer.lr = decayed_lr
        optimizer.betas = (0.5, 0.999)

    for i in range(num_iter_per_epoch):

        batch_indices = torch.LongTensor(np.random.choice(labeled_train.size()[0], batch_size, replace=False))
        x = labeled_train[batch_indices]
        y = labeled_target[batch_indices]
        batch_indices_unlabeled = torch.LongTensor(np.random.choice(unlabeled_train.size()[0], unlabeled_batch_size, replace=False))
        ul_x = unlabeled_train[batch_indices_unlabeled]

        # v_loss, ce_loss = train(model.train(), Variable(tocuda(x)), Variable(tocuda(y)), Variable(tocuda(ul_x)),
        #                         optimizer)
        v_loss, ce_loss = train(model.train(), Variable(tocuda(x)), Variable(tocuda(y)), Variable(tocuda(ul_x)),
                                 optimizer)

        if i % 100 == 0:
            # print(v_loss.item(), ce_loss.item())
            # print("Epoch :", epoch, "Iter :", i, "VAT Loss :", v_loss.data[0], "CE Loss :", ce_loss.data[0])
            print("Epoch :", epoch+1, "Iter :", i, "VAT Loss :", v_loss.item(), "CE Loss :", ce_loss.item())

    if epoch % eval_freq == 0 or epoch + 1 == args.num_epochs:

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
    checkpoint.save_checkpoint(model, epoch+1, 'checkpoints/{}'.format(args.dataset))


test_accuracy = 0.0
counter = 0
for (data, target) in test_loader:
    n = data.size()[0]
    acc = eval(model.eval(), Variable(tocuda(data)), Variable(tocuda(target)))
    test_accuracy += n*acc
    counter += n

# print("Full test accuracy :", test_accuracy.data[0]/counter)
print("Full test accuracy :", test_accuracy.item()/counter)
