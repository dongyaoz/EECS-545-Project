import argparse
from torchvision import datasets, transforms
import torch.optim as optim
from model import *
from utils import *
import os
import matplotlib.pyplot as plt

# import main

from Datasets import data
import checkpoint

epoch_list = []
trainacc_list = []
testacc_list = []

# batch_size = 64
batch_size = 16
# eval_batch_size = 100
eval_batch_size = 100
# unlabeled_batch_size = 128
unlabeled_batch_size = 128
num_labeled = 1000
num_valid = 1000

def tocuda(x):
    # if opt.use_cuda:
    return x.cuda()

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

num_labeled = 4000
#     train_loader = torch.utils.data.DataLoader(
#         datasets.CIFAR10(root=opt.dataroot, train=True, download=True,
#                       transform=transforms.Compose([
#                           transforms.ToTensor(),
#                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                       ])),
#         batch_size=batch_size, shuffle=True)
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='cifar10', train=True, download=True,
                      transform=transforms.Compose([
                          data.RandomTranslateWithReflect(4),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='cifar10', train=False, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ])),
    batch_size=eval_batch_size, shuffle=True)

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

for i in np.arange(1, 386):
    model = tocuda(VAT(True))
    model.apply(weights_init)
    model = checkpoint.restore_checkpointn(model, 'checkpoints/{}'.format('cifar10'), i)

    batch_indices = torch.LongTensor(np.random.choice(labeled_train.size()[0], batch_size, replace=False))
    x = labeled_train[batch_indices]
    y = labeled_target[batch_indices]
    train_accuracy = eval(model.eval(), Variable(tocuda(x)), Variable(tocuda(y)))
    # print("Train accuracy :", train_accuracy.data[0])
    print("Epoch {}".format(i))
    print("Train accuracy :", train_accuracy.item())

    for (data, target) in test_loader:
        test_accuracy = eval(model.eval(), Variable(tocuda(data)), Variable(tocuda(target)))
        # print("Test accuracy :", test_accuracy.data[0])
        print("Test accuracy :", test_accuracy.item())
        break
    epoch_list.append(i)
    trainacc_list.append(train_accuracy.item())
    testacc_list.append(test_accuracy.item())

plt.figure()
plt.plot(epoch_list, trainacc_list, label='Train')
plt.plot(epoch_list, testacc_list, label='Test')
plt.xlabel('Epoch number')
plt.ylabel('Accuracy')
plt.legend()
plt.show()