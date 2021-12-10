import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


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
        y_hat = get_teacher(ul_x + d)
        d = d.grad.data.clone().cpu()
        get_teacher.zero_grad()

    d = _l2_normalize(d)
    d = Variable(d.cuda())
    r_adv = eps *d
    # compute lds
    y_hat = get_teacher(ul_x + r_adv.detach())
    return y_hat


def entropy_loss(ul_y):
    p = F.softmax(ul_y, dim=1)
    return -(p*F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)


def d_mse(p1,p2):
    loss = nn.MSELoss()
    mse_loss = loss(p1, p2)
    return mse_loss