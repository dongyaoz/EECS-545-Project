import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


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



