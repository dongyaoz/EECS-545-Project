
from torch.nn import functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def softmax_mse_loss(input_logits, target_logits):

    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, reduction='sum') / num_classes



def symmetric_mse_loss(input1, input2):

    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2)**2) / num_classes

# def kl_div_with_logit(q_logit, p_logit):

#     q = F.softmax(q_logit, dim=1)
#     logq = F.log_softmax(q_logit, dim=1)
#     logp = F.log_softmax(p_logit, dim=1)

#     qlogq = ( q *logq).sum(dim=1).mean(dim=0)
#     qlogp = ( q *logp).sum(dim=1).mean(dim=0)

#     return qlogq - qlogp


def _l2_normalize(d):

    d = d.numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)

# def d_mse(p1,p2):
#     loss = nn.MSELoss()
#     mse_loss = loss(p1, p2)
#     return mse_loss


def vat_loss(model, ul_x, ul_y, xi=1e-6, eps=2.5, num_iters=1):

    # find r_adv

    d = torch.Tensor(ul_x.size()).normal_()
    for i in range(num_iters):
        d = xi *_l2_normalize(d)
        d = Variable(d.cuda(), requires_grad=True)
        y_hat = model(ul_x + d)
        delta_kl = softmax_mse_loss(ul_y.detach(), y_hat)
        delta_kl.backward()

        d = d.grad.data.clone().cpu()
        model.zero_grad()

    d = _l2_normalize(d)
    d = Variable(d.cuda())
    r_adv = eps *d
    # compute lds
    y_hat = model(ul_x + r_adv.detach())
    delta_kl = softmax_mse_loss(ul_y.detach(), y_hat)
    return delta_kl


def entropy_loss(ul_y):
    p = F.softmax(ul_y, dim=1)
    return -(p*F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)

