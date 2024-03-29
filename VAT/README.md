# Virtual Adversarial Training
<img src="https://github.com/9310gaurav/virtual-adversarial-training/blob/master/vat.gif" width="480">
Pytorch implementation of "Virtual Adversarial Training: a Regularization Method for Supervised and Semi-Supervised Learning" http://arxiv.org/abs/1704.03976

## For reproducing semi-supervised learning results for SVHN with VAT loss:
```python main.py --dataroot=<dataroot> --dataset=svhn --method=vat```
  
## For reproducing semi-supervised learning results for CIFAR10 with VAT loss:
```python main.py --dataroot=<dataroot> --dataset=cifar10 --method=vat --num_epochs=500 --epoch_decay_start=460 --epsilon=10.0 --top_bn=False```

## For reproducing semi-supervised learning results for SVHN with VAT loss + Entropy loss:
```python main.py --dataroot=<dataroot> --dataset=svhn --method=vatent```
  
## For reproducing semi-supervised learning results for CIFAR10 with VAT loss + Entropy loss:
```python main.py --dataroot=<dataroot> --dataset=cifar10 --method=vatent --num_epochs=500 --epoch_decay_start=460 --epsilon=10.0 --top_bn=False```

## To run ours
```python main.py --dataroot=cifar --dataset=cifar10 --method=vat --num_epochs=370 --epoch_decay_start=385 --epsilon=10.0 --top_bn=False```

checkpoint.py is used for saving the checkpoint and load the latest saved model (if exist) for training.

test.py is used for getting a plot for train and test accurcacy by loading and evaluating using saved models.
