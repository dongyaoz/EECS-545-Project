# Implementation of Mean Teacher 
This Repo is the implementation of the following paper

* [Mean Teacher](https://arxiv.org/abs/1703.01780) Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results 


We used the Convlarge architecture to train Mean Teacher


### versions we use:
1. Pytorch 1.6.0
2. Python 3.7.3 (<3.8)
3. torchvision 0.7.0 
4. cudatoolkit 10.2
5. TensorboardX

also work: conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

 
### Dataset 
we didn't include dataset, but after runing the following shell, three folders under "...\data-local\images\cifar\cifar10\by-image" should be \train, \test, and \val

```
./data-local/bin/prepare_cifar10.sh
```

###  Accuracy Achieved on Test Dataset

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Mean Teacher
    a) Student Model : 84.4%
    b) Teacher Model : 87.07%
```


## Running the Training 
This repo implemented several methods and control them using flags.
### For Mean Teacher 
Go the parameters.py and change the following flags as follows:

1. supervised_mode = False ( To use only 4000 labels for training)
2. lr = 0.2  ( setting the learning rate)
3. BN = False or True  ( for turning batch Normalization on or off)
4. sntg = False ( Do not use any SNTG loss )
5. Do not change any other settings and run main.py


## Tensorboard Visualization
To Visualize on Tensorboard, use the following command 
```
tensorboard --logdir=”path to ./ckpt”
```
Note that all the checkpoints are in the ./ckpt folder so simply start a tensorboard session to visualize it. Also all the saved checkpoints for student models are also saved there.
```
1. Baseline : 12-03-18:09/convlarge,Adam,200epochs,b256,lr0.15/test
2. Mean teacher without BN :
   12-03-20:12/convlarge,Adam,200epochs,b256,lr0.15/test
   12-03-23:38/convlarge,Adam,200epochs,b256,lr0.2/test
3. Mean Teacher with BN : 12-05-11:55/convlarge,Adam,200epochs,b256,lr0.2/test
4. Hybrid Net : 12-06-10:58/hybridnet,Adam,200epochs,b256,lr0.2/test
5. SNTG + Meant Teacher: 12-07-00:36/convlarge,Adam,200epochs,b256,lr0.2/test
```

## Acknowledgments
Our implementation has been inspired from the following sources.

* [Mean Teacher](https://github.com/iSarmad/MeanTeacher-SNTG-HybridNet) : We have mainly followed this Repo, but 
* [SNTG](https://github.com/xinmei9322/SNTG) - I have understood the concept of SNTG and converted Theano Implementation to Pytorch
* [Hybrid Network](https://github.com/dakshitagrawal97/HybridNet) - I have followed this repository to incorporate reconstruction loss in my implementation. 
