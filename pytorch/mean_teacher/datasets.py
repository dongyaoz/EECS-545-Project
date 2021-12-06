# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import torchvision.transforms as transforms

from . import data
from .utils import export

# add svhn
@export
def svhn():                
    channel_stats = dict(mean=[x/255.0 for x in [111.60893667531344, 113.161274663812, 120.5651276685803]],
                         std=[x/255.0 for x in [50.49768174025085,  51.25898430316379,  50.24421613903954]])
    train_transformation = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    data_dir = 'data-local/images/svhn/by-image'

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 10
    }


@export
def cifar10():
    channel_stats = dict(mean = [0.485,0.456,0.406], 
                         std = [0.229,0.224,0.225])
    train_transformation = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data-local/images/cifar/cifar10/by-image',
        'num_classes': 10
    }
