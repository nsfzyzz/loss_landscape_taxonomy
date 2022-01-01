from __future__ import print_function

import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets, transforms
from torch.autograd import Variable

import sys
sys.path.insert(1, './code/')
from data import get_loader
from arguments import get_parser
from models.resnet_cifar import resnet20_cifar
from models.resnet_depth import resnet_depth
import pickle
from utils import *
from tqdm import tqdm, trange
from pyhessian import hessian

import logging
import os

from torch.utils.data import TensorDataset

parser = get_parser(code_type='hessian')
args = parser.parse_args()
for arg in vars(args):
    print(arg, getattr(args, arg))

torch.manual_seed(args.seed)

arch_kwargs = {}

if args.different_width:
    from models.resnet_width import ResNet18
    from models.vgg_width import *
    from models.densenet40 import DenseNet3
    
    arch_kwargs = {'width': args.resnet18_width}
    if args.cifar100:
        arch_kwargs = {'width': args.resnet18_width, 'num_classes': 100}
    DenseNet40_kwargs = {'depth': 40, 'num_classes': 10, 'growth_rate': args.resnet18_width}
    
else:
    from models.resnet import ResNet18
    from models.vgg import *



# Get data

args.train_bs = args.mini_hessian_batch_size
args.test_bs = args.mini_hessian_batch_size

train_loader, test_loader, _ = get_loader(args)

if args.train_or_test == 'train':
    eval_loader = train_loader
elif args.train_or_test == 'test':
    eval_loader = test_loader

def return_model(file_name, arch_kwargs={}):

    checkpoint = torch.load(file_name)
    
    if args.arch == 'resnet20':  
        model = resnet20_cifar().cuda()
    elif 'ResNet18' in args.arch:
        model = ResNet18(**arch_kwargs).cuda()
    elif 'VGG11' in args.arch:
        model = VGG('VGG11', **arch_kwargs).cuda()
    elif 'DenseNet40' in args.arch:
        model = DenseNet3(**DenseNet40_kwargs).cuda()
        
    elif args.different_depth:
        model = resnet_depth(depth=args.depth, base_channel=args.resnet18_width).cuda()
    
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint)
    
    return model


criterion = nn.CrossEntropyLoss()  # label loss

######################################################
# Begin the computation
######################################################

hessian_result = {}

# turn model to eval mode
for exp_id in range(args.exp_num):

    # Hessian prepare steps
    
    assert (args.hessian_batch_size % args.mini_hessian_batch_size == 0)
    assert (50000 % args.hessian_batch_size == 0)
    batch_num = args.hessian_batch_size // args.mini_hessian_batch_size

    if batch_num == 1:
        for inputs, labels in eval_loader:
            hessian_dataloader = (inputs, labels)
            break
    else:
        hessian_dataloader = []
        for i, (inputs, labels) in enumerate(eval_loader):
            hessian_dataloader.append((inputs, labels))
            if i == batch_num - 1:
                break
            
    file_name = os.path.join(args.checkpoint_folder, f"net_exp_{exp_id}.pkl")
    es_file_name = os.path.join(args.checkpoint_folder, f"net_exp_{exp_id}_early_stopped_model.pkl")
    best_file_name = os.path.join(args.checkpoint_folder, f"net_exp_{exp_id}_best.pkl")
    if args.early_stopping:
        if os.path.exists(es_file_name):
            file_name = es_file_name
        elif os.path.exists(best_file_name):
            file_name = best_file_name
        print("use model {0}".format(file_name))
        
    print(f'********** start the experiment on model {file_name} **********')
        
    model = return_model(file_name, arch_kwargs = arch_kwargs)
    model.eval()
    if batch_num == 1:
        hessian_comp = hessian(model,
                               criterion,
                               data=hessian_dataloader,
                               cuda=args.cuda)
    else:
        hessian_comp = hessian(model,
                               criterion,
                               dataloader=hessian_dataloader,
                               cuda=args.cuda)

    print('********** finish data londing and begin Hessian computation **********')

    top_eigenvalues, _ = hessian_comp.eigenvalues()
    trace = hessian_comp.trace()

    print('\n***Top Eigenvalues: ', top_eigenvalues)
    print('\n***Trace: ', np.mean(trace))

    hessian_result[exp_id] = {'top_eigenvalue': top_eigenvalues, 'trace': np.mean(trace)}

f = open(args.result_location, "wb")
pickle.dump(hessian_result, f)
f.close()
    
print("Save results complete!!!")

