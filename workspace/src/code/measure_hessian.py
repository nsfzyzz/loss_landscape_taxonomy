"""The following code is adapted from 
PyHessian: Neural Networks Through the Lens of the Hessian
Z. Yao, A. Gholami, K Keutzer, M. Mahoney
https://github.com/amirgholami/PyHessian
"""

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

from models.resnet_width import ResNet18
arch_kwargs = {'width': args.resnet18_width}

# Get data

args.train_bs = args.mini_hessian_batch_size
args.test_bs = args.mini_hessian_batch_size

train_loader, test_loader = get_loader(args)

if args.train_or_test == 'train':
    eval_loader = train_loader
elif args.train_or_test == 'test':
    eval_loader = test_loader

def return_model(file_name, arch_kwargs={}):

    checkpoint = torch.load(file_name)
    
    model = ResNet18(**arch_kwargs).cuda()
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

