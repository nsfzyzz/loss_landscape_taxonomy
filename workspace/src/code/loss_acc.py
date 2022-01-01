from __future__ import print_function

import sys
sys.path.insert(1, './code/')

import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F

from data import get_loader
from arguments import get_parser
from model import load_checkpoint 
from utils import *

parser = get_parser(code_type='loss_acc')
args = parser.parse_args()

softmax1 = nn.Softmax().cuda()

train_loader, test_loader = get_loader(args)

results = {}

if args.train_or_test == "train":
    eval_loader = train_loader
elif args.train_or_test == "test":
    eval_loader = test_loader

if not args.ensemble_average_acc:
    for exp_id in range(5):
        
        file_name = return_file_name_single(args, exp_id)
        model = load_checkpoint(args, file_name)
        results[exp_id] = test_acc_loss(eval_loader, model, nn.CrossEntropyLoss())
else:
    models = []
    for exp_id in range(5):
        file_name = return_file_name_single(args, exp_id)
        models.append(load_checkpoint(args, file_name))
    results['ensemble_average'] = test_ensemble_average(models, eval_loader)
        
f = open(args.result_location, "wb")
pickle.dump(results, f)
f.close()
