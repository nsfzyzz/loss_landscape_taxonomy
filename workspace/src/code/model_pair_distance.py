from __future__ import print_function

import sys
sys.path.insert(1, './code/')

import numpy as np
import pickle
import matplotlib.pyplot as plt

from arguments import get_parser
from model import load_checkpoint 
from utils import *


def get_params(model): 
    # wrt data at the current step
    res = []
    for p in model.parameters():
        if p.requires_grad:
            res.append(p.data.view(-1))
    weight_flat = torch.cat(res)
    return weight_flat

def compute_distance(model1, model2):
    
    params1 = get_params(model1)
    params2 = get_params(model2)
    dist = (params1-params2).norm().item()
    
    return dist


parser = get_parser(code_type='model_dist')
args = parser.parse_args()

model_distance = {}

for exp_id1 in range(5):
    
    model_distance[exp_id1] = {}
    
    for exp_id2 in range(5):
        
        file_name1, file_name2 = return_file_name(args, exp_id1, exp_id2)
        
        model1 = load_checkpoint(args, file_name1)
        model2 = load_checkpoint(args, file_name2)
        
        model_distance[exp_id1][exp_id2] = {'dist': compute_distance(model1, model2)}
        
        temp_results = {'model_distance': model_distance}
        
        f = open(args.result_location, "wb")
        pickle.dump(temp_results, f)
        f.close()
        
        
        