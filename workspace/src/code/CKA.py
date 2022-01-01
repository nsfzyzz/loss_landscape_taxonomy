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
from CKA_utils import *

parser = get_parser(code_type='CKA')
args = parser.parse_args()

# Get data
train_loader, test_loader = get_loader(args)

representation_similarity = {}
classification_similarity = {}
cos = nn.CosineSimilarity(dim=0)

for exp_id1 in range(5):
    
    representation_similarity[exp_id1] = {}
    classification_similarity[exp_id1] = {}
    
    for exp_id2 in range(5):
        
        file_name1, file_name2 = return_file_name(args, exp_id1, exp_id2)
        
        model1 = load_checkpoint(args, file_name1)
        model2 = load_checkpoint(args, file_name2)
        
        if args.train_or_test == "train":
            eval_loader = train_loader
        elif args.train_or_test == "test":
            eval_loader = test_loader
        else:
            raise ValueError('Invalid input.')
        
        if not args.compare_classification:
            
            cka_from_features_average = []
            
            for CKA_repeat_runs in range(args.CKA_repeat_runs):

                cka_from_features = []

                latent_all_1, latent_all_2 = all_latent(model1, model2, eval_loader, num_batches = args.CKA_batches, args=args)

                for name in latent_all_1.keys():

                    print(name)

                    if args.flattenHW:
                        cka_from_features.append(feature_space_linear_cka(latent_all_1[name], latent_all_2[name]))
                    else:
                        cka_from_features.append(cka_compute(gram_linear(latent_all_1[name]), gram_linear(latent_all_2[name])))
                        
                cka_from_features_average.append(cka_from_features)
                
            cka_from_features_average = np.mean(np.array(cka_from_features_average), axis=0)
            
            print('cka_from_features shape')
            print(cka_from_features_average.shape)

            representation_similarity[exp_id1][exp_id2] = cka_from_features_average
        
        else:
            classification_similarity[exp_id1][exp_id2] = compare_classification(model1, model2, eval_loader, args=args, cos=cos)
        
        temp_results = {'representation_similarity': representation_similarity, 'classification_similarity': classification_similarity}
        
        f = open(args.result_location, "wb")
        pickle.dump(temp_results, f)
        f.close()
        
        
        