from __future__ import print_function

import argparse

def get_parser(code_type='training'):

    parser = argparse.ArgumentParser(description='Arguments for this experiment')
    
    #######################
    # data loader
    #######################
    
    if code_type in ['training', 'CKA', 'hessian', 'curve', 'loss_acc', 'ntk']:
        
        parser.add_argument('--train-bs', type=int, default = 128, help='The training batch size')
        parser.add_argument('--test-bs', type=int, default = 128, help='The testing batch size')
        parser.add_argument('--training-type', type=str, default = 'normal', help='training type')

        # for training with random labels
        parser.add_argument('--random-labels', dest='random_labels', default = False, action='store_true', help='train using random labels')
        parser.add_argument('--shuffle-random-data', dest='shuffle_random_data', default = False, action='store_true', help='shuffle random data during training')
        parser.add_argument('--num-classes', type=int, default=10)
        parser.add_argument('--label-corrupt-prob', type=float, default=1.0)
        parser.add_argument('--random-label-path', type=str, default='../../data/random_labels/random_label.pkl')
        parser.add_argument('--random-label-path-test', type=str, default='../../data/random_labels/random_label_test.pkl')
        parser.add_argument('--test-on-noise', dest='test_on_noise', default = False, action='store_true', help='change test data to have noisy labels also')

        # for training with a subset of data
        parser.add_argument('--data-subset', dest='data_subset', default = False, action='store_true', help='train using a subset of data')
        parser.add_argument('--subset', dest='subset', type=float, default = 1.0, help='the percentage of data used')
        parser.add_argument('--subset-noisy', dest = 'subset_noisy', default = False, action = 'store_true', help = 'use noisy labels on subset of data')        
        

    #######################
    # network architecture
    #######################
    
    if code_type in ['training', 'CKA', 'hessian', 'model_dist', 'curve', 'loss_acc', 'ntk']:
    
        parser.add_argument('--arch', type=str, default = 'resnet110', help='Model architecture')

        # training with different width
        parser.add_argument('--different-width', dest='different_width', default = False, action='store_true', help='training with resnet18 of different width')
        parser.add_argument('--resnet18-width', dest='resnet18_width', type=int, default = 64, help='Width of resnet18')
        
        # how many models are we training
        parser.add_argument('--exp-num', dest='exp_num', type=int, default = 5, help='how many experiments are we performing')
    
    #######################
    # training parameters
    #######################
    
    if code_type in ['training', 'curve']:
        
        parser.add_argument('--lr', type=float, default = 0.1, help='Learning rate')
        parser.add_argument('--epochs', type=int, default = 200, help='The number of epochs to train model')
        parser.add_argument('--weight-decay', type=float, default = 5e-4, help='Weight decay')
        parser.add_argument('--no-lr-decay', dest='no_lr_decay', default = False, action='store_true', help='no learning rate decay')
        parser.add_argument('--one-lr-decay', dest='one_lr_decay', default = False, action='store_true', help='one learning rate decay')
        parser.add_argument('--stop-epoch', type=int, default = 100, help='Stopping epoch number if there is no learning rate decay')
        parser.add_argument('--resume', type=str, default=None, help='choose model')
        parser.add_argument('--lr-decay-epoch', type=int, nargs='+', default=[100,150], help='Decrease learning rate at these epochs.')

        # for early stop
        parser.add_argument('--save-early-stop', dest='save_early_stop', default = False, action='store_true', help='save a checkpoint if the loss does not improve by an amount of delta')
        parser.add_argument('--min-delta', type=float, default = 0, help='amount improvement')
        parser.add_argument('--patience', type=int, default = 0, help='number of epochs to wait if loss does not improve by min_delta')

        # choosing training procedure
        parser.add_argument('--ignore-incomplete-batch', dest='ignore_incomplete_batch', default = False, action='store_true', help='ignore the last incomplete batch during training')
        parser.add_argument('--only-exploration', dest='only_exploration', default = False, action='store_true', help='train only before lr decay')
        parser.add_argument('--save-final', dest='save_final', default = False, action='store_true', help='store the final model')
        parser.add_argument('--save-middle', dest='save_middle', default = False, action='store_true', help='store the intermediate trained models')
        parser.add_argument('--save-best', dest='save_best', default = False, action='store_true', help='store the model with the best training loss')
        parser.add_argument('--save-frequency', default=10, type=int, help='which epochs to save')

        parser.add_argument('--saving-folder', dest='saving_folder', type=str, default = "", help='folder to store models')
    
        
    #########################
    # Some common parameters
    #########################
    
    if code_type in ['training']:
        
        parser.add_argument('--file-prefix', type=str, default = "", help='store file prefix')
        
    if code_type in ['CKA', 'hessian', 'model_dist', 'curve', 'loss_acc', 'ntk']:
        
        parser.add_argument('--early-stopping', dest='early_stopping', default = False, action='store_true', 
                    help='use early stopped checkpoints')
        parser.add_argument('--checkpoint-folder', type=str, default = '../checkpoint/gradient_power_law/CKA_analysis/mixup/', 
                            help='the folder to store checkpoint')
        parser.add_argument('--result-location', type=str, default = None, help='The result location')
    
    if code_type in ['CKA', 'hessian', 'curve', 'loss_acc']:
        
        parser.add_argument('--train-or-test', type=str, default='train', choices=['train', 'test'], help='use training or testing to measure hessian')

    #####################
    # parameters for CKA
    #####################
    
    if code_type in ['training', 'CKA']:
        
        parser.add_argument('--mixup-alpha', type=float, default = 1.0, help='parameter alpha in mixup training')
    
    if code_type in ['CKA']:
    
        parser.add_argument('--register-type', type=str, default = 'conv')
        parser.add_argument('--CKA-batches', type=int, default = 5, help='Number of batches to test CKA similarity')
        parser.add_argument('--CKA-repeat-runs', type=int, default = 1, help='Number of repeated runs to estimate CKA')
        parser.add_argument('--flattenHW', default = False, action = 'store_true', 
                            help = 'flatten the height and width dimension while only comparing the channel dimension')
        parser.add_argument('--compare-classification', dest='compare_classification', default = False, action='store_true', 
                    help='compare the classification results')
        parser.add_argument('--not-input', dest='not_input', default = False, action='store_true', 
                    help='no CKA computation on input data')
        parser.add_argument('--mixup-CKA', dest='mixup_CKA', default = False, action='store_true', 
                            help='measure CKA on mixup data')

    ########################
    # parameters for Hessian
    ########################
    
    if code_type in ['hessian']:
        
        # hessian parameters
        parser.add_argument('--mini-hessian-batch-size', type=int, default=200)
        parser.add_argument('--hessian-batch-size', type=int, default=2000, help='input batch size for hessian (default: 200)')
        parser.add_argument('--batch-norm', action='store_false', help='do we need batch norm or not')
        parser.add_argument('--residual', action='store_false', help='do we need residual connect or not')
        parser.add_argument('--cuda', action='store_false', help='do we use gpu or not')


    ########################
    # parameters for curve
    ########################
    
    if code_type in ['curve']:
        
        # for mode-connectivity curve training
        parser.add_argument('--dir', type=str, default='../checkpoints/', metavar='DIR', help='training directory (default: ../checkpoints/)')
        parser.add_argument('--result-suffix', type=str, default = '', help='save result suffix')
        parser.add_argument('--data_path', type=str, default=None, metavar='PATH', help='path to datasets location (default: None)')

        parser.add_argument('--num_bends', type=int, default=3, metavar='N', help='number of curve bends (default: 3)')
        parser.add_argument('--init_start', type=str, default=None, metavar='CKPT', help='checkpoint to init start point (default: None)')
        parser.add_argument('--fix_start', dest='fix_start', action='store_true', help='fix start point (default: off)')
        parser.add_argument('--init_end', type=str, default=None, metavar='CKPT', help='checkpoint to init end point (default: None)')
        parser.add_argument('--fix_end', dest='fix_end', action='store_true', help='fix end point (default: off)')
        parser.set_defaults(init_linear=True)
        parser.add_argument('--init_linear_off', dest='init_linear', action='store_false', help='turns off linear initialization of intermediate points (default: on)')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')

        # for curve evaluation
        parser.add_argument('--num_points', type=int, default=61, metavar='N', help='number of points on the curve (default: 61)')
        parser.add_argument('--only_eval', action='store_true', help='do not train curve, just evaluate')
        parser.add_argument('--to_eval', type=str, default=None, help='checkpoint to evaluate the curve on')


    ##########################
    # parameters for loss_acc
    ##########################

    if code_type in ['loss_acc']:
        parser.add_argument('--ensemble-average-acc', action='store_true')
        
    return parser
