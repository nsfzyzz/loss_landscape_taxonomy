import numpy as np
import torch
import torch.nn as nn
import os


def test(args, model, test_loader):
    print('Testing')
    model.eval()
    
    correct = 0
    total_num = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            total_num += len(data)
    
    print('testing_correct: ', correct / total_num, '\n')
    return correct / total_num 


def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def return_file_name(args, exp_id1, exp_id2):

    checkpoint1 = os.path.join(args.checkpoint_folder, f"net_exp_{exp_id1}.pkl")
    checkpoint2 = os.path.join(args.checkpoint_folder, f"net_exp_{exp_id2}.pkl")   
    
    early_stopped_checkpoint1 = os.path.join(args.checkpoint_folder, f"net_exp_{exp_id1}_early_stopped_model.pkl")
    early_stopped_checkpoint2 = os.path.join(args.checkpoint_folder, f"net_exp_{exp_id2}_early_stopped_model.pkl")
    
    best_checkpoint1 = f'{args.checkpoint_folder}/net_exp_{exp_id1}_best.pkl'
    best_checkpoint2 = f'{args.checkpoint_folder}/net_exp_{exp_id2}_best.pkl'
    
    if args.early_stopping:
        if os.path.exists(early_stopped_checkpoint1):
            checkpoint1 = early_stopped_checkpoint1
            print("Using the early stopping model!")
        elif os.path.exists(best_checkpoint1):
            checkpoint1 = best_checkpoint1
            print("Using best model!")

        if os.path.exists(early_stopped_checkpoint2):
            checkpoint2 = early_stopped_checkpoint2
            print("Using the early stopping model!")
        elif os.path.exists(best_checkpoint2):
            checkpoint2 = best_checkpoint2
            print("Using best model!")
    
    print(checkpoint1)
    print(checkpoint2)
        
    return checkpoint1, checkpoint2


def return_file_name_single(args, exp_id1):

    checkpoint1 = os.path.join(args.checkpoint_folder, f"net_exp_{exp_id1}.pkl")
    early_stopped_checkpoint1 = os.path.join(args.checkpoint_folder, f"net_exp_{exp_id1}_early_stopped_model.pkl")
    best_checkpoint1 = f'{args.checkpoint_folder}/net_exp_{exp_id1}_best.pkl'
    
    if args.early_stopping:
        if os.path.exists(early_stopped_checkpoint1):
            checkpoint1 = early_stopped_checkpoint1
            print("Using the early stopping model!")
        elif os.path.exists(best_checkpoint1):
            checkpoint1 = best_checkpoint1
            print("Using best model!")
    
    print(checkpoint1)
        
    return checkpoint1


def test_acc_loss(test_loader, model, criterion, regularizer=None, **kwargs):
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0

    model.eval()

    for input, target in test_loader:
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(input, **kwargs)
        nll = criterion(output, target)
        loss = nll.clone()
        if regularizer is not None:
            loss += regularizer(model)

        nll_sum += nll.item() * input.size(0)
        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'nll': nll_sum / len(test_loader.dataset),
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }


def test_ensemble_average(models, test_loader, weights =None):
    
    smx=nn.Softmax()
    
    print('Testing')
    for model in models:
        model.eval()
    
    num_models = len(models)
    if weights == None:
        weights = [1.0]*num_models
    
    correct = 0
    total_num = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            for ind in range(num_models):
                if ind == 0:
                    output = smx(models[ind](data)) * weights[ind]
                else:
                    output += smx(models[ind](data)) * weights[ind]
            
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            total_num += len(data)
    
    print('testing_correct: ', correct / total_num, '\n')
    return correct / total_num 