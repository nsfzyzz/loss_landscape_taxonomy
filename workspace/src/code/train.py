from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.insert(1, './code/')

from arguments import get_parser
from utils import *
from data import get_loader

parser = get_parser(code_type='training')
    
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    

def save_early_stop_model(args, model, loss_vals):
    
    if not args.save_early_stop or len(loss_vals)<30:
        
        print("Early stopping not satisfied.")
        
        return False
    
    else:
        
        for i in range(args.patience):
            
            if (loss_vals[-1-i] < loss_vals[-2-i] - args.min_delta) or (loss_vals[-1-i] > loss_vals[-2-i] + args.min_delta):
                
                print("Early stopping not satisfied.")
                
                return False
        
        args.save_early_stop = False
        
        print("Early stopping satisfied!!! Saving early stopped model.")
        
        return True


def train(args, model, train_loader, test_loader, optimizer, criterion, epoch):
    model.train()
    
    print('Current Epoch: ', epoch)
    train_loss = 0.
    total_num = 0
    correct = 0
    
    P = 0 # num samples / batch size
    for inputs, targets in train_loader:

        if args.ignore_incomplete_batch:
            if_condition = inputs.shape[0] != args.train_bs
            
            if if_condition:
                print("Neglect the last epoch so that num samples/batch size = int")
                break
        
        P += 1
        # loop over dataset
        inputs, targets = inputs.to("cuda"), targets.to("cuda")
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        train_loss += loss.item() * targets.size()[0]
        total_num += targets.size()[0]
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        
        loss.backward()
        optimizer.step()
            
    acc = test(args, model, test_loader)
    train_loss /= total_num
    print(f"Training Loss of Epoch {epoch}: {train_loss}")
    print(f"Testing of Epoch {epoch}: {acc}")  
    
    return train_loss


def main():
    
    args = parser.parse_args()
    
    if args.save_final or args.no_lr_decay or args.one_lr_decay:
        if args.saving_folder == '':
            raise ('you must give a position and name to save your model')
        if args.saving_folder[-1] != '/':
            args.saving_folder += '/'

    for arg in vars(args):
        print(arg, getattr(args, arg))

    print("------------------------------------------------------")
    print("Experiement: {0} training for {1}".format(args.training_type, args.arch))
    print("------------------------------------------------------")

    criterion = nn.CrossEntropyLoss().to("cuda")
    
    from models.resnet_width import ResNet18

    model = ResNet18(width=args.resnet18_width).cuda()
    
    if args.resume:
        
        model.load_state_dict(torch.load(f"{args.resume}"))
    
    train_loader, test_loader = get_loader(args)

    if args.training_type == 'small_lr':
        base_lr = 0.003
    else:
        base_lr = args.lr

    print("The base learning rate is {0}".format(base_lr))

    if args.training_type == 'small_lr':
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.training_type == 'no_decay':
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0)
    else:
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=args.weight_decay) 

    loss_vals = []
    best_train_loss = 1000000
    
    for epoch in range(args.epochs):

        print("---------------------")
        print("Start epoch {0}".format(epoch))
        print("---------------------")

        if epoch >= args.epochs*0.75:
            lr = base_lr * 0.01    
        elif epoch >= args.epochs*0.5:
            lr = base_lr * 0.1
        else:
            lr = base_lr
            
        if args.no_lr_decay:
            lr = base_lr
        elif args.one_lr_decay:
            # Here, the training is done with one learning rate decay
            # So it's hard to justify what temperature is used
            # Therefore, we train with longer first period
            if epoch >= args.epochs*0.75:
                lr = base_lr * 0.1
            else:
                lr = base_lr
        
        update_lr(optimizer, lr)        
        
        train_loss = train(args, model, train_loader, test_loader, optimizer, criterion, epoch)
        
        loss_vals.append(train_loss)
        
        if args.save_best and train_loss < best_train_loss:
            print('Model with the best training loss saved! The loss is {0}'.format(train_loss))
            torch.save(model.state_dict(), f'{args.saving_folder}net_{args.file_prefix}_best.pkl')
            best_train_loss = train_loss
        
        if args.only_exploration and epoch >= args.epochs*0.5:
            print("only log the process before lr decay")
            break
        
        if save_early_stop_model(args, model, loss_vals):
            torch.save(model.state_dict(), f'{args.saving_folder}net_{args.file_prefix}_early_stopped_model.pkl')
            
        if args.no_lr_decay and epoch==args.stop_epoch:
            torch.save(model.state_dict(), f'{args.saving_folder}net_{args.file_prefix}.pkl')
            print("Early stopping without learning rate decay.")
            break
            
        if args.one_lr_decay and epoch==args.stop_epoch:
            torch.save(model.state_dict(), f'{args.saving_folder}net_{args.file_prefix}.pkl')
            print("Early stopping with only one lr decay")
            break
        
        if (epoch%5 ==0 or epoch == args.epochs-1) and args.save_final:
            torch.save(model.state_dict(), f'{args.saving_folder}net_{args.file_prefix}.pkl')
            
        if args.save_middle and (epoch%args.save_frequency ==0 or epoch == args.epochs-1):
            torch.save(model.state_dict(), f'{args.saving_folder}net_{epoch}.pkl')

if __name__ == '__main__':
    main()
