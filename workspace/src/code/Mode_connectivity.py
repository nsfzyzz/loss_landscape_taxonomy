"""The following code is adapted from 
Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs
Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry Vetrov and Andrew Gordon Wilson
https://github.com/timgaripov/dnn-mode-connectivity
"""

from __future__ import print_function
import numpy as np
import os
import torch
import torch.nn.functional as F
import time
import tabulate

import sys
sys.path.insert(1, './code/')

from arguments import get_parser
from data import get_loader
from utils import *
import curves

parser = get_parser(code_type='curve')
parser.add_argument('--curve', type=str, default=curves.Bezier, metavar='CURVE',
                    help='curve type to use (default: None)')

def clean_state_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name=k
        #name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def learning_rate_schedule(base_lr, epoch, total_epochs):

    """The learning rate schedule for testing the mode connectivity
    """

    alpha = epoch / total_epochs
    if alpha <= 0.5:
        factor = 1.0
    elif alpha <= 0.9:
        factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
    else:
        factor = 0.01
    return factor * base_lr

def train(train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None):

    """The training function for calculating the mode connectivity.
    """

    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    model.train()
    for iter, (input, target) in enumerate(train_loader):
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(input)
        loss = criterion(output, target)
        if regularizer is not None:
            loss += regularizer(model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }

def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)

def isbatchnorm(module):
    return issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm) or \
           issubclass(module.__class__, curves._BatchNorm)

def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]

def _check_bn(module, flag):
    if isbatchnorm(module):
        flag[0] = True

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def reset_bn(module):
    if isbatchnorm(module):
        module.reset_running_stats()

def _get_momenta(module, momenta):
    if isbatchnorm(module):
        momenta[module] = module.momentum

def _set_momenta(module, momenta):
    if isbatchnorm(module):
        module.momentum = momenta[module]

def update_bn(loader, model, **kwargs):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    num_samples = 0
    for input, _ in loader:
        input = input.cuda(non_blocking=True)
        batch_size = input.data.size(0)

        momentum = batch_size / (num_samples + batch_size)
        for module in momenta.keys():
            module.momentum = momentum

        model(input, **kwargs)
        num_samples += batch_size

    model.apply(lambda module: _set_momenta(module, momenta))
 
def stats(values, dl):
    min = np.min(values)
    max = np.max(values)
    avg = np.mean(values)
    int = np.sum(0.5 * (values[:-1] + values[1:]) * dl[1:]) / np.sum(dl[1:])
    return min, max, avg, int


args = parser.parse_args()

from models.resnet_width import ResNet18
from models.resnet_width_curves import ResNet18Curve

arch_kwargs = {'width': args.resnet18_width}

arch = ResNet18
curve_arch = ResNet18Curve
curve = getattr(curves, args.curve)
train_loader, test_loader= get_loader(args)

if not args.only_eval:
    model = curves.CurveNet(
        args.num_classes,
        curve,
        curve_arch,
        args.num_bends,
        args.fix_start,
        args.fix_end,
        architecture_kwargs=arch_kwargs,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        print("USING CPU")
    base_model = None
    for path, k in [(args.init_start, 0), (args.init_end, args.num_bends - 1)]:    
        if path:
            checkpoint = torch.load(path)
            checkpoint = clean_state_dict(checkpoint)
            print('Loading %s as point #%d' % (path, k))
            if base_model is None:
                base_model = arch(**arch_kwargs).to(device)
            base_model.load_state_dict(checkpoint)
            model.import_base_parameters(base_model, k)
    model = model.to(device)
    print(model.parameters())
    criterion = F.cross_entropy
    regularizer = None if args.curve is None else curves.l2_regularizer(args.weight_decay)
    optimizer = torch.optim.SGD(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd if args.curve is None else 0.0
    )
    start_epoch = 1
    if args.resume is not None:
        print('Resume training from %s' % args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_nll', 'te_acc', 'time']
    start = time.time()
    train_loader, test_loader = get_loader(args)
    has_bn = check_bn(model)
    test_res = {'loss': None, 'accuracy': None, 'nll': None}
    for epoch in range(start_epoch, args.epochs + 1):
        time_ep = time.time()

        lr = learning_rate_schedule(args.lr, epoch, args.epochs)
        adjust_learning_rate(optimizer, lr)
        train_res = train(train_loader, model, optimizer, criterion, regularizer)
        train_time = time.time()
        test_res = test_acc_loss(test_loader, model, criterion, regularizer)
        if epoch % args.save_frequency == 0:
            save_checkpoint(
                args.dir,
                epoch,
                name=f'checkpoint{args.result_suffix}',
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict()
            )

        time_ep = time.time() - time_ep
        values = [epoch, lr, train_res['loss'], train_res['accuracy'], test_res['nll'],
                  test_res['accuracy'], time_ep]

        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
        if epoch % 40 == 1 or epoch == start_epoch:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

    if args.epochs % args.save_frequency != 0:
        save_checkpoint(
            args.dir,
            args.epochs,
            name=f'checkpoint{args.result_suffix}',
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )

model = curves.CurveNet(
    args.num_classes,
    curve,
    curve_arch,
    args.num_bends,
    args.fix_start,
    args.fix_end,
    architecture_kwargs=arch_kwargs,
)

model.cuda()
checkpoint = torch.load(args.to_eval)
model.load_state_dict(checkpoint['model_state'])

criterion = F.cross_entropy
regularizer = curves.l2_regularizer(args.weight_decay)

T = args.num_points
ts = np.linspace(0.0, 1.0, T)
tr_loss = np.zeros(T)
tr_nll = np.zeros(T)
tr_acc = np.zeros(T)
te_loss = np.zeros(T)
te_nll = np.zeros(T)
te_acc = np.zeros(T)
tr_err = np.zeros(T)
te_err = np.zeros(T)
dl = np.zeros(T)

previous_weights = None

columns = ['t', 'Train loss', 'Train nll', 'Train error (%)', 'Test nll', 'Test error (%)']

t = torch.FloatTensor([0.0]).cuda()
for i, t_value in enumerate(ts):
    t.data.fill_(t_value)
    weights = model.weights(t)
    if previous_weights is not None:
        dl[i] = np.sqrt(np.sum(np.square(weights - previous_weights)))
    previous_weights = weights.copy()

    update_bn(train_loader, model, t=t)
    tr_res = test_acc_loss(train_loader, model, criterion, regularizer, t=t)
    te_res = test_acc_loss(test_loader, model, criterion, regularizer, t=t)
    
    tr_loss[i] = tr_res['loss']
    tr_nll[i] = tr_res['nll']
    tr_acc[i] = tr_res['accuracy']
    tr_err[i] = 100.0 - tr_acc[i]
    te_loss[i] = te_res['loss']
    te_nll[i] = te_res['nll']
    te_acc[i] = te_res['accuracy']
    te_err[i] = 100.0 - te_acc[i]

    values = [t, tr_loss[i], tr_nll[i], tr_err[i], te_nll[i], te_err[i]]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
    if i % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)


tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int = stats(tr_loss, dl)
tr_nll_min, tr_nll_max, tr_nll_avg, tr_nll_int = stats(tr_nll, dl)
tr_err_min, tr_err_max, tr_err_avg, tr_err_int = stats(tr_err, dl)

te_loss_min, te_loss_max, te_loss_avg, te_loss_int = stats(te_loss, dl)
te_nll_min, te_nll_max, te_nll_avg, te_nll_int = stats(te_nll, dl)
te_err_min, te_err_max, te_err_avg, te_err_int = stats(te_err, dl)

print('Length: %.2f' % np.sum(dl))
print(tabulate.tabulate([
        ['train loss', tr_loss[0], tr_loss[-1], tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int],
        ['train error (%)', tr_err[0], tr_err[-1], tr_err_min, tr_err_max, tr_err_avg, tr_err_int],
        ['test nll', te_nll[0], te_nll[-1], te_nll_min, te_nll_max, te_nll_avg, te_nll_int],
        ['test error (%)', te_err[0], te_err[-1], te_err_min, te_err_max, te_err_avg, te_err_int],
    ], [
        '', 'start', 'end', 'min', 'max', 'avg', 'int'
    ], tablefmt='simple', floatfmt='10.4f'))

np.savez(
    args.result_location,
    ts=ts,
    dl=dl,
    tr_loss=tr_loss,
    tr_loss_min=tr_loss_min,
    tr_loss_max=tr_loss_max,
    tr_loss_avg=tr_loss_avg,
    tr_loss_int=tr_loss_int,
    tr_nll=tr_nll,
    tr_nll_min=tr_nll_min,
    tr_nll_max=tr_nll_max,
    tr_nll_avg=tr_nll_avg,
    tr_nll_int=tr_nll_int,
    tr_acc=tr_acc,
    tr_err=tr_err,
    tr_err_min=tr_err_min,
    tr_err_max=tr_err_max,
    tr_err_avg=tr_err_avg,
    tr_err_int=tr_err_int,
    te_loss=te_loss,
    te_loss_min=te_loss_min,
    te_loss_max=te_loss_max,
    te_loss_avg=te_loss_avg,
    te_loss_int=te_loss_int,
    te_nll=te_nll,
    te_nll_min=te_nll_min,
    te_nll_max=te_nll_max,
    te_nll_avg=te_nll_avg,
    te_nll_int=te_nll_int,
    te_acc=te_acc,
    te_err=te_err,
    te_err_min=te_err_min,
    te_err_max=te_err_max,
    te_err_avg=te_err_avg,
    te_err_int=te_err_int,
)
