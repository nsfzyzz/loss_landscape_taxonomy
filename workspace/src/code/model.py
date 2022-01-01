import torch
from models.resnet_width import ResNet18 as ResNet18_width


def get_new_model(args):
    
    model = ResNet18_width(width = args.resnet18_width).cuda()
    
    return model


def load_checkpoint(args, file_name):
    
    checkpoint = torch.load(file_name)
    model = get_new_model(args)
    model.load_state_dict(checkpoint)
    
    return model


def get_teacher_model(args):
    
    # We now assume that the teacher model is always ResNet18_wx
    assert 'ResNet18_w' in args.teacher_arch
    teacher = ResNet18_width(width = args.teacher_width).cuda()
    
    return teacher


def load_teacher_checkpoint(args):
    
    checkpoint = torch.load(args.teacher_ckpt)['state_dict']
    teacher = get_teacher_model(args)
    teacher.load_state_dict(checkpoint)
    
    return teacher