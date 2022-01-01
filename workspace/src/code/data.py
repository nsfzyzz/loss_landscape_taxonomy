from cifar10_data_random_labels import CIFAR10RandomLabels
from cifar10_data_subset import CIFAR10Subset
from torchvision import datasets, transforms
import torch


def get_loader(args):

    CIFAR10_mean, CIFAR10_var = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    
    transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_mean, CIFAR10_var),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_mean, CIFAR10_var),
    ])
    
    if args.random_labels:
    
        kwargs = {'num_workers': 2, 'pin_memory': True}
        ## Here, we try with shuffle or not shuffling data
        train_loader = torch.utils.data.DataLoader(
                        CIFAR10RandomLabels(root='../../data/random_labels', train=True, download=True,
                        label_path = args.random_label_path,
                        transform=transform_train, num_classes=args.num_classes,
                        corrupt_prob=args.label_corrupt_prob, trainset = True),
                        batch_size=args.train_bs, shuffle=args.shuffle_random_data, **kwargs)
        test_loader = torch.utils.data.DataLoader(
                        CIFAR10RandomLabels(root='../../data/random_labels', train=False,
                        label_path = args.random_label_path_test,
                        transform=transform_test, num_classes=args.num_classes,
                        corrupt_prob=args.label_corrupt_prob, trainset = False, test_on_noise=args.test_on_noise),
                        batch_size=args.test_bs, shuffle=False, **kwargs)
        
    elif args.data_subset:
        
        kwargs = {'num_workers': 2, 'pin_memory': True}
        train_set = CIFAR10Subset(root='../../data', train=True, download=True, subset=args.subset,
                                    label_path = args.random_label_path, corrupt_prob=args.label_corrupt_prob, 
                                    trainset = True, subset_noisy=args.subset_noisy, 
                                    transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train_bs, shuffle=True, **kwargs)
        
        if args.subset_noisy and args.test_on_noise:
            test_set = CIFAR10RandomLabels(root='../../data/random_labels', train=False,
                        label_path = args.random_label_path_test,
                        transform=transform_test, num_classes=args.num_classes,
                        corrupt_prob=args.label_corrupt_prob, trainset = False, test_on_noise=args.test_on_noise)
            
        else:
            test_set = datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_bs, shuffle=False)

    else:
        trainset = datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bs, shuffle=True)

        testset = datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_bs, shuffle=False)
    
    return train_loader, test_loader

