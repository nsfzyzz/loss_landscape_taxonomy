"""
cifar-10 dataset, with support for random labels
"""
import numpy as np
import os
import pickle
import torchvision.datasets as datasets


class CIFAR10RandomLabels(datasets.CIFAR10):
    """CIFAR10 dataset, with support for randomly corrupt labels.
    Params
    ------
    corrupt_prob: float
      Default 0.0. The probability of a label being replaced with
      random label.
    num_classes: int
      Default 10. The number of classes in the dataset.
    """
    def __init__(self, corrupt_prob=0.0, num_classes=10, label_path = '../../data/random_labels/random_label.pkl', trainset = True, test_on_noise=False, **kwargs):
        super(CIFAR10RandomLabels, self).__init__(**kwargs)
        self.n_classes = num_classes
        self.label_path = label_path
        if (trainset and corrupt_prob > 0) or test_on_noise:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
    
        if not os.path.isfile(self.label_path):
            print("Labels not ready yet.")
            labels = np.array(self.targets)
            np.random.seed(12345)
            mask = np.random.rand(len(labels)) <= corrupt_prob
            rnd_labels = np.random.choice(self.n_classes, mask.sum())
            labels[mask] = rnd_labels
            # we need to explicitly cast the labels from npy.int64 to
            # builtin int type, otherwise pytorch will fail...
            labels = [int(x) for x in labels]
            pickle.dump(labels, open(self.label_path, 'wb'))
            print("New labels generated")
        else:
            print("Reading random labels from file")
            print(self.label_path)
            labels = pickle.load(open(self.label_path, 'rb'))
            
        self.targets = labels