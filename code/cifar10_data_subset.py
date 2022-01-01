"""
cifar-10 dataset, but only with a subset of data
"""
import numpy as np
import os
import pickle
import torch
import torchvision.datasets as datasets


class CIFAR10Subset(datasets.CIFAR10):
    """CIFAR10 dataset, but only with a subset of data
    Params
    ------
    noise_magnitude: int
      Default 0. The noise magnitude added to the images
    """
    def __init__(self, 
                 subset=1.0, 
                 corrupt_prob=0.0, 
                 label_path = '../../data/random_labels/random_label.pkl', 
                 trainset = True, 
                 noise_magnitude=0, 
                 noise_path = '../../data/random_noise/random_noise.pkl', 
                 random_seed = 12345,
                 noise_compute='add',
                 image_noise=False,
                 test_on_noise=False, 
                 subset_noisy=False, 
                 noise_type='positive_uniform',
                 **kwargs):
        
        super(CIFAR10Subset, self).__init__(**kwargs)    
        
        self.label_path = label_path
        self.noise_path = noise_path
        self.noise_type = noise_type
        self.noise_compute = noise_compute
        self.random_seed = random_seed
        
        if subset_noisy:
            if trainset and corrupt_prob>0:
                self.corrupt_labels(corrupt_prob)
            elif test_on_noise:
                raise NameError('The subset case with test noise should have been implemented in the data.py file')
        
        if trainset and image_noise:
            self.corrupt_noise(noise_magnitude)
            
        self.select(subset)

    def select(self, subset):
        print(f'CIFAR10Subset: using {subset} of the whole dataset')
        num_data = int(len(self.data)*subset)
        self.data = self.data[:num_data]
        self.targets = self.targets[:num_data]
        
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
        
    def corrupt_noise(self, noise_magnitude):
        print(f'CIFAR10RandomNoise: Using {self.noise_type} at {self.noise_path} noise')
        if not os.path.isfile(self.noise_path):
            print("Noise not ready yet.")
            np.random.seed(self.random_seed)
            if self.noise_type == 'bernoulli':
                noise = np.random.randint(0, 2, size = self.data.shape, dtype=np.uint8)*2-1
            elif self.noise_type == 'gaussian':
                noise = np.random.normal(0, 1, size = self.data.shape)
            elif self.noise_type == 'uniform':
                noise = np.random.uniform(-1, 1, size = self.data.shape)
            elif self.noise_type == 'positive_uniform':
                noise = np.random.uniform(0, 1, size = self.data.shape)
            else:
                print(f'Noise type of {self.noise_type} not recognized, using bernoulli noise instead')
                noise = np.random.randint(0, 2, size=self.data.shape, dtype=np.uint8)*2-1
            pickle.dump(noise, open(self.noise_path, 'wb'))
            print("New random noise generated")
        else:
            print("Reading random noise from file")
            noise = pickle.load(open(self.noise_path, 'rb'))
        
        if self.noise_compute == 'add':
            self.data += np.uint8(noise * noise_magnitude)
        elif self.noise_compute == 'replace':
            self.data = np.uint8(noise * noise_magnitude)
        else:
            raise NameError('Noise compute method is not recognized.')
        self.data = np.clip(self.data, 0, 255)
