"""The following code is adapted from 
Similarity of Neural Network Representations Revisited
Simon Kornblith, Mohammad Norouzi, Honglak Lee and Geoffrey Hinton
https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

softmax1 = nn.Softmax().cuda()

def compare_classification(model1, model2, eval_loader, args=None, cos=None):
    
    model1.eval()
    model2.eval()
    total_divergence = 0.0
    total_num = 0
    total_batch = 0
    total_agreement = 0
    total_cosine = 0
    
    for inputs, targets in eval_loader:
        
        total_num += inputs.shape[0]
        total_batch += 1
        inputs, targets = inputs.cuda(), targets.cuda()
        if args.mixup_CKA:
            inputs, _, _, _ = mixup_data(inputs, targets, args.mixup_alpha, use_cuda = True)
        with torch.no_grad():
            outputs1 = model1(inputs)
            outputs2 = model2(inputs)
        
        outputs1_vec, outputs2_vec = outputs1.view(-1), outputs2.view(-1)

        sim = cos(outputs1_vec, outputs2_vec)

        p_1, p_2 = F.softmax(outputs1, dim=1), F.softmax(outputs2, dim=1)
        p_mixture = torch.clamp((p_1 + p_2) / 2., 1e-7, 1).log()
                
        divergence = (F.kl_div(p_mixture, p_1, reduction='sum') +
            F.kl_div(p_mixture, p_2, reduction='sum')) / 2.
        
        total_divergence += divergence.detach()
        
        pred1 = outputs1.data.max(1, keepdim=True)[1]
        pred2 = outputs2.data.max(1, keepdim=True)[1]
        total_agreement += pred1.eq(pred2).sum().detach()
        total_cosine += sim.detach()
        
    agreement = total_agreement.item()/total_num
    divergence = total_divergence.item()/total_num
    cosine_ave = total_cosine.item()/total_batch

    return agreement, divergence, cosine_ave

            
def register_inputs(inputs, activations):
    
    if 'input' in activations.keys():
        activations['input'] = torch.cat([activations['input'], inputs.detach()])
    else:
        activations['input'] = inputs.detach()

            
def register_output(output, activations):
    
    if 'output' in activations.keys():
        activations['output'] = torch.cat([activations['output'], output.detach()])
    else:
        activations['output'] = output.detach()
    
    if 'softmax' in activations.keys():
        activations['softmax'] = torch.cat([activations['softmax'], softmax1(output.detach()).detach()])
    else:
        activations['softmax'] = softmax1(output.detach()).detach()
        
        
def all_latent(model1, model2, eval_loader, num_batches = 10, args = None):
    
    # These two variables should be global
    latent_all_1 = {}
    latent_all_2 = {}

    model1.eval()
    model2.eval()
    
    for batch_idx, (inputs, targets) in enumerate(eval_loader):
        if batch_idx>num_batches:
            break
            
        inputs, targets = inputs.cuda(), targets.cuda()
        
        if args.mixup_CKA:
            inputs, _, _, _ = mixup_data(inputs, targets, args.mixup_alpha, use_cuda = True)
        
        if not args.not_input:
            register_inputs(inputs, latent_all_1)
            register_inputs(inputs, latent_all_2)
        
        with torch.no_grad():
            outputs1 = model1(inputs)
            outputs2 = model2(inputs)
            
        register_output(outputs1, latent_all_1)
        register_output(outputs2, latent_all_2)
        
    for name in latent_all_1.keys():
        
        shape = latent_all_1[name].shape
        
        if not args.flattenHW or len(shape) != 4:
            latent_all_1[name] = latent_all_1[name].view(latent_all_1[name].size(0), -1).cpu().data.numpy()
            latent_all_2[name] = latent_all_2[name].view(latent_all_2[name].size(0), -1).cpu().data.numpy()

        else:
            latent_all_1[name] = latent_all_1[name].permute(0,2,3,1).reshape(-1, shape[3]).cpu().data.numpy()
            latent_all_2[name] = latent_all_2[name].permute(0,2,3,1).reshape(-1, shape[3]).cpu().data.numpy()

    return latent_all_1, latent_all_2


def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel.

    Args:
        x: A num_examples x num_features matrix of features.

    Returns:
        A num_examples x num_examples Gram matrix of examples.
    """
    return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
    """Compute Gram (kernel) matrix for an RBF kernel.

    Args:
      x: A num_examples x num_features matrix of features.
      threshold: Fraction of median Euclidean distance to use as RBF kernel
        bandwidth. (This is the heuristic we use in the paper. There are other
        possible ways to set the bandwidth; we didn't try them.)

    Returns:
      A num_examples x num_examples Gram matrix of examples.
    """
    dot_products = x.dot(x.T)
    sq_norms = np.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = np.median(sq_distances)
    return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix.

    This is equvialent to centering the (possibly infinite-dimensional) features
    induced by the kernel before computing the Gram matrix.

    Args:
      gram: A num_examples x num_examples symmetric matrix.
      unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
        estimate of HSIC. Note that this estimator may be negative.

    Returns:
      A symmetric matrix with centered columns and rows.
    """
    if not np.allclose(gram, gram.T):
        raise ValueError('Input must be a symmetric matrix.')
    gram = gram.copy()

    if unbiased:
      # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
      # L. (2014). Partial distance correlation with methods for dissimilarities.
      # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
      # stable than the alternative from Song et al. (2007).
        n = gram.shape[0]
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, 0, dtype=np.float64)
        means -= np.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]

    return gram


def cka_compute(gram_x, gram_y, debiased=False):
    """Compute CKA.

    Args:
      gram_x: A num_examples x num_examples Gram matrix.
      gram_y: A num_examples x num_examples Gram matrix.
      debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
      The value of CKA between X and Y.
    """
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(
    xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
    n):
    """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
    # This formula can be derived by manipulating the unbiased estimator from
    # Song et al. (2007).
    return (
        xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
        + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


def feature_space_linear_cka(features_x, features_y, debiased=False):
    """Compute CKA with a linear kernel, in feature space.

    This is typically faster than computing the Gram matrix when there are fewer
    features than examples.

    Args:
        features_x: A num_examples x num_features matrix of features.
        features_y: A num_examples x num_features matrix of features.
        debiased: Use unbiased estimator of dot product similarity. CKA may still be
          biased. Note that this estimator may be negative.

    Returns:
        The value of CKA between X and Y.
    """
        
    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)

    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))

    if debiased:
        n = features_x.shape[0]
        # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
        sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
        sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
        squared_norm_x = np.sum(sum_squared_rows_x)
        squared_norm_y = np.sum(sum_squared_rows_y)
        
        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
            squared_norm_x, squared_norm_y, n)
        
        """
        dx = _debiased_dot_product_similarity_helper(
            normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
            squared_norm_x, squared_norm_x, n)
        dy = _debiased_dot_product_similarity_helper(
            normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
            squared_norm_y, squared_norm_y, n)
        
        print("One estimate is {0}".format(dx))
        
        if dx<0:
            print(dx)
            1/0
        if dy<0:
            print(dy)
            1/0
        """
        
        normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
            squared_norm_x, squared_norm_x, n))
        normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
            squared_norm_y, squared_norm_y, n))

    return dot_product_similarity / (normalization_x * normalization_y)
