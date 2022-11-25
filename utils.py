import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import scipy.special

# for left-multiplication for RGB -> Y'PbPr
RGB_TO_YUV = np.array([[0.29900, -0.16874, 0.50000],
                       [0.58700, -0.33126, -0.41869],
                       [0.11400, 0.50000, -0.08131]])


def normalize_data(x, mode=None):
    if mode is None or mode == 'rgb':
        return x / 127.5 - 1.
    elif mode == 'rgb_unit_var':
        return 2. * normalize_data(x, mode='rgb')
    elif mode == 'yuv':
        return (x / 127.5 - 1.).dot(RGB_TO_YUV)
    else:
        raise NotImplementedError(mode)


def log_min_exp(a, b, epsilon=1.e-6):
    """Computes the log(exp(a) - exp(b)) (b<a) in a numerically stable fashion."""
    y = a + torch.log1p(-torch.exp(b - a) + epsilon)
    return y


def categorical_kl_logits(logits1, logits2, eps=1.e-6):
    """KL divergence between categorical distributions.

    Distributions parameterized by logits.

    Args:
      logits1: logits of the first distribution. Last dim is class dim.
      logits2: logits of the second distribution. Last dim is class dim.
      eps: float small number to avoid numerical issues.

    Returns:
      KL(C(logits1) || C(logits2)): shape: logits1.shape[:-1]
    """
    out = (F.softmax(logits1 + eps, dim=-1) * (
                F.log_softmax(logits1 + eps, dim=-1) - F.log_softmax(logits2 + eps, dim=-1)))
    return torch.sum(out, dim=-1)


def categorical_kl_probs(probs1, probs2, eps=1.e-6):
    """KL divergence between categorical distributions.

    Distributions parameterized by logits.

    Args:
      probs1: probs of the first distribution. Last dim is class dim.
      probs2: probs of the second distribution. Last dim is class dim.
      eps: float small number to avoid numerical issues.

    Returns:
      KL(C(probs) || C(logits2)): shape: logits1.shape[:-1]
    """
    out = probs1 * (torch.log(probs1 + eps) - torch.log(probs2 + eps))
    return torch.sum(out, dim=-1)


def categorical_log_likelihood(x, logits):
    """Log likelihood of a discretized Gaussian specialized for image data.

    Assumes data `x` consists of integers [0, num_classes-1].

    Args:
      x: where to evaluate the distribution. shape = (bs, ...), dtype=int32/int64
      logits: logits, shape = (bs, ..., num_classes)

    Returns:
      log likelihoods
    """
    log_probs = F.log_softmax(logits, dim=-1)
    x_onehot = F.one_hot(x.to(torch.int64), logits.shape[-1])
    return torch.sum(log_probs * x_onehot, dim=-1)


def meanflat(x):
    """Take the mean over all axes except the first batch dimension."""
    return x.mean(dim=tuple(range(1, len(x.shape))))
