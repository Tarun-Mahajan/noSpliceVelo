import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

def entropy(logits, targets):
    """Entropy loss
        loss = (1/n) * -Σ targets*log(predicted)

    Args:
        logits: (array) corresponding array containing the logits of the categorical variable
        real: (array) corresponding array containing the true labels

    Returns:
        output: (array/float) depending on average parameters the result will be the mean
                            of all the sample losses or an array with the losses per sample
    """
    log_q = F.log_softmax(logits, dim=-1)
    return -torch.mean(torch.sum(targets * log_q, dim=-1))


def entropy_no_mean(logits, targets):
    """Entropy loss
        loss = (1/n) * -Σ targets*log(predicted)

    Args:
        logits: (array) corresponding array containing the logits of the categorical variable
        real: (array) corresponding array containing the true labels

    Returns:
        output: (array/float) depending on average parameters the result will be the mean
                            of all the sample losses or an array with the losses per sample
    """
    log_q = F.log_softmax(logits, dim=-1)
    return -(torch.sum(targets * log_q, dim=-1))