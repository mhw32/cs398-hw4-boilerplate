r"""Generic utilities."""

# everything loads from this file so do not make relative imports
import os
import shutil
import random
import pickle
import numpy as np

from collections import Counter
from collections import defaultdict, OrderedDict

import torch

# map from integers to feedback labels
IX_TO_LABEL = {
    0: 'shapeLoop-none',
    1: 'square-none',
    2: 'side-none',
    3: 'shapeLoopHeader-missingValue',
    4: 'shapeLoopHeader-wrongOrder',
    5: 'shapeLoopHeader-wrongDelta',
    6: 'shapeLoopHeader-wrongEnd',
    7: 'shapeLoopHeader-wrongStart',
    8: 'square-armsLength',
    9: 'square-unrolled',
    10: 'square-wrongNumSides',
    11: 'side-forgotLeft',
    12: 'side-forgotMove',
    13: 'side-wrongMoveLeftOrder',
    14: 'side-armsLength',
    15: 'turn-wrongAmount',
    16: 'turn-rightLeftConfusion',
    17: 'move-wrongAmount',
}
NUM_LABELS = len(IX_TO_LABEL)

# these are "special" tokens used often to handle language
# given a sentence like "A brown dog.", we add these tokens 
# to make it:
# 
#       <sos> A brown dog . <pad> ... <pad> <eos>
# 
# where pad tokens are added to make ALL sentences in a 
# minibatch to the same size!
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'

OPEN_BRACKET = '('
END_BRACKET = ')'

# --- utilities for processing labels ---

def labels_to_numpy(labels, label_dim, label_to_ix_dict):
    tensor_np = np.zeros(label_dim)
    for label in labels:
        if label in label_to_ix_dict:
            tensor_np[label_to_ix_dict[label]] = 1
    tensor_np = tensor_np.astype(np.int)
    
    return tensor_np


def labels_to_tensor(labels, label_dim, label_to_ix_dict):
    tensor_np = labels_to_numpy(labels, label_dim, label_to_ix_dict)
    tensor = torch.from_numpy(tensor_np).long()
    
    return tensor


def tensor_to_labels(tensor, label_dim, ix_to_label_dict):
    assert tensor.size(0) == label_dim
    labels = []
    for ix in range(tensor.size(0)):
        if tensor[ix] >= 0.5:
            label = ix_to_label_dict[ix]
            labels.append(label)
    
    return ','.join(labels)

# --- miscellanous utilities ---

def train_test_split(array_list, train_frac=0.8, val_frac=0.1, test_frac=0.1):
    r"""Split data into three subsets (train, validation, and test).

    @param: array_list 
            list of np.arrays/torch.Tensors
            we will split each entry accordingly
    @param train_frac: float [default: 0.8]
                       must be within (0.0, 1.0)
    @param val_frac: float [default: 0.8]
                     must be within [0.0, 1.0)     
    @param train_frac: float [default: 0.8]
                       must be within (0.0, 1.0)
    """
    assert (train_frac + val_frac + test_frac) == 1.0

    train_list, test_list = [], []
    if val_frac > 0.0:
        val_list = []

    for array in array_list:
        size = len(array)

        train_array = array[:int(train_frac * size)]
        if val_frac > 0.0:
            val_array = array[
                int(train_frac * size):
                int((train_frac + val_frac) * size)
            ]
        test_array = array[int((train_frac + val_frac) * size):]

        train_list.append(train_array)
        test_list.append(test_array)

        if val_frac > 0.0:
            val_list.append(val_array)

    if val_frac > 0.0:
        return train_list, val_list, test_list
    else:
        return train_list, test_list


class OrderedCounter(Counter, OrderedDict):
    r"""Counter that remembers the order elements are first encountered"""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    # saves a copy of the model (+ properties) to filesystem
    if not os.path.isdir(folder):
        os.makedirs(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))