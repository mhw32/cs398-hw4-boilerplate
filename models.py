import sys
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn_utils

from trainer.utils import NUM_LABELS, IX_TO_LABEL


class FeedbackNN(nn.Module):
    """
    Neural network responsible for ingesting a tokenized student 
    program, and spitting out a categorical prediction.

    We give you the following information:
        vocab_size: number of unique tokens 
        num_labels: number of output feedback labels
    """
    
    def __init__(self, vocab_size, num_labels):
        raise NotImplementedError

    def forward(self, token_seq, token_length):
        # `token_seq` is a (batch_size x max_token_length) tensor where 
        # each element is a number between 0 to |V| - 1. `token_length` is 
        # a (batch_size) tensor where each element represents the 
        # non-padded length.
        raise NotImplementedError
