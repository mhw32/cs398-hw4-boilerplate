import os
import sys
import random
import pickle
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data.dataset import Dataset

from trainer.utils import (
    PAD_TOKEN,
    UNK_TOKEN,
    EOS_TOKEN,
    SOS_TOKEN,
    OrderedCounter,
)


class RubricDataset(Dataset):
    r"""Dataset for training a model on a synthetic dataset using a rubric

    @param data_path: string
                      path to the rubric sampled dataset
    @param label_dim: int
                      number of label dimensions
    @param vocab: ?object [default: None]
                  vocabulary object
    @param max_seq_len: integer [default: 50]
                        maximum number of tokens in a sequence.
    @param min_occ: integer [default: 3]
                    minimum number of times for a token to
                    be included in the vocabulary.
    """
    def __init__(self, data_path, label_dim, vocab=None, max_seq_len=50, min_occ=3):
        super(RubricDataset, self).__init__()

        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.min_occ = min_occ
        self.label_dim = label_dim
        self.data_path = data_path

        with open(self.data_path, 'rb') as fp:
            data = pickle.load(fp)
        
        programs = data['programs']
        labels   = data['labels']
        
        if self.vocab is None:
            self.vocab = self.create_vocab(programs)
        
        self.w2i, self.i2w = self.vocab['w2i'], self.vocab['i2w']
        self.vocab_size = len(self.w2i)

        token_seqs, token_lens = self.process_programs(programs)
        labels = self.process_labels(labels)
        
        self.token_seqs = token_seqs
        self.token_lens = token_lens
        self.labels = labels

    def create_vocab(self, data):
        w2c = OrderedCounter()
        w2i, i2w = dict(), dict()

        special_tokens = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        for program in data:
            tokens = program.split()
            w2c.update(tokens)

        for w, c in w2c.items():
            if c > self.min_occ:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)
        print("Vocabulary of %i keys created." % len(w2i))

        vocab = dict(w2i=w2i, i2w=i2w)
        return vocab

    def process_programs(self, data):
        seqs, lengths = [], []

        n = len(data)
        for i in range(n):
            program = data[i]
            tokens = program.split()
            tokens = [SOS_TOKEN] + tokens[:self.max_seq_len] + [EOS_TOKEN]
            length = len(tokens)

            tokens.extend([PAD_TOKEN] * (self.max_seq_len + 2 - length))
            tokens = [self.w2i.get(token, self.w2i[UNK_TOKEN]) for token in tokens]

            seqs.append(tokens)
            lengths.append(length)

        return seqs, lengths

    def process_labels(self, annotations):
        labels = np.array(annotations)
        labels = torch.from_numpy(labels).float()
        return labels

    def __len__(self):
        return len(self.token_seqs)

    def __getitem__(self, index):
        token_seq, token_len = self.token_seqs[index], self.token_lens[index]
        token_seq = torch.from_numpy(np.array(token_seq)).long()
        label = self.labels[index]
        return token_seq, token_len, label


class ValidationDataset(RubricDataset):
    """Copy of RubricDataset but used only for validating on real student data"""
    def __init__(self, data_path, label_dim, vocab, max_seq_len=50, min_occ=3):
        assert vocab is not None
        super(ValidationDataset, self).__init__(data_path, label_dim, vocab, 
                                                max_seq_len=max_seq_len, min_occ=min_occ)
