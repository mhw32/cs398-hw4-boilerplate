"""Test on real data validation set!"""

import os
import torch
import pickle

from models import FeedbackNN
from config import TRAINING_PARAMS, DATA_DIR, CHECKPOINT_DIR

import trainer


if __name__ == "__main__":
    real_data_path = os.path.join(DATA_DIR, 'student_val_data.pickle')
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'model_best.pth.tar')  # checkpoint.pth.tar

    results = trainer.lib.transfer_pipeline(FeedbackNN, checkpoint_path, real_data_path)
    print(results)
