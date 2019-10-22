# Boilerplate for Neural Network training for Homework 4

We assume that you have a raw dataset generated from your grammar, composed of a list of (pseudocode, label) pairs. Precisely, there should be only be 18 labels. Your grammar can contain more, but when saving it to a pickle file, please only dump a list of these 18 labels (remove all others).

```
{
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
```

The contents of the pickle file should be a dictionary with 2 keys: `program` and `label`, the former is a list of Pseudoprogram strings; the latter is a list of list of strings (each of which must be one of 18 labels). Put this pickle file into the `data/` sub-directory.

## Helper Modules

We have abstracted a lot of preprocessing and training into two sub-modules: `codeDotOrg` and `trainer`. We encourage the avid reader to poke around! Please email Mike / Chris if there are bugs.

## Writing the Model

Currently, the `models.py` is empty! Your job is to specify the forward pass and the model architecture. We give you access to a few things: `vocab_size` which is the number of unique tokens from your synthetic dataset and `num_labels`, which should be 18. 

## How to Get Started

Preprocess your raw rubric-sampled data. This will dump three files into the `data/` directory: a training, validation, and testing pickle file. The contents of each will two keys again: `program` will be a list of "flattened program strings"; `label` remains unchanged.

```
python preprocess.py <path_to_raw_data.pickle>
```

Use the `trainer` library to train your model. This assumes you have run the preprocess script. This will dump a bunch of files into the `checkpoints` directory, notably a `checkpoint.pth.tar` file that represents the last iteration and a `model_best.pth.tar` file that represents the best iteration (measured by performance on a validation set).
```
python train.py
```

Use the `trainer` library to test transfer performance on a small  set (500 examples) of real student programs. Your implementation will be graded on a similar (but larger) set of hidden examples (not given to you).
```
python transfer.py
```