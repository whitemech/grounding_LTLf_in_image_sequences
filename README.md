# Grounding LTLf specifications in image sequences
This repository holds the implementation of the paper "Grounding LTLf specifications in images, Elena Umili, Roberto Capobianco and Giuseppe De Giacomo, NeSy 2022"
## Requirements
To install all the dependencies run 
`pip install -r requirements.txt`
## How to reproduce the paper results
To run the experiments with the MNIST dataset and the DECLARE constraints run the file `experiments.py`.

The file accepts some parameters. You can read the list of parameters by running `python experiments.py --help`.

```experiments.py:
  --LOG_DIR: path to save the results
    (default: 'Results/')
  --MAX_LENGTH_TRACES: maximum traces length used to create the dataset
    (default: '4')
    (an integer)
  --[no]MUTUALLY_EXCLUSIVE_SYMBOLS: if True symbols are mutually exclusive in traces
    (default: 'true')
  --PLOTS_DIR: path to save the plots
    (default: 'Plots/')
  --[no]TRAIN_ON_RESTRICTED_DATASET: if True test images from MNIST are used to render symbols
    (default: 'false')
  --TRAIN_SIZE_TRACES: portion of traces used for training
    (default: '0.4')
    (a number)
```
The paper proposes three experiments:
- one with the complete dataset: the latter is constructed with all the MNIST training images and 50% of the traces with length in 1-4 steps having mutually exclusive symbols. You can reproduce this experiment by running:
```
python experiments.py --MUTUALLY_EXCLUSIVE_SYMBOLS 'true' --TRAIN_SIZE_TRACES '0.5' --TRAIN_ON_RESTRICTED_DATASET 'false' --LOG_DIR 'Results_complete/' --PLOTS_DIR 'Plots_complete/'
```
- one with a restricted dataset composed with the MNIST test images and 40% of the traces with length between 1 and 4, having mutually exclusive symbols. You can reproduce this experiment by running:
```
python experiments.py --MUTUALLY_EXCLUSIVE_SYMBOLS 'true' --TRAIN_SIZE_TRACES '0.4' --TRAIN_ON_RESTRICTED_DATASET 'true' --LOG_DIR 'Results_restricted/' --PLOTS_DIR 'Plots_restricted/'
```
- in the last experiment we test the system on traces with non mutually exclusive symbols by using the complete dataset settings. You reproduce it with
```
python experiments.py --MUTUALLY_EXCLUSIVE_SYMBOLS 'false' --TRAIN_SIZE_TRACES '0.5' --TRAIN_ON_RESTRICTED_DATASET 'false' --LOG_DIR 'Results_complete_non_mutex/' --PLOTS_DIR 'Plots_complete_non_mutex/'
```
