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
