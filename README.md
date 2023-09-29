# Grounding LTLf specifications in image sequences
This repository holds the implementation of the paper "Grounding LTLf specifications in image sequences, Elena Umili, Roberto Capobianco and Giuseppe De Giacomo, accepted by the 20th International Conference on Principles of Knowledge Representation and Reasoning (KR2023)"
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
- one with the **complete dataset**: the latter is constructed with all the MNIST training images and 50% of the traces with length in 1-4 steps having mutually exclusive symbols. You can reproduce this experiment by running:
```
python experiments.py --MUTUALLY_EXCLUSIVE_SYMBOLS=true --TRAIN_SIZE_TRACES '0.5' --TRAIN_ON_RESTRICTED_DATASET=false --LOG_DIR 'Results_complete/' --PLOTS_DIR 'Plots_complete/'
```
- one with a **restricted dataset** composed with the MNIST test images and 40% of the traces with length between 1 and 4, having mutually exclusive symbols. You can reproduce this experiment by running:
```
python experiments.py --MUTUALLY_EXCLUSIVE_SYMBOLS=true --TRAIN_SIZE_TRACES '0.4' --TRAIN_ON_RESTRICTED_DATASET=true --LOG_DIR 'Results_restricted/' --PLOTS_DIR 'Plots_restricted/'
```
- in the last experiment we test the system on traces with **non mutually exclusive symbols** by using the **complete dataset** settings. You reproduce it with:
```
python experiments.py --MUTUALLY_EXCLUSIVE_SYMBOLS=false --TRAIN_SIZE_TRACES '0.5' --TRAIN_ON_RESTRICTED_DATASET=false --LOG_DIR 'Results_complete_non_mutex/' --PLOTS_DIR 'Plots_complete_non_mutex/'
```
## How to cite us
```
@inproceedings{KR2023-65,
    title     = {{Grounding LTLf Specifications in Image Sequences}},
    author    = {Umili, Elena and Capobianco, Roberto and De Giacomo, Giuseppe},
    booktitle = {{Proceedings of the 20th International Conference on Principles of Knowledge Representation and Reasoning}},
    pages     = {668--678},
    year      = {2023},
    month     = {8},
    doi       = {10.24963/kr.2023/65},
    url       = {https://doi.org/10.24963/kr.2023/65},
  }
```
```
@inproceedings{DBLP:conf/nesy/UmiliCG22,
  author       = {Elena Umili and
                  Roberto Capobianco and
                  Giuseppe De Giacomo},
  title        = {Grounding LTLf Specifications in Images},
  booktitle    = {Proceedings of the 16th International Workshop on Neural-Symbolic
                  Learning and Reasoning as part of the 2nd International Joint Conference
                  on Learning {\&} Reasoning {(IJCLR} 2022), Cumberland Lodge, Windsor
                  Great Park, UK, September 28-30, 2022},
  pages        = {45--63},
  year         = {2022},
  crossref     = {DBLP:conf/nesy/2022},
  url          = {https://ceur-ws.org/Vol-3212/paper4.pdf},
  timestamp    = {Fri, 10 Mar 2023 16:23:33 +0100},
  biburl       = {https://dblp.org/rec/conf/nesy/UmiliCG22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

```
