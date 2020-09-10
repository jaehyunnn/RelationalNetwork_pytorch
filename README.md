# Relation Networks Implementation

<p align="center">
  <img src="https://d3i71xaburhd42.cloudfront.net/007112213ece771be72cbecfd59f048209facabd/6-Figure2-1.png" width="800">
</p>



This is the implementation of the paper: 

 ["**A simple neural network module for relational reasoning**,"](https://arxiv.org/pdf/1706.01427.pdf) A. Santoro *et al*., 2017 <br>



## Required package ###
  - Python 3
  - PyTorch ,torchvision
  - termcolor ,tqdm


## Usage ###
  - **train.py** is the main training script

```bash
# select model_type (base, improved, improved2)

$ python train.py --model-type 'base'
$ python train.py --model-type 'improved'
$ python train.py --model-type 'improved2'
```

<br>

  - **eval.py** evaluates on Sort-of-CLEVR dataset

```bash
# select model_type (base, improved, improved2)

$ python eval.py --model-type 'base' --load-model 'trained_model/base_model.pth.tar'
$ python eval.py --model-type 'improved' --load-model 'trained_model/improved_model.pth.tar' 
$ python eval.py --model-type 'improved2' --load-model 'trained_model/improved_model2.pth.tar' 
```


## Experimental Results ###

| Models                                | Overall | Non-relational question | Relational question |
| ------------------------------------- | ------- | ----------------------- | ------------------- |
| Reproduced RNs (*base*)               | 96.4 %  | 99.5 %                  | 93.4 %              |
| RNs + Weighed pairs (*improved*)      | 97.5 %  | 99.8 %                  | 95.1 %              |
| RNs + Enhanced features (*improved2*) | 97.7 %  | 99.8 %                  | 95.6 %              |

## Files 

```
.
├── datsets/
    ├── sort-of-clevr_test.pickle
    └── sort-of-clevr_train.pickle
├── util/
    ├── torch_util.py
    └── train_test_fn.py
├── models/
    ├── base_model.py
    └── improved_model.py
├── trained_models/
    ├── base_model.pth.tar
    └── improved_model.pth.tar
├── so_clevr_dataset.py
├── eval.py
├── train.py
└── README.md
```
## Note

If you need the trained model (chekpoint) or dataset (sort-of-clevr), feel free to send me an e-mail.

## Author ##

[@ Jae-Hyun Park](https://github.com/jaehyunnn)