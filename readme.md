> **Notice:** PLCIFAR10 was collected and pre-processed by me, with courtesy and proprietary to the authors of referred literatures on them. The pre-processed data sets can be used at your own risk and for academic purpose only.
# PLCIFAR10
This repository provides the PLCIFAR10 dataset of the paper "Realistic Evaluation of Deep Partial-Label Learning Algorithms" and dataset details can be found in the paper. 

## Introduction
This repository is the official dataset release of PLCIFAR10, a partial-label version of CIFAR-10 with human-annotated partial labels. For each image from the training set of CIFAR-10, PLCIFAR10 provides about 10 candidate label sets given by individual annotators, where each candidate label set may contain multiple labels and may contain the true label. For each example, we have a list of lists, where each sublist contains partial labels given by an individual annotator.

## Example

We recommend two instances of PLCIFAR10. The first is **PLCIFAR10-Aggregate**, which assigns to each example the aggregation of all partial labels from all annotators. The second is **PLCIFAR10-Vaguest**, which assigns each example the largest candidate label set from the annotators.

Since we only provide the annotations, CIFAR-10 should be downloaded and loaded accordingly:
```python
import torchvision.datasets as dsets
import numpy as np
original_dataset = dsets.CIFAR10(root=root, train=True, download=True) # root is the path of the CIFAR-10 dataset
data = original_dataset.data
```

### PLCIFAR10-Aggregate
```python
import os
import pickle
dataset_path = os.path.join(root, f"plcifar10.pkl") # "root" should be replaced with the path of the file
partial_label_all = pickle.load(open(dataset_path, "rb"))
partial_targets = np.zeros((len(data), 10))
for key, value in partial_label_all.items():
    for candidate_label_set in value:
        for label in candidate_label_set:
        	partial_targets[key, label] = 1
```
### PLCIFAR10-Vaguest
```python
import os
import pickle
dataset_path = os.path.join(root, f"plcifar10.pkl") # "root" should be replaced with the path of the file
partial_targets = np.zeros((len(data), 10))
for key, value in partial_label_all.items():
    vaguest_candidate_label_set = []
    largest_num = 0
    for candidate_label_set in value:
        if len(candidate_label_set) > largest_num:
            vaguest_candidate_label_set = candidate_label_set
            largest_num = len(candidate_label_set)
    for label in vaguest_candidate_label_set:
        partial_targets[key, label] = 1 
```
Different customized versions of PLCIFAR10 are also welcome.

## Citation
```
@inproceedings{wang2025realistic,
    author = {Wang, Wei and Wu, Dong-Dong and Wang, Jindong and Niu, Gang and Zhang, Min-Ling and Sugiyama, Masashi},
    title = {Realistic evaluation of deep partial-label learning algorithms},
    booktitle = {Proceedings of the 13th International Conference on Learning Representations},
    year = {2025}
}

@Techreport{krizhevsky2009learning,
author = {Krizhevsky, Alex and Hinton, Geoffrey E.},
institution = {University of Toronto},
title = {Learning multiple layers of features from tiny images},
year = {2009}
}
```
