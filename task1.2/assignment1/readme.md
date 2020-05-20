# Training a classifier

## Requirements

+ torch
+ torchvision

## Loading and normalizing CIFAR10

Using `torchvision`, it’s extremely easy to load CIFAR10.
Simce the output of torchvision datasets are PILImage images of range [0, 1], we need transform them to Tensors of normalized range [-1, 1]

To load and normalize CIFAR10, run this command:

```data
python data.py
```

## Define a Convolutional Neural Network

Define a Convolutional Neural Network in `net.py`.
In this part, we defined "convilution layers", "pooling layers","Linear layer" and "Non-linear Activations" by `torch.nn` and `torch.nn.functional` API.

## Training

To train the model(s) in the paper, run this command:

```train
python train.py
```

## Evaluation

To evaluate my model on cifar-10, run:

```eval
python eval.py
```

<!-- ## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

> 📋Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models. -->

## Results


| Model name  | Accuracy |
| ----------- | -------- |
| vanilla net | 59%      |

