# Chainer IMSAT

Implementation of [Learning Discrete Representations via Information Maximizing Self-Augmented Training (IMSAT)](https://arxiv.org/abs/1702.08720) in Chainer.

## Requirements

- chainer 3.0.0
- cupy 2.0.0
- munkres

## How to train

- Train MNIST dataset with GPU

``` shell
python train_cluster.py --epoch 50 --batchsize 250 --gpu 0 --dataset mnist
```

- Train 20news dataset with GPU

``` shell
python train_cluster.py --epoch 50 --batchsize 250 --gpu 0 --dataset 20news
```

## Reference

- [Weihua Hu, Takeru Miyato, Seiya Tokui, Eiichi Matsumoto and Masashi Sugiyama. Learning Discrete Representations via Information Maximizing Self-Augmented Training. In ICML, 2017](https://arxiv.org/abs/1702.08720)
- [weihua916/imsat: Reproducing code for the paper "Learning Discrete Representations via Information Maximizing Self-Augmented Training"](https://github.com/weihua916/imsat)
- [musyoku/IMSAT: Chainer implementation of Information Maximizing Self Augmented Training](https://github.com/musyoku/IMSAT)
- [crcrpar/imsat at renewal](https://github.com/crcrpar/imsat/tree/renewal)
