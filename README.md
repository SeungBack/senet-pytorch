## SENet-pytorch

- Implementation of SE-ResNet-50 in pytorch on CIFAR-100 dataset

## Requirements

This codes were tested on
- python 3.7
- pytorch 1.2.0
- torchvision 0.4.0
- CUDA 10.1

```
conda create -n senet python=3.7
pip install torch, torchvision, tqdm, tensorboard, tensorboardX, thop
mkdir results dataset
CUDA_VISIBLE_DEVICES=1 python main.py -bs 256 --block 'se'
```

## References

Gradcam: https://github.com/yiskw713/ClassActivationMapping
