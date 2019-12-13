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

total number of trainable parameter is : 24265380.0
number of FLOPS :  1458194432.0

total number of trainable parameter is : 34607460.0
number of FLOPS :  1480918784.0

total number of trainable parameter is : 34607460.0
number of FLOPS :  1478838272.0

total number of trainable parameter is : 34612500.0
number of FLOPS :  1482950912.0

total number of trainable parameter is : 35058348.0
number of FLOPS :  1481809664.0

