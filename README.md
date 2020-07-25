# PyTorch-SRGAN

A modern PyTorch implementation of SRGAN

## Requirements

- PyTorch
- torchvision
- tensorboard_logger (https://github.com/TeamHG-Memex/tensorboard_logger)

## Datasets

The raw data is saved in folder "raw data"

The preprocessed data is saved in folder "datas", named "datas.npz", which is dictionary, has two elements, one is hf, the other is lf. You can use datas["hf"] or datas["lf"] to access them.

## Training

```
usage: train [-h] [--dataset DATASET] 
             [--batchSize BATCHSIZE]
             [--imageSize IMAGESIZE] 
             [--upSampling UPSAMPLING]
             [--nEpochs NEPOCHS] [--generatorLR GENERATORLR]
             [--discriminatorLR DISCRIMINATORLR] [--cuda]
             [--generatorWeights GENERATORWEIGHTS]
             [--discriminatorWeights DISCRIMINATORWEIGHTS]                [--out OUT]
             [--clip_value]
```

Example: ```./train --cuda```

This will start a training session in the GPU. First it will pre-train the generator using MSE error for 2 epochs, then it will train the full GAN (generator + discriminator) for 100 epochs, using content (mse + vgg) and adversarial loss. Although weights are already provided in the repository, this script will also generate them in the checkpoints file.

## Testing (Not Implemented)

```
usage: test [-h] [--dataset DATASET] [--dataroot DATAROOT] [--workers WORKERS]
            [--batchSize BATCHSIZE] [--imageSize IMAGESIZE]
            [--upSampling UPSAMPLING] [--cuda] [--nGPU NGPU]
            [--generatorWeights GENERATORWEIGHTS]
            [--discriminatorWeights DISCRIMINATORWEIGHTS]

```

## Results

### Training

The following results have been obtained with the current training setup:

- Dataset: 270 pairs of samples consisting of high and low resolution images
- Input image size: 16x16
- Output image size: 64x64 (16x)
