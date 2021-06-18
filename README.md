# EVA6_Session7_Advanced_Concepts

Time to try our hands on something more than just digits. How about some cars ... planes ... maybe a few animals here and there? Welcome to our experimentation of Advanced Concepts using CIFAR10 dataset.

### Understanding the CIFAR-10 dataset

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Here are the classes in the dataset, as well as 10 random images from each:

![image](https://user-images.githubusercontent.com/31658286/122556219-dab61e80-d058-11eb-8e6e-a2ac3ab24365.png)

The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.

_Source_: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)


## Objectives

- [ ] A GPU based code with Model architecture of C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead. It would be a bonus if we can figure out how to use Dilated kernels instead of MP or strided convolution)
- [ ] Total Receptive Field of more than 44
- [ ] One of the layers must use Depthwise Separable Convolution
- [ ] One of the layers must use Dilated Convolution
- [ ] Have to use GAP. Optional: Add FC after GAP to target # of classes
- [ ] Use albumentation library and apply:
  - [ ] Horizontal flip
  - [ ] shiftScaleRotate
  - [ ] coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
- [ ] Minimun 85% Test Accuracy
- [ ] Total Parameters below 200K


## Code Structure


## Observations

## Logs

## Misclassified Images

## Conculsions and notes

### Collaborators
Abhiram Gurijala  
Arijit Ganguly  
Rohin Sequeira  
