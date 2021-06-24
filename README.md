# EVA6_Session7_Advanced_Concepts

Time to try our hands on something more than just digits. How about some cars ... planes ... maybe a few animals here and there? Welcome to our experimentation of Advanced Concepts using CIFAR10 dataset.

### Understanding the CIFAR-10 dataset

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Here are the classes in the dataset, as well as 10 random images from each:

![image](https://user-images.githubusercontent.com/31658286/122556219-dab61e80-d058-11eb-8e6e-a2ac3ab24365.png)

The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.

_Source_: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

## Concept Time!

### Dialated Convolution

![dilated_convolution](https://user-images.githubusercontent.com/31658286/123273137-ff5e3a80-d51f-11eb-802f-bca4da9e492e.gif)

Dilated convolution is a way of increasing the receptive view (global view) of the network exponentially and linear parameter accretion. With this purpose, it finds usage in applications thats care more about integrating the knowledge of the wider context with less cost.

The key application the dilated convolution authors have in mind is a dense prediction:vision applications where the predicted object has a similar size and structure to the input image.
For example, semantic segmentation with one label per pixel;
image super-resolution, denoising, demosaicing, bottom-up saliency, keypoint detection, etc.

In many such applications one wants to integrate information from different spatial scales and balance two properties:

∙ local, pixel-level accuracy, such as precise detection of edges, and

∙ integrating the knowledge of the wider, global context

![image](https://user-images.githubusercontent.com/31658286/123273667-798ebf00-d520-11eb-81ea-84fd1e922e9b.png)

![image](https://user-images.githubusercontent.com/31658286/123273824-9aefab00-d520-11eb-9ed2-638434daa46b.png)


### Depthwise Separable Convolution

![image](https://user-images.githubusercontent.com/31658286/123274042-ca9eb300-d520-11eb-8fe0-dbe8b7d9040c.png)


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

Code is split into different modules(as it should be!). If you are looking for the final notebook, you can find it [here]().  

* [./dataset/](https://github.com/Arijit-datascience/CIFAR10_Image_Recognition/tree/main/dataset) contains the code for data downloading, prepping and preprocessing. You can find code related to transformations and augmentations here.  
   * [dataset.py](https://github.com/Arijit-datascience/CIFAR10_Image_Recognition/blob/main/dataset/dataset.py): Data loading and processing code is here.

* [./models/](https://github.com/Arijit-datascience/CIFAR10_Image_Recognition/tree/main/models) will take you to our modelling directory which contains code for our network structure and the training and testing modules.  
   * [model.py](https://github.com/Arijit-datascience/CIFAR10_Image_Recognition/blob/main/models/model.py): Network Architecture code.
   * [test.py](https://github.com/Arijit-datascience/CIFAR10_Image_Recognition/blob/main/models/test.py): Test code.
   * [train.py](https://github.com/Arijit-datascience/CIFAR10_Image_Recognition/blob/main/models/train.py): Train code.

* [./plots_accuracy/](https://github.com/Arijit-datascience/CIFAR10_Image_Recognition/tree/main/plots_accuracy) has code for our visualization needs.  
   * [plots.py](https://github.com/Arijit-datascience/CIFAR10_Image_Recognition/blob/main/plots_accuracy/plots.py): Visualization for Train, Test logs, sample images that were miss predicted.

* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/RohinSequeira/CIFAR10_Image_Recognition/blob/main/CIFAR10_Image_Recognition.ipynb)

## Logs

## Misclassified Images

## Conculsions and notes

- [x] A GPU based code with Model architecture of C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead. It would be a bonus if we can figure out how to use Dilated kernels instead of MP or strided convolution)
  - [x] _**Dilated Convolution in place of Max Pooling Achieved!**_
- [x] Total Receptive Field of more than 44
- [x] One of the layers must use Depthwise Separable Convolution
- [x] One of the layers must use Dilated Convolution
- [x] Have to use GAP. Optional: Add FC after GAP to target # of classes
- [x] Use albumentation library and apply:
  - [x] Horizontal flip
  - [x] shiftScaleRotate
  - [x] coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
- [x] Minimun 85% Test Accuracy
- [x] Total Parameters below 200K

### Collaborators
Abhiram Gurijala  
Arijit Ganguly  
Rohin Sequeira  
