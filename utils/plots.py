import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def show_sample_images(data_loader, classes, mean=.5, std=.5, num_of_images = 10, is_norm = True):
    """ Display images from a given batch of images """
    smpl = iter(data_loader)
    im,lb = next(smpl)
    plt.figure(figsize=(20,20))
    if num_of_images > im.size()[0]:
        num = im.size()[0]
        print(f'Can display max {im.size()[0]} images')
    else:
        num = num_of_images
        print(f'Displaying {num_of_images} images')
    for i in range(num):
        if is_norm:
            img = im[i].squeeze().permute(1,2,0)*std+mean
        plt.subplot(10,10,i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(classes[lb[i]],fontsize=15)

def valid_accuracy_loss_plots(train_loss, train_acc, test_loss, test_acc):

    # Use plot styling from seaborn.
    sns.set(style='whitegrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1)
    plt.rcParams["figure.figsize"] = (25,6)

    # Plot the learning curve.
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.plot(np.array(train_loss), 'red', label="Training Loss")
    ax1.plot(np.array(test_loss), 'blue', label="Validation Loss")

    # Label the plot.
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_ylim(0.3,1)
    ax1.legend()

    ax2.plot(np.array(train_acc), 'red', label="Training Accuracy")
    ax2.plot(np.array(test_acc), 'blue', label="Validation Accuracy")

    # Label the plot.
    ax2.set_title("Training & Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_ylim(30,90)
    ax2.legend()

    plt.show()

def misclassification(predictions, targets, data):
    pred = predictions.view(-1)
    target = targets.view(-1)

    index = 0
    misclassified_image = []

    for label, predict in zip(target, pred):
        if label != predict:
            misclassified_image.append(index)
        index += 1

    plt.figure(figsize=(10,5))
    plt.suptitle('Misclassified Images');

    for plot_index, bad_index in enumerate(misclassified_image[0:10]):
        p = plt.subplot(2, 5, plot_index+1)
        img = data.squeeze().permute(1,2,0)
        p.imshow(img[bad_index].reshape(3,32,32))
        p.axis('off')
        p.set_title(f'Pred:{pred[bad_index]}, Actual:{target[bad_index]}')