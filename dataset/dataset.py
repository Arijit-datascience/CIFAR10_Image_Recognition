import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Calculating Mean and Standard Deviation
def cifar10_mean_std():
    simple_transforms = transforms.Compose([
                                           transforms.ToTensor(),
                                           ])
    exp_train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=simple_transforms)
    exp_test = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=simple_transforms)

    train_data = exp_train.data
    test_data = exp_test.data

    exp_data = np.concatenate((train_data,test_data),axis=0) # contatenate entire data
    exp_data = np.transpose(exp_data,(3,1,2,0)) # reshape to (60000, 32, 32, 3)

    norm_mean = (np.mean(exp_data[0])/255, np.mean(exp_data[1])/255, np.mean(exp_data[2])/255)
    norm_std   = (np.std(exp_data[0])/255, np.std(exp_data[1])/255, np.std(exp_data[2])/255)

    return(tuple(map(lambda x: np.round(x,3), norm_mean)), tuple(map(lambda x: np.round(x,3), norm_std)))

def get_transforms(norm_mean,norm_std):
    """get the train and test transform"""
    print(norm_mean,norm_std)
    train_transform = A.Compose(
        [
        A.HorizontalFlip(p=0.5),
        A.RandomCrop(height=128, width=128),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.25),
        A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=(norm_mean[0]*255.0,norm_mean[1]*255.0,norm_mean[2]*255.0)),
        A.Normalize(norm_mean, norm_std),
        ToTensorV2()
    ]
    )

    test_transform = A.Compose(
        [
        A.Normalize(norm_mean, norm_std),
        ToTensorV2()
    ]
    )
    
    return(train_transform,test_transform)

def get_datasets(train_transform,test_transform):
    
    class Cifar10_SearchDataset(datasets.CIFAR10):
        def __init__(self, root="./data", train=True, download=True, transform=None):
            super().__init__(root=root, train=train, download=download, transform=transform)
            
        def __getitem__(self, index):
            image, label = self.data[index], self.targets[index]

            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed["image"]

            return image, label
    
    train_set = Cifar10_SearchDataset(root='./data', train=True,download=True, transform=train_transform)
    test_set  = Cifar10_SearchDataset(root='./data', train=False,download=True, transform=test_transform)

    return(train_set,test_set)

def get_dataloaders(train_set,test_set):

    SEED = 1
    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)

    # dataloader arguments
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=2, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64, num_workers=1)

    # dataloaders
    train_loader = torch.utils.data.DataLoader(train_set, **dataloader_args)

    test_loader  = torch.utils.data.DataLoader(test_set, **dataloader_args)
    return(train_loader,test_loader)
