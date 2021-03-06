# Module to define model architecture for CIFAR10 data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

"""Class to define Model architecture for classification of cifar10_mean_std"""
class Net(nn.Module):
    def __init__(self, dropout_value = 0.01):
        super(Net, self).__init__()

        # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
        ) # Input: 32x32x3 | Output: 32x32x64 | RF: 5x5

        # TRANSITION BLOCK 1
        self.transblock1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), stride=2),
            nn.ReLU(),
        ) # Input: 32x32x64 | Output: 16x16x32 | RF: 10x10

        # CONVOLUTION BLOCK 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value), # Input: 8x8x32 | Output: 6x6x64 | RF: 34x34

            #Depthwise Seperable Convolution
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, groups=32, bias=False),
            nn.Conv2d(32, 64, kernel_size=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value), # Input: 16x16x32 | Output: 16x16x64 | RF: 14x14
        )

        # TRANSITION BLOCK 2
        self.transblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), stride=2),
            nn.ReLU(),
        ) # Input: 16x16x32 | Output: 8x8x32 | RF: 17x17

        # CONVOLUTION BLOCK 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value), # Input: 8x8x32 | Output: 6x6x64 | RF: 34x34
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value), # Input: 16x16x64 | Output: 16x16x64 | RF: 36x36

            #Depthwise Seperable Convolution
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, groups=64, bias=False),
            nn.Conv2d(64, 32, kernel_size=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),# Input: 16x16x32 | Output: 16x16x64 | RF: 14x14
        )

        # TRANSITION BLOCK 3
        self.transblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), stride=2),
            nn.ReLU(),
        )# Input: 16x16x64 | Output: 8x8x32 | RF: 44x44

        # CONVOLUTION BLOCK 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
        ) # Input: 8x8x32 | Output: 8x8x64 | RF: 46x46

        # OUTPUT BLOCK
        self.opblock = nn.Sequential(
            nn.AvgPool2d(kernel_size=5) # Input: 8x8x64 | Output: 1x1x64 | RF: 46x46
            #nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # Input: 1x1x64 | Output: 1x1x10 | RF: 46x46


    def forward(self, x):
        x = self.convblock1(x)
        x = self.transblock1(x)
        
        x = self.convblock2(x)
        x = self.transblock2(x)
        
        x = self.convblock3(x)
        x = self.transblock3(x)
        
        x = self.convblock4(x)
        x = self.opblock(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
