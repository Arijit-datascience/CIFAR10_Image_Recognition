# Module to define model architecture for CIFAR10 data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

"""Class to define Model architecture for classification of cifar10_mean_std"""
class depthwise_separable_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pad_size, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=pad_size, groups=in_channels))
                      
        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                          nn.ReLU(),
                          nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Net(nn.Module):
    def __init__(self, dropout_value = 0.05):
        super(Net, self).__init__()

        # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
            
        ) # output_size = 32, RF = 5

        # TRANSITION BLOCK 1
        self.transblock1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), dilation=2, groups=32, bias=False)
        )

        # CONVOLUTION BLOCK 2
        self.convblock2 = nn.Sequential(
            depthwise_separable_conv(in_channels=32, out_channels=64, kernel_size=3, pad_size=0, bias=False),

            depthwise_separable_conv(in_channels=64, out_channels=128, kernel_size=3, pad_size=0, bias=False),
            
            nn.Dropout(dropout_value)
        ) # output_size = 18, RF = 14

        # TRANSITION BLOCK 2
        self.transblock2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), dilation=2, groups=32, bias=False)
            
        ) # output_size = 10, RF = 21

        # CONVOLUTION BLOCK 3 -> DWS and Dialted Conv
        self.convblock3 = nn.Sequential(
            depthwise_separable_conv(in_channels=32, out_channels=64, kernel_size=3, pad_size=0, bias=False),

            depthwise_separable_conv(in_channels=64, out_channels=128, kernel_size=3, pad_size=0, bias=False),
            nn.Dropout(dropout_value)

        ) # output_size = 9, RF = 40

        # TRANSITION BLOCK 3
        self.transblock3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), dilation=2, groups=64, bias=False)

        )# output_size = 2, RF = 27
        
        # CONVOLUTION BLOCK 4
        self.convblock4 = nn.Sequential(
            
            nn.Conv2d(64, 32, 3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 16, 3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(16)

        ) # output_size = 6, RF = 76

        # OUTPUT BLOCK
        self.opblock = nn.Sequential(
            nn.AvgPool2d(kernel_size=8), # output_size = 1

            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 1, RF = 116


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