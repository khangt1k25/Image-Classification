import torch
from torch import Tensor
import torch.nn as nn
import time
class Lenet(nn.Module):
    def __init__(self, n_classes=10, dropout=0.2):
        super(Lenet, self).__init__()

        self.n_classes = n_classes
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5), #28*28*6
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, padding=0), #14*14*6
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), #10 * 10* 16
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, padding=0), # 5*5*16
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5), #1*1*120
            nn.Tanh(),
        )
        self.linear = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, n_classes),
            nn.Dropout(dropout)
        )
    
    def forward(self, images: Tensor):
        """
        input:
            images: batch_size * 3 *32 *32 
        
        outptu:
            tensor: batch_size * n_classes
        """
        outputs = self.convolution(images)
        outputs = outputs.reshape(outputs.size(0), -1)
        return self.linear(outputs)


class Alexnet(nn.Module):
    def __init__(self, n_classes, dropout=0.2):
        super(Alexnet, self).__init__()

        self.convolution = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.linear = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_classes),
        )
    
    def forward(self, images):
        outs = self.convolution(images)
        outs = self.avgpool(outs)
        outs = outs.reshape(outs.size(0), -1)
        return self.linear(outs)
        

        