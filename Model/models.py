import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules import linear
from torchvision import models

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
    def __init__(self, n_classes:int = 10,
                 dropout:float = 0.2,
                 use_pretrained: bool = False
        ):
        super(Alexnet, self).__init__()
        self.n_classes = n_classes
        self.use_pretrained = use_pretrained
        if not self.use_pretrained:
            self.convolution = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
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
        else:
            alexnet = models.alexnet(pretrained=True)
            convolution_layers = list(alexnet.children())[:-1]
            linear_layers = list(alexnet.children())[-1][:-2]
            self.convolution = nn.Sequential(*convolution_layers)
            self.linear1 = nn.Sequential(*linear_layers)
            for param in self.convolution.parameters():
                param.requires_grad = False
            
            for param in self.linear1.parameters():
                param.requires_grad = False

            self.linear = nn.Linear(4096, 10)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, images):
        if not self.use_pretrained:
            outs = self.convolution(images)
            outs = self.avgpool(outs)
            outs = outs.reshape(outs.size(0), -1)
        else:
            outs = self.convolution(images)
            outs = outs.reshape(outs.size(0), -1)
            outs = self.linear1(outs)
        return self.linear(self.dropout(outs))
        

        