"""
reference:
    (512, 512, 3) -> (3, 512, 512):
        http://taewan.kim/post/transpose_reshape/
"""

import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    

class CNN(nn.Module):
    def __init__(self, in_channels=4, out_channels=12, features=[64, 128, 256, 512]):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.layers.append(DoubleConv(in_channels, feature))
            in_channels = feature
        self.fc_layer = nn.Linear(features[-1], out_channels)
        
    def forward(self,x):
        ## (2xConv + Pool) x 4
        for layer in self.layers:
            x = layer(x)
            x = self.pool(x)
 
        ## Global Max Pooling - https://discuss.pytorch.org/t/global-max-pooling/1345
        x = F.max_pool2d(x, kernel_size=x.size()[2:])

        ## By using view method, change the shape of tensor to [batch_size,rest_of_tensor]
        x = x.view(x.size()[0],-1)

        ## Fully Connected Layer
        x = self.fc_layer(x)

        return x