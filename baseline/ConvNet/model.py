"""
reference:
    (512, 512, 3) -> (3, 512, 512):
        http://taewan.kim/post/transpose_reshape/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

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
    def __init__(self, in_channels=1, out_channels=12, features=[64, 128, 256, 512]):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.layers.append(DoubleConv(in_channels, feature))
            in_channels = feature
        self.fc_layer = nn.Linear(features[-1], out_channels)
        
    def forward(self,x):
        # print("input tensor shape: \n\t",x.shape)

        ## (2xConv + Pool) x 4
        for layer in self.layers:
            x = layer(x)
            x = self.pool(x)

        # print("tensor shape after 4 convs: \n\t",x.shape)
# 
        ## Global Max Pooling - https://discuss.pytorch.org/t/global-max-pooling/1345
        # print("global max pool kernel size:\n\t",x.size()[2:])
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        # print("tensor shape after global max pool:\n\t",x.shape)


        # view 함수를 이용해 텐서의 형태를 [batch_size,나머지]로 변환
        x = x.view(x.size()[0],-1)
        # print("tensor shape after reshaping:\n\t",x.shape)
# 
        ## Fully Connected Layer
        x = self.fc_layer(x)
        # print("tensor shape after MLP:\n\t",x.shape)

        return x
    
# model = CNN()

# batch_size = 1
# # a = torch.rand(batch_size, 3, 512, 512)
# # print("original tensor shape: \n\t",a.shape)
# # print(model(a))

# image_path = "/home/yehyun/2022-Fall-Research/data/padded_image/0_pad.png"
# image = np.array(Image.open(image_path).convert("L").resize((512,512)))
# print(image.shape)
# # image = np.transpose(image, (2,0,1))
# image = torch.Tensor(image).unsqueeze(0)
# print(image.shape)
# label = [19,246,303,206,301,240,331,189,329,239,500,243]
# # print(model(torch.Tensor(image)))
# print(model(image))