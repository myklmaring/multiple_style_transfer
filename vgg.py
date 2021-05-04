import torch.nn as nn
from torchvision import models


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg = models.vgg16(pretrained=True).features

        self.slice1 = vgg[:4]
        self.slice2 = vgg[4:9]
        self.slice3 = vgg[9:16]
        self.slice4 = vgg[16:23]

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1 = self.slice1(x)
        relu2 = self.slice2(relu1)
        relu3 = self.slice3(relu2)
        relu4 = self.slice4(relu3)
        relu_dict = {'relu1': relu1, 'relu2': relu2, 'relu3': relu3, 'relu4': relu4}
        return relu_dict
