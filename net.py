import torch.nn as nn
from torchvision import models

features_sizes ={'AlexNet':256 * 6 * 6,
                'VGG19':512 * 7 * 7,
                'VGG16': 512 * 7 * 7,
                'ResNet18':512 * 1 * 1,
                'GoogLeNet': 1024 * 1 * 1}

mid_nets ={'AlexNet':models.alexnet(pretrained=True),
           'VGG19':models.vgg19(pretrained=True),
           'VGG16': models.vgg16(pretrained=True),
           'ResNet18': models.resnet18(pretrained=True),
           'GoogLeNet': models.googlenet(pretrained=True)}

class my_net_type_1(nn.Module):
    def __init__(self, bits,mid_net_name):
        super(my_net_type_1, self).__init__()
        self.bits = bits
        self.mid_net_name = mid_net_name
        mid_net = mid_nets[mid_net_name]
        self.features = nn.Sequential(*list(mid_net.features.children()))
        self.remain = nn.Sequential(*list(mid_net.classifier.children())[:-1])
        self.Linear1 = nn.Linear(4096, self.bits)
        self.sigmoid = nn.Sigmoid()
        self.Linear2 = nn.Linear(self.bits, 10)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), features_sizes[self.mid_net_name])
        x = self.remain(x)
        x = self.Linear1(x)
        features = self.sigmoid(x)
        result = self.Linear2(features)
        return features, result

class my_net_type_2(nn.Module):
    def __init__(self, bits,mid_net_name):
        super(my_net_type_2, self).__init__()
        self.bits = bits
        self.mid_net_name = mid_net_name
        mid_net = mid_nets[mid_net_name]
        all_layers = mid_net.children()
        self.features = nn.Sequential(*list(all_layers)[:-1])
        self.Linear1 = nn.Linear(features_sizes[self.mid_net_name], self.bits)
        self.sigmoid = nn.Sigmoid()
        self.Linear2 = nn.Linear(self.bits, 10)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), features_sizes[self.mid_net_name])
        x = self.Linear1(x)
        features = self.sigmoid(x)
        result = self.Linear2(features)
        return features, result