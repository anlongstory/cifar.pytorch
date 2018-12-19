import torch.nn as nn
import torch

vgg_11=[64,'m',128,'m',256,256,'m',512,512,'m',512,512,'m']
vgg_13=[64,64,'m',128,128,'m',256,256,'m',512,512,'m',512,512,'m']
vgg_16=[64,64,'m',128,128,'m',256,256,256,'m',512,512,512,'m',512,512,512,'m']
vgg_19=[64,64,'m',128,128,'m',256,256,256,256,'m',512,512,512,512,'m',512,512,512,512,'m']

class VGGnet(nn.Module):
    def __init__(self, vgg_type):
        super(VGGnet,self).__init__()

        num_classes = 10
        self.feature = self.build_net(vgg_type)
        self.classifier = nn.Linear(512,num_classes)


    def forward(self,x):
        out = self.feature(x)
        out = out.view(out.size(0),-1)
        out = self.classifier(out)

        return out

    def  build_net(self,vgg_type):
        layer=[]
        in_channel = 3
        for channel in vgg_type:
            if  channel != 'm':
                layer.append(nn.Conv2d(in_channel,channel,kernel_size=3,padding=1))
                layer.append(nn.BatchNorm2d(channel))
                layer.append(nn.ReLU(True))
                in_channel = channel

            else:
                layer.append(nn.MaxPool2d(kernel_size=2,stride=2))

        layer.append(nn.AvgPool2d(kernel_size=1,stride=1))
        return nn.Sequential(*layer)

def test():
    net = VGGnet(vgg_13)
    print(net)

    x = torch.randn(1,3,32,32)
    y=net(x)
    print(y.size())

# test()