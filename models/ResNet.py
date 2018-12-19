import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,in_channel,out_channel,same_shape=True):
        super(BasicBlock,self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2

        self.conv1 = nn.Conv2d(in_channel,out_channel*self.expansion,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel*self.expansion)

        self.conv2 = nn.Conv2d(out_channel*self.expansion,out_channel*self.expansion,kernel_size=3,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel*self.expansion)
        if  not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel,out_channel*self.expansion,1,stride=stride)
            self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out),True)

        out = self.conv2(out)
        out = self.bn2(out)

        if not self.same_shape:
            x=self.bn3(self.conv3(x))

        return F.relu(x+out,True)

class BottleneckBlock(nn.Module):
    expansion =4
    def __init__(self,in_channel,out_channel,same_shape=True):
        super(BottleneckBlock,self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2

        self.conv1 = nn.Conv2d(in_channel,out_channel,1,bias= False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel,out_channel,3,stride=stride,padding=1,bias= False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(out_channel,out_channel*self.expansion,1,bias= False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        if (not same_shape) or  (in_channel != out_channel*self.expansion):
            self.conv4 = nn.Conv2d(in_channel,out_channel*self.expansion,1,stride=stride,bias=False)
            self.bn4 = nn.BatchNorm2d(out_channel*self.expansion)

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)),True)
        out = F.relu(self.bn2(self.conv2(out)),True)
        out= self.bn3(self.conv3(out))

        if not self.same_shape:
            x = self.bn4(self.conv4(x))
            return F.relu(out+x,True)
        return F.relu(out,True)

class Resnet(nn.Module):
    def __init__(self,block_type,num_of_blocks,num_classes):
        super(Resnet,self).__init__()
        self.in_channel = 64

        self.conv1 = nn.Conv2d(in_channels=3,
                                out_channels=64,
                                kernel_size=3,
                                padding=1,
                                bias= False)
        self.bn1 = nn.BatchNorm2d(64)

        self.block1 = self.build_net(block_type,64,num_of_blocks[0],same_shape=True)
        self.block2 = self.build_net(block_type,128,num_of_blocks[1],same_shape=False)
        self.block3 = self.build_net(block_type,256,num_of_blocks[2],same_shape=False)
        self.block4 = self.build_net(block_type,512,num_of_blocks[3],same_shape=False)
        self.classifier = nn.Linear(512*block_type.expansion,num_classes)


    def build_net(self,block_type,out_channel,num_of_blocks,same_shape):
        same_shapes = [same_shape]+[True]*(num_of_blocks-1)
        layer = []
        for each in same_shapes:
            layer.append(block_type(self.in_channel,out_channel,each))
            self.in_channel = out_channel * block_type.expansion
        return nn.Sequential(*layer)


    def forward(self, x):
       out = F.relu(self.bn1(self.conv1(x)),True)
       out = self.block1(out)
       out = self.block2(out)
       out = self.block3(out)
       out = self.block4(out)
       out = F.avg_pool2d(out,4)
       out=out.view(out.shape[0],-1)
       out=self.classifier(out)
       return out

def Resnet18():
    return Resnet(BasicBlock,[2,2,2,2],10)

def Resnet34():
    return Resnet(BasicBlock,[3,4,6,3],10)

def Resnet50():
    return Resnet(BottleneckBlock,[3,4,6,3],10)

def Resnet101():
    return Resnet(BottleneckBlock,[3,4,23,3],10)

def Resnet152():
    return Resnet(BottleneckBlock,[3,8,36,3],10)

def test():
    net = Resnet18()
    test_x = torch.zeros(1,3,32,32)
    # print(net)
    test_y = net(test_x)
    print(test_y.shape)

# test()



