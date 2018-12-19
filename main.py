from utils import train
from torchvision.datasets import CIFAR10
import torchvision.transforms as transform
from models import *

# hyperparameters

base_lr = 0.1
epoches = 131
batch_size = 128
momentum = 0.9
weight_decay = 0.0005
step = 30   # use to set the frequence of learning rate change

######################




# Models
# net = Resnet18() # or Resnet34(), Resnet50(), Resnet101(), Resnet152()
net = VGGnet(vgg_11) # or vgg_13, vgg_16, vgg_19

######################




print("====> Loading Data:")

train_transform = transform.Compose(
    [
        transform.RandomCrop(32,padding=4),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])
    ]
)
test_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

train_set = CIFAR10(r'./data',train=True,transform= train_transform, download= True)
train_data = torch.utils.data.DataLoader(train_set,batch_size = batch_size,shuffle = True)
test_set = CIFAR10(r'./data',train=False,transform=test_transform)
test_data = torch.utils.data.DataLoader(test_set,batch_size = batch_size,shuffle=False)

print("====> Data is Loaded!")


optimizer = torch.optim.SGD(net.parameters(),lr=base_lr,momentum=momentum,weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()


def adjust_lr(optimizer,epoch):
    lr = base_lr*(0.1**(epoch//step))
    for parameter in optimizer.param_groups:
        parameter['lr'] = lr

print("====> Training:")

for epoch in range(epoches):
    adjust_lr(optimizer,epoches)
    train(net,train_data,test_data,epoch,optimizer,criterion)


