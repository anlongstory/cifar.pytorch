from __future__ import print_function

from datetime import datetime
import torch


def get_cc(output,label):
    total = output.shape[0]
    _,pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct/total

def train(net,train_data,valid_data,epoch,optimizer,criterion):
    if torch.cuda.is_available():
        net=net.cuda()
    prev_time = datetime.now()
    train_loss = 0
    train_acc = 0
    net = net.train()
    for im,label in train_data:
        if  torch.cuda.is_available():
            im = im.cuda()
            label = label.cuda()

        # forward
        output = net(im)
        loss = criterion(output,label)

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += get_cc(output,label)

    curr_time = datetime.now()
    h,remainder = divmod((curr_time-prev_time).seconds,3600)
    m,s = divmod(remainder,60)
    time_str = " Time %02d:%02d:%02d" % (h,m,s)

    if  valid_data is not None:
        valid_loss = 0
        valid_acc = 0
        net = net.eval()
        for im,label in valid_data:
            if torch.cuda.is_available():
                im = im.cuda()
                label = label.cuda()

            output = net(im)
            loss = criterion(output,label)
            valid_loss += loss.item()
            valid_acc += get_cc(output,label)
        epoch_str=("Epoch %d. Train Loss: %f, Train.Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                   % (epoch,train_loss/len(train_data),train_acc/ len(train_data),
                      valid_loss / len(valid_data), valid_acc / len(valid_data)))
    else:
        epoch_str = ("Epoch %d. Train Loss: %f, Train.Acc: %f,  "
                     % (epoch, train_loss / len(train_data), train_acc / len(train_data)))
    f=open('./log.txt','a+')
    prev_time = curr_time
    print(epoch_str + time_str)
    f.write(epoch_str + time_str+'\n')
    f.close()




