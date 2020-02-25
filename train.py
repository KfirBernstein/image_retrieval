import os
# import shutil
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm

from net import my_net_type_1,my_net_type_2
from DataLoader import load_data
from torch.autograd import Variable
import torch.optim.lr_scheduler

class_types ={'AlexNet':1,
              'VGG19':1,
              'VGG16': 1,
              'ResNet18':0,
              'GoogLeNet': 0}

start_epoch = 1
train_acc_list = []
train_loss_list = []
test_acc_list = []
test_loss_list = []
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for (inputs, targets) in tqdm(trainloader, total=len(trainloader)):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        optimizer.zero_grad()
        _, outputs = net(inputs)
        loss = softmaxloss(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += softmaxloss(outputs, targets).item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
    acc = 100*int(correct)/int(total)
    print(epoch, 'Train Loss: %.3f | Train Acc: %.3f%%'
        % (train_loss/len(trainloader), acc))
    train_loss_list.append(train_loss/len(trainloader))
    train_acc_list.append(acc)
    return train_loss/len(trainloader)

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for (inputs, targets) in tqdm(testloader, total=len(testloader)):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        _, outputs = net(inputs)
        loss = softmaxloss(outputs, targets)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
    acc = 100*int(correct) / int(total)
    print('Test Loss: %.3f | Test Acc: %.3f%%'% (test_loss/len(testloader), acc))
    test_loss_list.append(test_loss/len(testloader))
    test_acc_list.append(acc)
    if epoch == args.epoch:
        print('Saving')
        if not os.path.isdir('{}'.format(args.path)):
            os.mkdir('{}'.format(args.path))
        torch.save(net.state_dict(), './{}/{}'.format(args.path, args.name))
def outputListToFile(my_list,output):
    with open(output, 'w') as f:
        for item in my_list:
            f.write("%s\n" % item)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Models For Image Retrieval')
    parser.add_argument('--name', type=str, default='my_model.pkl', metavar='model_name',help='model name (default = my_model.pkl)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',help='SGD momentum (default: 0.9)')
    parser.add_argument('--epoch', type=int, default=50, metavar='epoch',help='epoch (default: 50)')
    parser.add_argument('--batch', type=int, default=200, metavar='batch',help='batch (default: 100)')
    parser.add_argument('--bits', type=int, default=48, metavar='bts',help='binary bits')
    parser.add_argument('--path', type=str, default='', metavar='P',help='path directory')
    parser.add_argument('--mid', type=str, default='AlexNet', metavar='mid_net',help='middle net to use (default: AlexNet)')

    args = parser.parse_args()
    args.path = args.path+'model_'+args.mid+'_'+str(args.batch)
    trainloader, testloader = load_data(args.batch)
    if class_types[args.mid]:
        net = my_net_type_1(args.bits,args.mid)
    else:
        net = my_net_type_2(args.bits,args.mid)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()
    softmaxloss = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[64], gamma=0.1)
    for epoch in range(start_epoch, start_epoch + args.epoch):
        train(epoch)
        test(epoch)
        scheduler.step()
    outputListToFile(train_acc_list, args.path+'/train_acc.txt')
    outputListToFile(train_loss_list, args.path+'/train_loss.txt')
    outputListToFile(test_acc_list, args.path+'/test_acc.txt')
    outputListToFile(test_loss_list, args.path+'/test_loss.txt')

