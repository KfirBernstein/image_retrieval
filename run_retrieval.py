import os
import argparse
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from tqdm import tqdm

from net import my_net_type_1,my_net_type_2
from DataLoader import load_data,load_original_data

class_types ={'AlexNet':1,
              'VGG19':1,
              'VGG16': 1,
              'ResNet18':0,
              'GoogLeNet': 0}

def save_top_imgs(trainloader,testloader,sort_indices,i,k,path = ''):
    if not os.path.isdir(path):
        os.mkdir(path)
    if not os.path.isdir(path+'/'+str(i)):
        os.mkdir(path+'/'+str(i))
    it = iter(testloader)
    images, _ = it.next()
    torchvision.utils.save_image(images[i], path+'/'+str(i)+'/img_test.png')
    it = iter(trainloader)
    images, _ = it.next()
    for j in range(k):
        index = sort_indices[j]
        torchvision.utils.save_image(images[index], path+'/'+str(i)+'/img_train_' + str(j+1) + '.png')

def save_top_retrieved_labels(trainloader,sort_indices,i,k,path):
    it = iter(trainloader)
    _, labels = it.next()
    retrieved_labels = np.zeros(k)
    for j in range(k):
        index = sort_indices[j]
        retrieved_labels[j] = labels[index]
    np.save(path+'/'+str(i)+'/labels', retrieved_labels)

def binary_output(dataloader):
    if class_types[args.mid]:
        net = my_net_type_1(args.bits,args.mid)
    else:
        net = my_net_type_2(args.bits,args.mid)
    net.load_state_dict(torch.load('./{}/{}'.format(args.path, args.name)))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()
    full_batch_output = torch.cuda.FloatTensor()
    full_batch_label = torch.cuda.LongTensor()
    net.eval()
    for (inputs, targets) in tqdm(dataloader, total=len(dataloader)):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs, _ = net(inputs)
        full_batch_output = torch.cat((full_batch_output, outputs.data), 0)
        full_batch_label = torch.cat((full_batch_label, targets.data), 0)
    return torch.round(full_batch_output), full_batch_label #round: assign 0 or 1

def precision(train_binary, train_label, test_binary, test_label):
    train_binary = train_binary.cpu().numpy()
    train_binary = np.asarray(train_binary, np.int32)
    train_label = train_label.cpu().numpy()
    test_binary = test_binary.cpu().numpy()
    test_binary = np.asarray(test_binary, np.int32)
    test_label = test_label.cpu().numpy()
    trainloader, testloader = load_original_data()
    if args.rand:
        classes = np.max(test_label) + 1
        ## take random images for each class
        for i in range(classes):
            if i == 0:
                test_sample_binary = test_binary[np.random.RandomState(seed=i).permutation(np.where(test_label==i)[0])[:args.rand]]
                test_sample_label = np.array([i]).repeat(args.rand)
            else:
                test_sample_binary = np.concatenate([test_sample_binary,
                                  test_binary[np.random.RandomState(seed=i).permutation(np.where(test_label==i)[0])[:args.rand]]])
                test_sample_label = np.concatenate([test_sample_label, np.array([i]).repeat(args.rand)])
    else:
        test_sample_binary = test_binary
        test_sample_label = test_label
    num_queries = test_sample_binary.shape[0]
    trainset_len = train_binary.shape[0]
    AP = np.zeros(num_queries)
    num_samples = np.arange(1, trainset_len + 1)
    sum_tp = np.zeros(trainset_len)
    for i in range(num_queries):
        print('Query ', i+1)
        query_label = test_sample_label[i]
        query_binary = test_sample_binary[i,:]
        query_result = np.count_nonzero(query_binary != train_binary, axis=1)
        sort_indices = np.argsort(query_result)
        if i in [5,1005,3005,5005,7005,9005]:
            save_top_imgs(trainloader, testloader,sort_indices, i, 5, path=args.path+'/retrieval')
            save_top_retrieved_labels(trainloader, sort_indices, i,1000 ,path=args.path+'/retrieval')
        buffer_yes = np.equal(query_label, train_label[sort_indices]).astype(int)
        P = np.cumsum(buffer_yes) / num_samples
        AP[i] = np.sum(P * buffer_yes) /sum(buffer_yes)
        sum_tp = sum_tp + np.cumsum(buffer_yes)
    precision_at_k = sum_tp / num_samples / num_queries
    index = [100, 200, 400, 600, 800, 1000]
    index = [i - 1 for i in index]
    print('precision at k:', precision_at_k[index])
    np.save(args.path+'/precision_at_k', precision_at_k)
    np.save(args.path + '/AP_per_query', AP)
    map = np.mean(AP)
    print('MAP:', map)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Image Retrieval')
    parser.add_argument('--name', type=str, default='my_model.pkl', metavar='model_name',help='model name (default = my_model.pkl)')
    parser.add_argument('--batch', type=int, default=100, metavar='batch',help='batch')
    parser.add_argument('--bits', type=int, default=48, metavar='bts',help='binary bits')
    parser.add_argument('--path', type=str, default='', metavar='P',help='path directory')
    parser.add_argument('--mid', type=str, default='AlexNet', metavar='mid_net',help='middle net to use (default = AlexNet)')
    parser.add_argument('--rand', type=int, default=0, metavar='rand',help='number of random images per class (default = 0 means all)')

    args = parser.parse_args()
    args.path = args.path+'model_'+args.mid+'_'+str(args.batch)
    if os.path.exists(args.path+'/result/train_binary') and os.path.exists(args.path+'/result/train_label') and \
       os.path.exists(args.path+'/result/test_binary') and os.path.exists(args.path+'/result/test_label'):
        train_binary = torch.load(args.path+'/result/train_binary')
        train_label = torch.load(args.path+'/result/train_label')
        test_binary = torch.load(args.path+'/result/test_binary')
        test_label = torch.load(args.path+'/result/test_label')
    else:
        trainloader, testloader = load_data(args.batch)
        train_binary, train_label = binary_output(trainloader)
        test_binary, test_label = binary_output(testloader)
        if not os.path.isdir(args.path+'/result'):
            os.mkdir(args.path+'/result')
        torch.save(train_binary, args.path+'/result/train_binary')
        torch.save(train_label, args.path+'/result/train_label')
        torch.save(test_binary, args.path+'/result/test_binary')
        torch.save(test_label, args.path+'/result/test_label')
    precision(train_binary, train_label, test_binary, test_label)
