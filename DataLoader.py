from torchvision import datasets, models, transforms
import torch

def load_data(batch = 100):
    transform_train = transforms.Compose(
        [transforms.Resize(256),transforms.RandomCrop(227),transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = transforms.Compose(
        [transforms.Resize(227),transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True,transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch,shuffle=False, num_workers=2)
    testset = datasets.CIFAR10(root='./data', train=False, download=True,transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch,shuffle=False, num_workers=2)
    return trainloader, testloader

def load_original_data():
    trainset = datasets.CIFAR10(root='./data', train=True, download=True,transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=50000,shuffle=False, num_workers=2)
    testset = datasets.CIFAR10(root='./data', train=False, download=True,transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=10000,shuffle=False, num_workers=2)
    return trainloader, testloader