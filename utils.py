import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


def get_classes():
    classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return classes


def get_model():
    from models import SimpleModel
    return SimpleModel()

def get_otimizer(params, lr, momentum, type):
    

    if type == 'sgd':
        optimizer = optim.SGD(params, lr=lr, momentum=momentum)
    elif type == 'adam':
        optimizer = optim.Adam(params, lr=lr)
    else:
        raise ValueError("Not support optim type {}".format(type))
    
    return optimizer


def get_transform():
    return  torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # need to change
                ]
            
            )


def get_dataset_dataloader(root='./cifar10', split='train', transform=None, batch_size=32, shuffle=True):
    if split == 'train':
        trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                                download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=shuffle, num_workers=2)
        return trainset, trainloader
    
    elif split == 'test':
        
        testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=shuffle, num_workers=2)
        return testset, testloader
    else:
        raise ValueError('Not support split name: {}. Please try train or test.'.format(split))


def get_xent_criterion():
    return nn.CrossEntropyLoss()


def plot_and_visualize():
    pass
    # need to be filled

