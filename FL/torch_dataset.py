import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from operator import itemgetter
import random
random.seed(0)


def get_cifar_iid(batch_size, total_num_clients):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    total_data = len(trainset)
    random_list = random.sample(range(total_data), total_data)
    data_per_client = int(total_data / total_num_clients)
    datasets = []
    for i in range(total_num_clients):

        indexes = random_list[i*data_per_client: (i+1)*data_per_client]
        datasets.append(list(itemgetter(*indexes)(trainset)))

    trainloader_list = []
    for d in datasets:
        trainloader_list.append(torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True))

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return trainloader_list, testloader


def cifar_five_class_per_user(batch_size, total_num_clients, num_selected_clients):



    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)


    total_data = len(trainset)
    data_per_client = int(total_data / total_num_clients)

    classes_per_client = 5
    data_class_per_client = data_per_client/classes_per_client
    num_classes = 10
    datasets_per_class = [[] for _ in range(num_classes)]
    values = np.array(trainset.targets)

    for i in range(num_classes):
        indexes = (np.where(values == i))
        datasets_per_class[i] = list(itemgetter(*(indexes[0]))(trainset))

    datasets = [[] for _ in range(total_num_clients)]
    index_per_class = [0 for _ in range(num_classes)]
    dict = {}
    for i in range(10):
        dict[i] = 0

    for client in total_num_clients:

        for i in range(10):
            if dict[i] == 5000:
                del dict[i]

        selected_classes = np.random.randint(0,10,5)

        for sc in selected_classes:
            start_index = index_per_class[sc]
            end_index = start_index+data_class_per_client
            index_per_class[sc] += data_class_per_client
            indeces = range(start_index, end_index)
            datasets[client].append(list(itemgetter(*indeces)(datasets_per_class[sc])))



    random.shuffle(datasets)
    trainloader_list = []
    for d in datasets:
        trainloader_list.append(
            torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True))

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4,
                                             pin_memory=True)

    return trainloader_list, testloader


def cifar_one_class_per_user(batch_size, total_num_clients, shuffle):

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    total_data = len(trainset)
    data_per_client = int(total_data / total_num_clients)
    num_classes = 10
    datasets_per_class = [[] for _ in range(num_classes)]
    values = np.array(trainset.targets)
    for i in range(num_classes):
        indexes = (np.where(values == i))
        datasets_per_class[i] = list(itemgetter(*(indexes[0]))(trainset))

    datasets = [[] for _ in range(total_num_clients)]
    client = 0
    for dc in datasets_per_class:
        l = len(dc)
        indexes = random.sample(range(l), l)

        for i in np.arange(0, l, data_per_client):
            ii = indexes[i:i + data_per_client]
            datasets[client] = list(itemgetter(*ii)(dc))
            client += 1
    if shuffle:
        random.shuffle(datasets)
    trainloader_list = []

    # code to check
    # lab = [[] for _ in range(total_num_clients)]
    # for cnt,d in enumerate(datasets):
    #     for el in d:
    #         lab[cnt].append(el[1])
    # set_lab = []
    # for cnt,l in enumerate(lab):
    #     set_lab.append(set(l))

    for d in datasets:
        trainloader_list.append(
            torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True))

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4,
                                             pin_memory=True)

    return trainloader_list, testloader