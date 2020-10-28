import torch
import torchvision
import torchvision.transforms as transforms
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

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


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


# def get_cifar_noniid(batch_size):
#     indexes_animals
#     indexes_means
#
#

