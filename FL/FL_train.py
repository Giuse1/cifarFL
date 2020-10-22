from FL.FL_user import LocalUpdate
import copy
import torch
from FL.torch_dataset import get_cifar_iid
import random
random.seed(0)


def train_model(global_model, criterion, num_rounds, local_epochs, num_users, batch_size, learning_rate):
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    total_num_users = 500

    trainloader_list, valloader = get_cifar(batch_size=batch_size)

    # random_list = range(num_users)

    for round in range(num_rounds):
        print('-' * 10)
        print('Epoch {}/{}'.format(round, num_rounds - 1))

        for phase in ['train', 'val']:
            if phase == 'train':
                local_weights = []
                samples_per_client = []

                random_list = random.sample(range(total_num_users), num_users)

                for idx in random_list:
                    local_model = LocalUpdate(dataloader=trainloader_list[idx], id=idx, criterion=criterion,
                                              local_epochs=local_epochs, learning_rate=learning_rate)
                    w, local_loss, local_correct, local_total = local_model.update_weights(
                        model=copy.deepcopy(global_model).double())
                    local_weights.append(copy.deepcopy(w))
                    samples_per_client.append(local_total)

                global_weights = average_weights(local_weights, samples_per_client)
                global_model.load_state_dict(global_weights)

            else:
                val_loss_r, val_accuracy_r = model_evaluation(model=global_model.double(),
                                                              dataloader=valloader, criterion=criterion)

                val_loss.append(val_loss_r)
                val_acc.append(val_accuracy_r)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, val_loss_r, val_accuracy_r))

    return train_loss, train_acc, val_loss, val_acc


def model_evaluation(model, dataloader, criterion):
    with torch.no_grad():
        model.eval()  # Set model to evaluate mode
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        running_loss = 0.0
        running_corrects = 0
        running_total = 0

        for (i, data) in enumerate(dataloader):
            inputs, labels = data['input'].to(device), data['label'].to(device)

            outputs = model(inputs.double())
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)
            running_total += labels.shape[0]

        epoch_loss = running_loss / running_total
        epoch_acc = running_corrects.double() / running_total

        return epoch_loss, epoch_acc


def train_model_aggregated(global_model, criterion, num_rounds, local_epochs, num_users, users_per_group, batch_size,
                           learning_rate):
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    total_num_users = 500


    trainloader_list = getDataloaderList(path='data/', transform=transform, batch_size=batch_size, shuffle=True)
    valloader_list = getDataloaderList(path='data_test/', transform=transform, batch_size=batch_size, shuffle=True)

    num_groups = num_users / users_per_group
    for round in range(num_rounds):
        print('-' * 10)
        print('Epoch {}/{}'.format(round, num_rounds - 1))

        for phase in ['train', 'val']:
            if phase == 'train':
                local_weights = []
                samples_per_client = []

                random_list = random.sample(range(total_num_users), num_users)

                for i in range(int(num_groups)):
                    for j in range(users_per_group):
                        idx = random_list[j + i * users_per_group]
                        local_model = LocalUpdate(dataloader=trainloader_list[idx], id=idx, criterion=criterion,
                                                  local_epochs=local_epochs, learning_rate=learning_rate)

                        if j == 0:
                            w, local_loss, local_correct, local_total = local_model.update_weights(
                                model=copy.deepcopy(global_model).double())
                            samples_per_client.append(local_total)
                        else:
                            model_tmp = copy.deepcopy(global_model)
                            model_tmp.load_state_dict(w)
                            w, local_loss, local_correct, local_total = local_model.update_weights(
                                model=model_tmp.double())
                            samples_per_client[i] += local_total

                    local_weights.append(copy.deepcopy(w))

                global_weights = average_weights(local_weights, samples_per_client)
                global_model.load_state_dict(global_weights)

            else:
                val_loss_r, val_accuracy_r = model_evaluation(model=global_model.double(),
                                                              dataloader_list=valloader_list, criterion=criterion)

                val_loss.append(val_loss_r)
                val_acc.append(val_accuracy_r)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, val_loss_r, val_accuracy_r))

    return train_loss, train_acc, val_loss, val_acc


def average_weights(w, samples_per_client):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(0, len(w)):
            if i == 0:
                w_avg[key] = torch.true_divide(w[i][key], 1 / samples_per_client[i])
            else:
                w_avg[key] += torch.true_divide(w[i][key], 1 / samples_per_client[i])
        w_avg[key] = torch.true_divide(w_avg[key], sum(samples_per_client))
    return w_avg
