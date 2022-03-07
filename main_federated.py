import torch
import torch.nn as nn
import torchvision
from model import CNNMnist, cifar
from FL.FL_train import train_model_aggregated_random, train_model_aggregated_non_random, train_model
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
torch.manual_seed(1)

total_num_users = 100
num_users = 50
users_per_group = 10 #5

num_rounds = 150
local_epochs = 1
#num_users = 100
#users_per_group = 10
#total_num_users = 500
batch_size = 16
learning_rate = 0.001
mode = "hybrid_noniid_non_random"


print(f"NUM_USERS: {num_users}")
print(f"users_per_group: {users_per_group}")
print(f"num_rounds: {num_rounds}")
print(f"local_epochs: {local_epochs}")
print(f"batch_size: {batch_size}")
print(f"learning_rate: {learning_rate}")
print(f"mode: {mode}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = cifar(3,10)
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

if mode == "standard_iid":
    train_loss, train_acc, val_loss, val_acc = train_model(model_ft, criterion, num_rounds=num_rounds, local_epochs=local_epochs, total_num_users=total_num_users, num_users=num_users,
                                                       batch_size=batch_size, learning_rate=learning_rate, iid=True)
elif mode == "standard_noniid":
    train_loss, train_acc, val_loss, val_acc = train_model(model_ft, criterion, num_rounds=num_rounds,
                                                           local_epochs=local_epochs, total_num_users=total_num_users,
                                                           num_users=num_users,
                                                           batch_size=batch_size,
                                                           learning_rate=learning_rate, iid=False)
elif mode == "hybrid_iid":
    train_loss, train_acc, val_loss, val_acc = train_model_aggregated_random(model_ft, criterion, num_rounds=num_rounds,
                                                           local_epochs=local_epochs, total_num_users=total_num_users,
                                                           num_users=num_users,
                                                           users_per_group=users_per_group, batch_size=batch_size,
                                                           learning_rate=learning_rate, iid=True)
elif mode == "hybrid_noniid":
    train_loss, train_acc, val_loss, val_acc = train_model_aggregated_random(model_ft, criterion, num_rounds=num_rounds,
                                                           local_epochs=local_epochs, total_num_users=total_num_users,
                                                           num_users=num_users,
                                                           users_per_group=users_per_group, batch_size=batch_size,
                                                           learning_rate=learning_rate, iid=False)


elif mode == "hybrid_noniid_non_random":
    train_loss, train_acc, val_loss, val_acc = train_model_aggregated_non_random(model_ft, criterion, num_rounds=num_rounds,
                                                           local_epochs=local_epochs, total_num_users=total_num_users,
                                                           num_users=num_users,
                                                           users_per_group=users_per_group, batch_size=batch_size,
                                                           learning_rate=learning_rate)



