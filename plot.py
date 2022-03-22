import matplotlib.pyplot as plt
import numpy as np

def read_file(path):
    f = open(path, "r")
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for line in f:
        a = line.split(" ")
        if a[0] == "train":
            train_loss.append(float(a[2]))
            train_acc.append(float(a[4]))
        elif a[0] == "val":
            val_loss.append(float(a[2]))
            val_acc.append(float(a[4]))

    return train_loss, train_acc, val_loss, val_acc

_, _, _, val_acc0 = read_file("results/standard_noniid_le1_lr0,005.txt")
_, _, _, val_acc3 = read_file("results/hybrid_noniid_random_le1_lr0,005_decay_each_client0999.txt")
_, _, _, val_acc7 = read_file("results/hybrid_noniid_non_random_le1_lr0,005_decay_each_client0.999.txt")

N_EPOCHS = 450

val_acc0 = val_acc0[:N_EPOCHS]
val_acc3 = val_acc3[:N_EPOCHS]
val_acc7 = val_acc7[:N_EPOCHS]

plt.figure()
plt.plot(val_acc0, label='FL - 50 users')
plt.plot(val_acc3, "tab:orange",label='EAGLE - 50 users in groups of 10')
plt.plot(val_acc7, "tab:red",label='EAGLE - 50 users in non-randomly created groups of 10')

plt.legend()
plt.grid()
plt.yticks(np.arange(0, 0.9, 0.1))
plt.xlabel("Number of rounds")
plt.ylabel("Accuracy")
#plt.title("Accuracy w.r.t. to number of rounds (l.r. 0.001, 5 local epochs)")
# plt.savefig("non iid comparison wrt number of rounds lr 0,005")
plt.savefig("noniid_round.png")
plt.show()

#############################################

plt.figure()
plt.plot(val_acc0, label='FL - 50 users')
plt.plot(np.arange(0,N_EPOCHS*10,10), val_acc3, "tab:orange",label='EAGLE - 50 users in groups of 10')
plt.plot(np.arange(0,N_EPOCHS*10,10), val_acc7, "tab:red",label='EAGLE - 50 users in non-randomly created groups of 10')


plt.legend()
plt.grid()
#plt.xticks(range(0, 101, 5))
plt.yticks(np.arange(0, 0.9, 0.1))
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy")
# plt.title("Accuracy w.r.t. to number of time slots (l.r. 0.001, 5 local epochs)")
# plt.savefig("non iid comparison wrt number of rounds lr 0,005")
plt.show()

######################

plt.figure()
plt.plot(np.arange(200,200*51,200),val_acc0, label='FL - 50 users')
plt.plot(np.arange(110,110*51,110), val_acc3, "tab:orange",label='EAGLE - 50 users in groups of 10')
plt.plot(np.arange(110,110*51,110), val_acc7, "tab:red", label='EAGLE - 50 users in non-randomly created groups of 10')

plt.legend()
plt.grid()

plt.yticks(np.arange(0, 0.9, 0.1))
plt.xlabel("Number of transmissions")
plt.ylabel("Accuracy")
plt.title("Accuracy w.r.t. to number of transmissions (l.r. 0.001, 5 local epochs)")
# plt.savefig("non iid comparison wrt number of rounds lr 0,005")
plt.show()

