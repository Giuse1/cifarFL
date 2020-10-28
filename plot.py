import matplotlib.pyplot as plt
import numpy as np
import torch

# dict = torch.load("../cifar.pth", map_location=torch.device('cpu'))
# n = 0
# for d in dict.items():
#
#     to_add = 1
#     for i in range(len(d[1].shape)):
#         to_add *= d[1].shape[i]
#     n += to_add
# a = np.array([(b'aaa', 1, 4.2),
#                (b'bbb', 2, 8.4),
#                (b'ccc', 3, 12.6)],
#               dtype=[('foo', 'S3'), ('bar', 'i4'), ('baz', 'f8')])
# print(type(a[0]))
# print(n)


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




train_loss0, train_acc0, val_loss0, val_acc0 = read_file("results/standard_nonIID_lr0,001.txt")
_, _, _, val_acc4 = read_file("results/standard_nonIID_lr0,001_50u.txt")
_, _, _, val_acc5 = read_file("results/standard_nonIID_lr0,001_20u.txt")
_, _, _, val_acc6 = read_file("results/standard_nonIID_lr0,001_10u.txt")
_, _, _, val_acc1 = read_file("results/hybrid_nonIID_group2_le0,001.txt")
_, _, _, val_acc2 = read_file("results/hybrid_nonIID_group5_lr0,001.txt")
_, _, _, val_acc3 = read_file("results/hybrid_nonIID_group10_lr0,001.txt")


plt.figure()
plt.plot(val_acc0, label='SFL - 1 local epoch, 100 users')
plt.plot(val_acc4, label='SFL - 1 local epoch, 50 users')
plt.plot(val_acc5, label='SFL - 1 local epoch, 20 users')
plt.plot(val_acc6, label='SFL - 1 local epoch, 10 users')

plt.plot(val_acc1, label='HFL - 1 local epoch, 100 users in groups of 2')
plt.plot(val_acc2, label='HFL - 1 local epoch, 100 users in groups of 5')
plt.plot(val_acc3, label='HFL - 1 local epoch, 100 users in groups of 10')


plt.legend()
plt.grid()
#plt.xticks(range(0, 101, 5))

plt.yticks(np.arange(0, 1, 0.1))
plt.xlabel("Number of rounds")
plt.ylabel("Accuracy")
plt.title("Accuracy w.r.t. to number of rounds (lr 0.001)")
# plt.savefig("non iid comparison wrt number of rounds lr 0,001")
plt.show()


plt.figure()
plt.plot(np.arange(200,200*151,200),val_acc0, label='SFL - 1 local epoch, 100 users')
plt.plot(np.arange(100,100*301,100),val_acc4, label='SFL - 1 local epoch, 50 users')
plt.plot(np.arange(40,40*751,40),val_acc5, label='SFL - 1 local epoch, 20 users')
plt.plot(np.arange(20,20*1501,20)[:len(val_acc6)],val_acc6, label='SFL - 1 local epoch, 10 users')

plt.plot(np.arange(150,150*151,150), val_acc1, label='HFL - 1 local epoch, 100 users in groups of 2')
plt.plot(np.arange(120,120*151,120), val_acc2, label='HFL - 1 local epoch, 100 users in groups of 5')
plt.plot(np.arange(110,110*151,110), val_acc3, label='HFL - 1 local epoch, 100 users in groups of 10')

plt.legend()
plt.grid()
#plt.xticks(range(0, 101, 5))

plt.yticks(np.arange(0, 1, 0.1))
plt.xlabel("Number of transmissions")
plt.ylabel("Accuracy")
plt.title("Accuracy w.r.t. to number of transmissions (lr 0.001)")
# plt.savefig("non iid comparison wrt number of rounds lr 0,001")
plt.show()


plt.figure()
plt.plot(val_acc0, label='SFL - 1 local epoch, 100 users')
plt.plot(val_acc4, label='SFL - 1 local epoch, 50 users')
plt.plot(val_acc5, label='SFL - 1 local epoch, 20 users')
plt.plot(np.arange(0,len(val_acc6)), val_acc6, label='SFL - 1 local epoch, 10 users')

plt.plot(np.arange(0,150*2,2), val_acc1, label='HFL - 1 local epoch, 100 users in groups of 2')
plt.plot(np.arange(0,150*5,5), val_acc2, label='HFL - 1 local epoch, 100 users in groups of 5')
plt.plot(np.arange(0,150*10,10), val_acc3, label='HFL - 1 local epoch, 100 users in groups of 10')


plt.legend()
plt.grid()
#plt.xticks(range(0, 101, 5))

plt.yticks(np.arange(0, 1, 0.1))
plt.xlabel("Number of time slots")
plt.ylabel("Accuracy")
plt.title("Accuracy w.r.t. to number of time slots (lr 0.001)")
# plt.savefig("non iid comparison wrt number of rounds lr 0,001")
plt.show()