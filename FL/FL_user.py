import torch
import matplotlib.pyplot as plt
import numpy as np


def imshow(img, i):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(str(i) + ".png")


class LocalUpdate(object):
    def __init__(self, dataloader, id, criterion, local_epochs, learning_rate):
        self.id = id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = criterion
        self.local_epochs = local_epochs
        self.dataloader = dataloader
        self.learning_rate = learning_rate

    def update_weights(self, model):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        for _ in range(self.local_epochs):

            local_correct = 0
            local_loss = 0.0
            for (i, data) in enumerate(self.dataloader):
                images, labels = data[0].to(self.device), data[1].to(self.device)
                # for c,img in enumerate(images):
                #  imshow(img,c)
                #  print(classes[labels[c]])
                # print(ciao)
                optimizer.zero_grad()
                log_probs = model(images.double())
                loss = self.criterion(log_probs, labels)
                _, preds = torch.max(log_probs, 1)
                local_correct += torch.sum(preds == labels).cpu().numpy()

                loss.backward()
                optimizer.step()
                local_loss += loss.item() * images.size(0)

        return model.state_dict(), local_loss, local_correct, len(self.dataloader.dataset)
