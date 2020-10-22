from torch import nn
from torchvision import models

def shufflenet(num_classes):
    model = models.shufflenet_v2_x1_0()
    output_channels = model.fc.in_features
    model.fc = nn.Linear(output_channels, num_classes)

    return model