import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN(nn.Module):
    def __init__(self, channels, classes, nodes_linear, mutually_exc=True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 3, 7, stride=2)
        self.conv2 = nn.Conv2d(3, 6, 7, stride=2)
        self.fc1 = nn.Linear(nodes_linear, classes)

        self.classes = classes
        if mutually_exc:
            self.activation = nn.Softmax(dim=-1)
        else:
            self.activation = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = torch.flatten(x, 1) # flatten all dimensions except batch

        x = F.relu(self.fc1(x))

        return self.activation(x)

class Linear_classifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.lin1 = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = F.softmax(self.lin1(x), dim=-1)
        return x
