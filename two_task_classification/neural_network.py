import torch.nn as nn
import torch.nn.functional as functional


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d()
        self.softmax = nn.Softmax(dim=1)

        self.full_connect1 = nn.Linear(111, 128)
        self.full_connect2 = nn.Linear(64, 3)
        self.full_connect3 = nn.Linear(64, 2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.relu(x)

        # print(x.shape)
        x = x.view(-1, 100)
        x = self.full_connect1(x)
        x = self.relu(x)
        x = functional.dropout(x, p=0.1, training=self.training)

        # Separate training from here
        x_species = self.full_connect2(x)
        x_species = self.softmax(x_species)
        x_class = self.full_connect3(x)
        x_class = self.softmax(x_class)

        return x_species, x_class










