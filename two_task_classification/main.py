import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as functional
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms



#
ROOT_DIR = r'C:\Users\liu68\Documents\computer_vision_tf2\dataset\rabbit_cat_chicken'
TRAIN_DIR = r'\train'
VAL_DIR = r'\val'
TRAIN_ANNO = r'\Multi_train_annotation.csv'
VAL_ANNO = r'\Multi_val_annotation.csv'
SPECIES = ['rabbit', 'rat', 'chicken']
ClASSES = ['mammal', 'bird']


# step one : load data set
# 1.1 define custom torch.utils.data.Dataset
class MyDataset(Dataset):
    """This is Dataset load class for preparing loading data and preprocessing data"""

    def __init__(self, root_dir, annotation_file, transform=None):

        self.root_dir = root_dir
        self.annotation_file = annotation_file
        self.transform = transform

        if not os.path.isfile(self.annotation_file):
            print(self.annotation_file + 'does not exist!')
        self.file_info = pd.read_csv(self.annotation_file, index_col=0)
        self.length = len(self.file_info)

    def __len__(self):
        """It's necessary to define __len__, because it's the requirement of Dataset's subclass"""

        return self.length

    def __getitem__(self, idx):
        """define this special method For torch.utils.data.DataLoader generate random data"""
        image_path = self.file_info['path'][idx]
        if not os.path.isfile(image_path):
            print(image_path + 'does not exist')
            return

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_class = self.file_info.iloc[idx]['classes']
        label_species = self.file_info.iloc[idx]['species']

        sample = {'image': image, 'class': label_class, 'species': label_species}
        return sample


# 1.1 choose data augmentation
train_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.RandomPerspective(),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomAffine(0.5),
                                       transforms.ToTensor()])

val_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor()])

# 1.3 define DataLoader for train_set and test_set
train_dataset = MyDataset(root_dir=ROOT_DIR+TRAIN_DIR,
                          annotation_file=ROOT_DIR+TRAIN_ANNO,
                          transform=train_transforms)

val_dataset = MyDataset(root_dir=ROOT_DIR+VAL_DIR,
                        annotation_file=ROOT_DIR+VAL_ANNO,
                        transform=val_transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset))
data_loaders = {'train': train_loader, 'val': val_loader}

# 1.4 randomly examine data and it's label
def visualize_dataset(loader):
    """This function randomly examines whether image and it's label correct or not"""
    idx = np.random.randint(0, len(loader.dataset))
    sample = train_loader.dataset[idx]
    print(idx, sample['image'].shape, ClASSES[sample['class']], SPECIES[sample['species']])
    img = sample['image']
    plt.imshow(img.reshape(224, 224, 3))
    plt.show()
# visualize_dataset(val_loader)


# Step Two: define convolution neural network structure
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

        self.full_connect1 = nn.Linear(16 * 55 * 55, 64)
        self.full_connect2 = nn.Linear(64, 3)
        self.full_connect3 = nn.Linear(64, 2)

    def forward(self, x):

        x = self.relu(self.maxpool(self.conv1(x)))

        x = self.relu(self.maxpool(self.conv2(x)))

        # print('x shape: {x.shape}')
        x = x.view(-1, 16 * 55 * 55)
        x = self.full_connect1(x)
        x = self.relu(x)
        x = functional.dropout(x, p=0.1, training=self.training)

        # Separate training from here
        x_species = self.full_connect2(x)
        x_species = self.softmax(x_species)
        x_class = self.full_connect3(x)
        x_class = self.softmax(x_class)

        return x_species, x_class


network = Net()
network.load_state_dict(torch.load('./best_model.pth'))

# Step Three: define criterion, criterion, learning rate scheduler
optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


# Step Four: Train
def train_model(model, criterion, optimizer, lr_scheduler, epochs=3):

    # record information of loss and accuracy while training
    loss_list = {'train': [], 'val': []}
    accuracy_list_species = {'train': [], 'val': []}
    accuracy_list_class = {'train': [], 'val': []}

    # record the test model parameters
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = {'overall': 0.0, 'species': 0.0, 'class': 0.0}

    for epoch in range(epochs):
        print(f'{epoch+1}/{epochs} \n -------------------')

        for phase in ['train', 'val']:
            # for accelerating, because computation graph is different in training and evaluating process
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, correct_species, correct_class = 0, 0, 0

            for idx, data in enumerate(data_loaders[phase]):
                print(phase+f' processing: {idx+1}th batch')
                inputs, labels_species, labels_class = data['image'], data['species'], data['class']
                optimizer.zero_grad()

                # when training, requires_grad == True. when evaluating requires_grad == False
                with torch.set_grad_enabled(phase == 'train'):
                    x_species, x_class = model(inputs)

                    # max_val represents corresponding label
                    _, preds_species = torch.max(x_species, 1)
                    _, preds_class = torch.max(x_class, 1)

                    # total loss just add each other
                    loss = criterion(x_species, labels_species) + criterion(x_class, labels_class)

                    if phase == 'train':
                        loss.backward()  # compute gradient
                        optimizer.step()  # gradient update
                        lr_scheduler.step()  # lr update

                # record necessary loss value and accuracy
                running_loss += loss.item() * inputs.size(0)
                correct_species += torch.sum(preds_species == labels_species)
                correct_class += torch.sum(preds_class == labels_class)

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc_species = correct_species.double() / len(data_loaders[phase].dataset)
            epoch_acc_class = correct_class.double() / len(data_loaders[phase].dataset)
            epoch_acc = 0.6 * epoch_acc_species + 0.4 * epoch_acc_class

            loss_list[phase].append(epoch_loss)
            accuracy_list_class[phase].append(100*epoch_acc_class)
            accuracy_list_species[phase].append(100 * epoch_acc_species)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc_species: {epoch_acc_species:.2%} Acc_class: {epoch_acc_class:.2%}')

            # if current val is better than before, updating best_model_wts
            if phase == 'val' and epoch_acc > best_acc['overall']:
                best_acc['overall'] = epoch_acc
                best_acc['species'] = epoch_acc_species
                best_acc['class'] = epoch_acc_class
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Best val acc: {:.2%}'.format(best_acc['overall']))

    # save best model.parameters()
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'best_model.pth')
    print('Best Val Acc: {:.2%}'.format(best_acc['overall']))

    return model, loss_list, accuracy_list_species, accuracy_list_class

model, loss_list, accuracy_list_species, accuracy_list_class = \
    train_model(network, criterion, optimizer, exp_lr_scheduler, epochs=30)


# Step Five: evaluate
def plot_loss_accuracy():

    fig, ax = plt.subplots()
    ax.plot(range(len(loss_list['train'])), loss_list['train'], label='train')
    ax.plot(range(len(loss_list['val'])), loss_list['val'], label='val')
    ax.legend()
    ax.set_title('Loss')
    fig.savefig('loss')

    fig, ax = plt.subplots()
    ax.plot(range(len(accuracy_list_class['train'])), accuracy_list_class['train'], label='train')
    ax.plot(range(len(accuracy_list_class['val'])), accuracy_list_class['val'], label='val')
    ax.legend()
    ax.set_title('accuracy of class')
    fig.savefig('accuracy of class')

    fig, ax = plt.subplots()
    ax.plot(range(len(accuracy_list_species['train'])), accuracy_list_species['train'], label='train')
    ax.plot(range(len(accuracy_list_species['val'])), accuracy_list_species['val'], label='val')
    ax.legend()
    ax.set_title('Accuracy Of Species')
    fig.savefig('accuracy of species')

plot_loss_accuracy()



# loss_list.to_csv('loss.csv', encoding='utf-8')
# accuracy_list_class.to_csv('species_acc.csv', encoding='utf-8')
# accuracy_list_class.to_csv('class_acc.csv', encoding='utf-8')



















