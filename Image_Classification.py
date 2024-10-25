'''
summary:
This script trains and evaluates a ResNet18 model for image classification using transfer learning. 
The model is trained on TPU with PyTorch XLA support, and key functionalities include data augmentation, dynamic train-validation splitting, performance tracking, 
and visualization of training metrics. A confusion matrix is plotted at the end to evaluate the model’s performance on the test set.
'''
--------------------------------------------------------------------------------------------
#%cd "/content/drive/MyDrive/SoundClassification"
%cd "/content/drive/MyDrive/ImageClassification_Task2"
--------------------------------------------------------------------------------------------


# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch_xla.core.xla_model as xm  # XLA support for TPUs
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import time
import os

# 1. Define data augmentation and normalization for training, validation, and test sets
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # Randomly resize and crop for augmentation
        transforms.RandomHorizontalFlip(),  # Horizontal flip for data augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet mean and std
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),  # Resize for validation
        transforms.CenterCrop(224),  # Center crop for validation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet mean and std
    ]),
}

# 2. Load the datasets using ImageFolder (assumes data is in subdirectories by class)
train_path = 'seg_train/seg_train'  # Path to training dataset
test_path = 'seg_test/seg_test'  # Path to test dataset

full_trainset = torchvision.datasets.ImageFolder(root=train_path, transform=data_transforms['train'])
testset = torchvision.datasets.ImageFolder(root=test_path, transform=data_transforms['val'])

# 3. Create train/validation split dynamically (80/20 split)
train_size = int(0.8 * len(full_trainset))  # 80% of dataset for training
val_size = len(full_trainset) - train_size  # 20% for validation
trainset, valset = random_split(full_trainset, [train_size, val_size])

# Create data loaders
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
valloader = DataLoader(valset, batch_size=32, shuffle=False)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

dataloaders = {'train': trainloader, 'val': valloader}
dataset_sizes = {'train': len(trainset), 'val': len(valset)}

# 4. Define the neural network architecture (ResNet18 with transfer learning)
import torchvision.models as models

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')  # Use ResNet18 with pretrained weights
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, len(full_trainset.classes))  # Adjust output layer for number of classes

    def forward(self, x):
        return self.resnet(x)

net = Net()

# 5. Move model to TPU (or CPU/GPU)
device = xm.xla_device() if torch.cuda.is_available() else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)

# 6. Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 7. Add a learning rate scheduler (decays LR by a factor of 0.1 every 7 epochs)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 8. Define the training function with validation accuracy tracking and saving best model
def train_model_tpu(net, dataloaders, optimizer, scheduler, criterion, num_epochs=5):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_model_wts = None
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # Set model to training mode
            else:
                net.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimization only in training phase
                    if phase == 'train':
                        loss.backward()
                        xm.optimizer_step(optimizer, barrier=True)

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()  # Adjust learning rate

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save the model with the best validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = net.state_dict()

    print('Training complete')
    print(f'Best Validation Accuracy: {best_acc:.4f}')

    # Load best model weights
    if best_model_wts:
        net.load_state_dict(best_model_wts)

    return train_losses, val_losses, train_accuracies, val_accuracies

# 9. Train the model
train_losses, val_losses, train_accuracies, val_accuracies = train_model_tpu(net, dataloaders, optimizer, scheduler, criterion, num_epochs=50)

# 10. Plot the training and validation accuracies
plt.figure(figsize=(10,5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy over Epochs')
plt.ylim(0.5,1)
plt.legend()
plt.show()

# 11. Plot the training and validation losses
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation Loss over Epochs')
plt.ylim(0,1)
plt.legend()
plt.show()

# 12. Evaluate the model on the test set
def calculate_accuracy(loader, model):
    correct = 0
    total = 0
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

test_acc = calculate_accuracy(testloader, net)
print(f'Test Accuracy: {test_acc:.2f}%')

# 13. Confusion Matrix function
def plot_confusion_matrix(net, testloader):
    all_preds = []
    all_labels = []

    net.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=full_trainset.classes, yticklabels=full_trainset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# 14. Plot confusion matrix on test set
plot_confusion_matrix(net, testloader)
