import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import scipy
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def train_model(model, criterion, optimizer, train_loader, device, n_epochs=5):

    losses = []

    # freeze the backbone and set the predition head to trainable
    ct = 0
    for child in model.children():
      ct += 1
      if ct < 5:
        for param in child.parameters():
            param.requires_grad = False

    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            # get the inputs and assign them to cuda
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            #reset optimizer for this batch
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # calculate the loss/acc later
            running_loss += loss.item()

        epoch_loss = running_loss/len(train_loader)
        print("Epoch %s, loss: %.4f" % (epoch+1, epoch_loss))
        
        losses.append(epoch_loss)
    print('Finished Training')
    return model, losses


def eval_model(model, test_loader, device):
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100.0 * correct / total
    print('Accuracy of the network on the test images: %d %%' % (
        test_acc))
    return test_acc
