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


def knn_features(model, data_loader, device):
    features, targets = None, None
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            if features is None and targets is None:
                features = outputs
                targets = labels
            else:
                features = torch.cat((features,outputs),0)
                targets = torch.cat((targets,labels),0)

    return features,targets


def knn_classifier(train_features, train_labels, test_features, test_labels, k=20, num_classes=196, t = 0.07):
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    train_features = train_features.t()
    similarities = torch.mm(test_features, train_features)
    distances, indices = similarities.topk(k,largest=True, sorted=True)
    candidates = train_labels.view(1, -1).expand(test_labels.shape[0], -1)
    retrieved_neighbors = torch.gather(candidates, 1, indices)

    retrieval_one_hot.resize_(test_labels.shape[0] * k, num_classes).zero_()
    retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
    distances_transform = distances.clone().div_(t).exp_()
    probs = torch.sum(
        torch.mul(
                retrieval_one_hot.view(test_labels.shape[0], -1, num_classes),
                distances_transform.view(test_labels.shape[0], -1, 1),
            ),
            1,
        )
    _, predictions = probs.sort(1, True)

    correct = predictions.eq(test_labels.view(-1,1))
    accuracy = correct.narrow(1, 0, 1).sum().item() * 100 / test_labels.shape[0]
    #print("Accuracy: " + str(accuracy))
    return accuracy