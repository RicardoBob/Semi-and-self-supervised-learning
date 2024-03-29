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


def knn_features(model, data_loader):
    with torch.no_grad():
        features, targets = None, None
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            outputs = model(inputs).clone()
            if features is None and targets is None:
                features = outputs
                targets = labels
            else:
                features = torch.cat((features,outputs),0)
                targets = torch.cat((targets,labels),0)
        features = nn.functional.normalize(features, dim=1, p=2)
    
    return features,targets


def knn_classifier(train_features, train_labels, test_features, test_labels, k=20, T=0.07, num_classes=196):
    with torch.no_grad():
        top1, top5, total = 0.0, 0.0, 0
        train_features = train_features.t()
        num_test_images, num_chunks = test_labels.shape[0], 10
        imgs_per_chunk = num_test_images // num_chunks
        retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
        
        i = 0
        
        for idx in range(0, num_test_images, imgs_per_chunk):
            # get the features for test images
            features = test_features[
                idx : min((idx + imgs_per_chunk), num_test_images), :
            ]
            targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
            batch_size = targets.shape[0]
  
            # calculate the dot product and compute top-k neighbors
            similarity = torch.mm(features, train_features)
            if i == 0:
                print(similarity[:5][:5])
                print(similarity.shape, str(batch_size) + "x" + str(train_labels.shape[0]))
            distances, indices = similarity.topk(k, largest=True, sorted=True)
            if i == 0:    
                print(distances[:5][:5])
                print(distances.shape, str(batch_size) + "x" + str(k))
            candidates = train_labels.view(1, -1).expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices)


            retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
            distances_transform = distances.clone().div_(T).exp_()
            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batch_size, -1, num_classes),
                    distances_transform.view(batch_size, -1, 1),
                ),
                1,
            )
            _, predictions = probs.sort(1, True)
  
            # find the predictions that match the target
            correct = predictions.eq(targets.data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
            total += targets.size(0)
            
            i+=1

        top1 = top1 * 100.0 / total
        top5 = top5 * 100.0 / total
    return top1, top5