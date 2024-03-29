import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import scipy
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math
import torch.distributed as dist
import os
from torch.utils.tensorboard import SummaryWriter
from linear_evaluation import train_model, eval_model
from knn_evaluation import knn_features, knn_classifier
import utils
import vision_transformer as vits
from classes import DataAugmentationDINO, DINOLoss


################ SET DEVICE AND RANDOM SEEDS ######################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)
#print(torch.cuda.get_device_name(device))
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = True


##################### LOAD DATASET ###############################

def load_data(dataset_path, global_crops_scale, local_crops_scale, local_crops_number, batch_size, valid_split):
    transform = DataAugmentationDINO(global_crops_scale, local_crops_scale, local_crops_number)
    val_transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224))])

    #load stanford cars dataset
    train_set = torchvision.datasets.StanfordCars(root = dataset_path, split="train", download=True, transform=transform)
    val_set = torchvision.datasets.StanfordCars(root = dataset_path, split="train", download=True, transform=val_transform)

    #train-val split
    num_train = len(train_set)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_split * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    #train-val samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    #data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, sampler=train_sampler, num_workers=2)
    knn_train_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, sampler=train_sampler, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, sampler=valid_sampler, num_workers=2)

    return train_loader, val_loader, knn_train_loader


################### PRETRAINED MODELS ##########################

def load_models(arch, patch_size, out_dim):
    url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth"
    state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)

    student = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    teacher = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    embed_dim = student.embed_dim
    student = utils.MultiCropWrapper(student, vits.DINOHead(embed_dim,out_dim))
    teacher = utils.MultiCropWrapper(teacher, vits.DINOHead(embed_dim,out_dim))

    state_dict['student'] = {k.replace("module.", ""): v for k, v in state_dict['student'].items()}
    student.load_state_dict(state_dict['student'], strict=True)
    state_dict['teacher'] = {k.replace("module.", ""): v for k, v in state_dict['teacher'].items()}
    teacher.load_state_dict(state_dict['teacher'], strict=True)

    student, teacher = student.cuda(), teacher.cuda()

    for p in teacher.parameters():
            p.requires_grad = False

    return teacher,student


######################## ONE EPOCH #############################

def training_step(student, teacher, dino_loss, train_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch, clip_grad, freeze_last_layer, lr):
    
    running_loss = 0.0

    comment = "base_lr="+str(lr)
    writer = SummaryWriter(comment=comment)

    for it, (images, _) in enumerate(train_loader,0):
        
        lr = lr*1.039
        # update weight decay and learning rate according to their schedule
        it = len(train_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            #param_group["lr"] = lr_schedule[it]
            param_group["lr"] = lr
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]

        teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
        student_output = student(images)
        loss = dino_loss(student_output, teacher_output, epoch)

        #student optimizer step
        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            param_norms = utils.clip_gradients(student, clip_grad)
        utils.cancel_gradients_last_layer(epoch, student,
                                          freeze_last_layer)
        optimizer.step()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        running_loss += loss

        writer.add_scalar("Loss_during_epoch", loss, it+1)
        writer.add_scalar("Lr_during_epoch", lr, it+1)

    epoch_loss = running_loss / len(train_loader)

    writer.flush()
    writer.close()

    return epoch_loss 


#################### MAIN LOOP ##############################

def train_dino(arch, patch_size, out_dim, global_crops_scale, local_crops_scale, 
                local_crops_number, batch_size, warmup_teacher_temp, teacher_temp, 
                warmup_teacher_temp_epochs, epochs, momentum_teacher, lr, min_lr, clip_grad, 
                weight_decay, weight_decay_end, warmup_epochs, freeze_last_layer, dataset_path, valid_split, save_path):
    
    train_loader, val_loader, knn_train_loader = load_data(dataset_path, global_crops_scale, local_crops_scale, local_crops_number, batch_size, valid_split)

    teacher, student = load_models(arch, patch_size, out_dim)
    
    # loss
    dino_loss = DINOLoss(
        out_dim,
        local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        epochs,
    ).cuda()

    # optimizer (adamw for vit)
    params_groups = utils.get_params_groups(student)
    optimizer = torch.optim.AdamW(params_groups)

    #schedule for learning rate, weight decay and teacher momentum
    lr_schedule = utils.cosine_scheduler(
        lr * (batch_size * utils.get_world_size()) / 256.,  # linear scaling rule
        min_lr,
        epochs, len(train_loader),
        warmup_epochs=warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        weight_decay,
        weight_decay_end,
        epochs, len(train_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(momentum_teacher, 1,
                                                   epochs, len(train_loader))

    # tensorboard writer
    comment = "fixed_lr_nepochs=" + str(epochs)+"_lr="+str(lr)
    #writer = SummaryWriter(comment=comment)
    
    for epoch in range(epochs):

        # single epoch
        epoch_loss = training_step(student, teacher, dino_loss, train_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch, clip_grad, freeze_last_layer,lr)

        # knn validation
        train_features, train_labels = knn_features(teacher.backbone,knn_train_loader)
        val_features, val_labels = knn_features(teacher.backbone,val_loader)

        top1, top5 = knn_classifier(train_features, train_labels, val_features, val_labels)

        # tensorboard logs
        #writer.add_scalar("Loss/train", epoch_loss, epoch+1)
        #writer.add_scalar("Top1_Accuracy/validation", top1, epoch+1)
        #writer.add_scalar("Top5_Accuracy/validation", top5, epoch+1)
        
        # save model
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'dino_loss': dino_loss.state_dict(),
        }

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(save_dict, os.path.join(save_path, 'checkpoint.pth'))


    #close tensorbloard 
    #writer.flush()
    #writer.close()


def main():
    ######## HYPERPARAMETERS ########

    # model
    arch = 'vit_small'
    patch_size = 16
    out_dim = 65536

    #data
    global_crops_scale = (0.4, 1.)
    local_crops_scale = (0.05, 0.4)
    local_crops_number = 8
    batch_size = 128
    valid_split = 0.1

    #training
    warmup_teacher_temp = 0.04
    teacher_temp = 0.04
    warmup_teacher_temp_epochs = 0
    epochs = 1
    momentum_teacher = 0.996
    #lr = 5e-6
    min_lr = 1e-6
    clip_grad = 3.0
    weight_decay = 0.04
    weight_decay_end = 0.4
    warmup_epochs = 1
    freeze_last_layer = 1

    #misc
    dataset_path = '/data/projects/cpca2227818a0/ricardo.teixeira/dataset'
    save_path = '/data/projects/cpca2227818a0/ricardo.teixeira/saved_models'

    ######## RUN DINO ########
    
    for lr in [1e-5, 1e-6, 1e-7, 1e-8]:
        train_dino(arch, patch_size, out_dim, global_crops_scale, local_crops_scale, 
                local_crops_number, batch_size, warmup_teacher_temp, teacher_temp, 
                warmup_teacher_temp_epochs, epochs, momentum_teacher, lr, min_lr, clip_grad, 
                weight_decay, weight_decay_end, warmup_epochs, freeze_last_layer, dataset_path, valid_split, save_path)


main()
