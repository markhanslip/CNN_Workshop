#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torchvision
from torchvision import transforms, datasets, models
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import argparse
from IPython.display import Image, display, HTML
import os
#from torch.utils.tensorboard import SummaryWriter
from functools import partial

optimizer=None
loss_fn=None
model=None
mean = None
std = None
batch_size=0
train_data_loader=None
valid_data_loader=None
lr=0
weight_decay=0
model_name=''
train_loss_per_epoch=[]
valid_loss_per_epoch=[]

def load_data(data_path, batch_size):
    global classes
    global train_data_loader
    global valid_data_loader
    global mean
    global std
    global num_workers

    input_res = 224
    batch_size = opt.batch_size # 16 seems to work well

    transform = transforms.Compose([
        transforms.Resize(input_res),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    full_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
    full_dataLoader = torch.utils.data.DataLoader(full_dataset, shuffle=False, num_workers=num_workers)

    # if train and valid not pre-separated:
    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size

    train_data, valid_data = torch.utils.data.random_split(full_dataset, [train_size, valid_size])

    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    valid_data_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classes=full_dataLoader.dataset.classes
    print('classes are '+str(classes))

def load_model(model_name, device, optim_type, loss, lr, weight_decay):
    global classes
    global optimizer
    global loss_fn
    global model

    if model_name == 'densenet121':
        model = models.densenet121(pretrained=False, num_classes=len(classes))
        print('model is densenet121')
        for param in model.parameters():
            param.requires_grad=True

    if model_name == 'resnet18':
        model = models.resnet18(pretrained=False, num_classes=len(classes))
        print('model is resnet18')
        for param in model.parameters():
            param.requires_grad=True

    if model_name == 'resnet34':
        model = models.resnet34(pretrained=False, num_classes=len(classes))
        print('model is resnet34')
        for param in model.parameters():
            param.requires_grad=True

    if optim_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # add more options here

    if optim_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if loss == 'cross-entropy':
        loss_fn = nn.CrossEntropyLoss() # add more options here

    if device == 'cuda':
        device = torch.device("cuda")
        print("device is cuda")
    elif device == 'cpu':
        device = torch.device("cpu")
        print("device is cpu")
    model.to(device);

def train(model, optimizer, loss_fn, train_data_loader, valid_data_loader, epochs, device):

    global train_loss_per_epoch
    global valid_loss_per_epoch

    for epoch in range(epochs):
        epoch += 1 # start at epoch 1 rather than 0
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        print('training epoch {}...'.format(epoch))

        for batch in train_data_loader:
            optimizer.zero_grad()
            inputs, labels = batch
            if device=="cuda":
                inputs = inputs.to(device)
            output = model(inputs)
            if device=="cuda":
                labels = labels.to(device)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item()
        training_loss /= len(train_data_loader)
        print("epoch {} training done".format(epoch))

        model.eval()
        correct = 0
        num_correct = 0
        num_examples = 0

        print("about to run testing")
        for batch in valid_data_loader:
            inputs, labels = batch
            if device=="cuda":
                inputs = inputs.to(device)
            output = model(inputs)
            if device=="cuda":
                labels = labels.to(device)
            loss = loss_fn(output, labels)
            valid_loss += loss.data.item()
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], labels).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
#             writer.add_scalar('accuracy', num_correct / num_examples, epoch)
            #for i, m in enumerate(model.children()):
            #    m.register_forward_hook(partial(send_stats, i))
        valid_loss /= len(valid_data_loader)

        train_loss_per_epoch.append(training_loss/len(train_data_loader))
        valid_loss_per_epoch.append(valid_loss/len(valid_data_loader))

        print("epoch: {}, training loss: {:.3f}, validation loss: {:.3f}, accuracy = {:.2f}".format(epoch, training_loss, valid_loss, num_correct / num_examples))
    print('Finished Training')

def save_model(model_path, model_name):
    global mean
    global std
    global model

    torch.save({
    'model':model.state_dict(),
    'classes':classes,
    'resolution':224,
    'modelType':model_name,
    'mean':mean,
    'std':std
}, model_path)

    print('saved model file to '+model_path)

def plot_training():

    global train_loss_per_epoch
    global valid_loss_per_epoch

    plt.plot(train_loss_per_epoch, label='Training Loss', c='r')
    plt.plot(valid_loss_per_epoch, label='Validation Loss', c='g')
    plt.legend()
    plt.show()

def run_training(opt):
    mean_std(opt.data_path)
    load_data(opt.data_path, opt.batch_size)
    load_model(opt.model_name, opt.device, opt.optim_type, opt.loss, opt.lr, opt.weight_decay)
    train(model, optimizer, loss_fn, train_data_loader, valid_data_loader, opt.epochs, opt.device)
    save_model(opt.model_path, opt.model_name)
    plot_training()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data', help='path to folder containing 2 or more folders of spectrograms labelled by class')
    parser.add_argument('--model_name', type=str, default='resnet18', help='choose from resnet18, resnet34, densenet121')
    parser.add_argument('--device', type=str, help='cuda or cpu')
    parser.add_argument('--optim_type', type=str, default='adam', help='choose from adam or adamw')
    parser.add_argument('--loss', type=str, default='cross-entropy', help='cross-entropy only option for now')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=None, help='add L2 penalty, recommend for small datasets')
    parser.add_argument('--epochs', type=int, help='number of epochs to train')
    parser.add_argument('--model_path', type=str, help='path to model file will be saved, .pth extension')
    parser.add_argument('--batch_size', type=int, help='size of minibatch, larger usually improves learning, lower saves memory')

    opt = parser.parse_args()
    run_training(opt)
