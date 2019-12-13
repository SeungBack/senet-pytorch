import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import ResNet, ResBottleNeck, SEBottleNeck, MSEBottleNeck, WSEBottleNeck, DSEBottleNeck,init_weight
import numpy as np
from PIL import Image
import argparse
import os
from tqdm import tqdm
from radam import RAdam
from tensorboardX import SummaryWriter
import time
import datetime
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import cv2
from radam import RAdam
from thop import profile

def trainer(args):

    # load CIFAR 100
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR100('./dataset', train=True, transform=train_transform, download=True)
    val_dataset = datasets.CIFAR100('./dataset', train=False, transform=val_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.block == 'res':
        block = ResBottleNeck
    elif args.block == 'se':
        block = SEBottleNeck
    elif args.block == 'mse':
        block = MSEBottleNeck
    elif args.block == 'wse':
        block = WSEBottleNeck
    elif args.block == 'dse':
        block = DSEBottleNeck

    model = ResNet(inplanes=args.inplanes, block=block).to(device)

    if args.load_weights:
        model.load_state_dict(torch.load(args.load_weights))
    else:
        model.apply(init_weight)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # calculate number of parameter & flops
    flops, params = profile(model, inputs=(torch.rand(1, 3, 32, 32).to(device), ))
    print("total number of trainable parameter is :", params)
    print("number of FLOPS : ", flops)

    # define optimizer & criterion
    optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    # for save and log results
    now = datetime.datetime.now()
    output_folder = './results/' + now.strftime('%Y-%m-%d_%H-%M-%S')
    os.mkdir(output_folder)
    writer = SummaryWriter(logdir=output_folder)

    # training loop
    for epoch in range(args.max_epoch):
        model.train()

        for i, (images, labels) in enumerate(tqdm(train_loader)):

            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            curr_iter = epoch * len(train_loader) + i
            writer.add_scalar('batch_loss', loss.item(), curr_iter)
            if i % (len(train_loader)//4) == 0:
                print('Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.4f}'.format(epoch+1, args.max_epoch, i+1, len(train_loader), loss.item()))

        # evalulation
        val_loss, val_top1_acc, val_top5_acc = evaluate(args, model, device, val_loader, criterion)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_top1_acc', val_top1_acc, epoch)
        writer.add_scalar('val_top5_acc', val_top5_acc, epoch)

        print('Epoch [{}/{}], Val Loss: {:.4f}, Val Acc: top1-{:.2f}, top5-{:.2f}'.format(epoch+1, args.max_epoch, val_loss, val_top1_acc, val_top5_acc))

        if epoch % 5 == 0:
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), '{}/epoch_{}.tar'.format(output_folder, epoch))
            else:
                torch.save(model.state_dict(), '{}/epoch_{}.tar'.format(output_folder, epoch))


def evaluate(args, model, device, data_loader, criterion):
    model.eval()
    with torch.no_grad():

        val_loss = 0
        n_top1_correct = 0
        n_top5_correct = 0
        n_total = len(data_loader)

        for images, labels in tqdm(data_loader):

            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            loss = criterion(preds, labels)

            val_loss += loss.item()

            _, preds_top1 = preds.max(1)
            n_top1_correct += (preds_top1==labels).float().sum()

            _, preds_top5 = preds.topk(5, 1, True, True)
            n_top5_correct += (preds_top5==labels.expand_as(preds_top5)).float().sum()

        val_loss = val_loss / n_total
        val_top1_acc = n_top1_correct / n_total * 100
        val_top5_acc = n_top5_correct / n_total * 100

        return val_loss, val_top1_acc, val_top5_acc