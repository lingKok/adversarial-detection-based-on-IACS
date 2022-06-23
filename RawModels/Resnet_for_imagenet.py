import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier
from torch.utils.data import SubsetRandomSampler
from torchvision.models import resnet34
from torchvision.transforms import transforms
import torchvision.datasets as datasets



def resnet34_imagenet():
    model = resnet34(pretrained=True)
    in_feature = model.fc.in_features
    model.fc = nn.Linear(in_feature,10)
    # model = PyTorchClassifier(model,loss=None,input_shape=(3,224,224),nb_classes=10)
    return model

def loadData(path='./imagenet',batch_size=10,shuffle=True,val_size =0.0):

    #set the path of dataset
    train_path = os.path.join(path + 'train')
    val_path = os.path.join(path + 'val')

    #set transform method, the default img size of imagenet is 224
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize(330),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalizer
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # normalizer
    ])

    #import train\val dataset
    train_data = datasets.ImageFolder(root=train_path, transform=train_transform,)
    valid_data = datasets.ImageFolder(root=val_path, transform=valid_transform,)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(val_size*num_train))
    np.random.seed(0)
    np.random.shuffle(indices)
    train_idx,valid_idx = indices[split:],indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)

    # load dataset into Dataloader
    train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=0)
    #
    print(len(train_data)) #1151273
    print(len(valid_data)) #50000
    return train_queue,valid_queue

def train_one_epoch(epoch):
    raw_model.model.train()
    for i,data in enumerate(train_load):
        images,labels=data
        len=images.size()[0]
        images,labels=images.to(device),labels.to(device)
        # stylized_images=get_stylized_imges(images)
        # print(images.shape)
        # print(stylized_images.shape)
        # data=torch.cat((images,stylized_images),0)
        # label=torch.cat((labels,labels))
        optimizer.zero_grad()
        output=raw_model.model(images)
        loss=loss_func(output,labels)
        # loss2=loss_func2(output[:len],output[len:])
        # loss=loss1+loss2

        train_loss.update(loss.cpu().data,len)
        loss.backward()
        optimizer.step()
        if i%200==0:
            # write.add_images('adamin_images',)
            # print('epoch: {} loss: {:2f}'.format(epoch,train_loss.avg))
            print('epoch: {} loss: {:2f}'.format(epoch,train_loss.avg))
    # save_path='models/cifar/normal/resnet34_'+str(epoch)+'.pth'
    # torch.save(model.state_dict(),save_path)



def validate():
    from utils import AverageMeter, accuracy
    raw_model.model.eval()
    test_loss.reset()
    test_acc.reset()
    for i ,data in  enumerate(val_load):
        images,labels = data
        len=images.size()[0]
        images, labels = images.to(device), labels.to(device)
        # stylized_images = get_stylized_imges(images)
        # data=torch.cat((images,stylized_images),0)
        # label=torch.cat((labels,labels))
        output=raw_model.model(images)
        loss=loss_func(output,labels.long())
        # loss2=loss_func2(output[:len],output[len:])
        # loss=loss1+loss2
        test_loss.update(loss.cpu().data,len)
        acc=accuracy(output,labels)
        test_acc.update(acc,len)
        if i%200==0:
            # print('loss:{:.2f} acc: {:.2f}'.format(test_loss.avg,test_acc.avg))
            print('loss:{:.2f} acc: {:.2f}'.format(test_loss.avg,test_acc.avg))
    print('overall loss:{:.2f} acc: {:.2f}'.format(test_loss.avg, test_acc.avg))
    return test_acc.avg

if __name__ == '__main__':
    from utils import AverageMeter

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw_model = resnet34_imagenet()

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=raw_model.parameters(), lr=0.0001)
    raw_model = PyTorchClassifier(raw_model,loss_func,input_shape=(3,224,224),nb_classes=10,optimizer=optimizer)
    path = './IMAGENET10/model/'
    file_name = 'IMAGENET10_raw.pt'
    save_path = os.path.join(path,file_name)
    if os.path.exists(save_path):
        raw_model.model.load_state_dict(torch.load(save_path))
        print('load weight success!')
    raw_model.model.to(device)
    train_load,val_load = loadData('./IMAGENET10/dataset/')
    train_loss = AverageMeter()

    test_loss = AverageMeter()
    test_acc = AverageMeter()

    # raw_model.fit(train_data.imgs.numpy(),train_data.targets.numpy(),batch_size=128,nb_epochs=1)
    for i in range(10):
        train_one_epoch(i)
        validate()

    state_dict = raw_model.model.state_dict()
    torch.save(state_dict,save_path)



