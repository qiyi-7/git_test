import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import datasets,transforms,models

import os
import argparse


import cv2
import torch.utils.data as data
from PIL import Image 

class pets_data(data.Dataset):
    def __init__(self, txt_name, train):
        self.train = train
        self.txt_list = open(txt_name).read().split('\n')[:-1]
        self.txt_size = len(self.txt_list)
        print('txt_size:',self.txt_size)

    def __getitem__(self, index):        
          
        image_name_id = self.txt_list[index].split(' ')              
        im = Image.open('/home/qiyi/PETS/images/' + image_name_id[0] + '.jpg')
        
        if self.train:
            im = train_transforms(im)
        else:
            im = test_transforms(im)
        targets = torch.tensor(int(image_name_id[1])-1)

        return im, targets

    def __len__(self):
        return self.txt_size



train_dataset = pets_data('/home/qiyi/PETS/annotations/trainval.txt', True)
val_dataset = pets_data('/home/qiyi/PETS/annotations/test.txt', False)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)







print('==> Building model..')
net = models.resnet50(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 37)


checkpoint = torch.load('./ckpt.pth')
net.load_state_dict(checkpoint)


net = net.cuda()
print(net)

print('==> Preparing data..')

train_transforms = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9,weight_decay=1e-3)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)


# Training
def train(epoch):
    net.train()
    exp_lr_scheduler.step()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        
        inputs, targets = inputs.cuda(), targets.cuda()     
           
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 20 == 0:
            print('epoch:',epoch,'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))




def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            print(inputs)
            
            outputs = net(inputs)
            print(outputs)
            break
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 20 == 0:
                print(epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    # Save checkpoint.
    best_acc = 0
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        #torch.save(net.state_dict(), './ckpt.pth')
        best_acc = acc


for epoch in range(0, 1):
    #train(epoch)
    test(epoch)
#torch.save(net.state_dict(),'final.pth')




