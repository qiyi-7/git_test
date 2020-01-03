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
from PIL import ImageFont
from PIL import ImageDraw
from numpy import unicode
         


print('==> Building model..')
net = models.resnet50(pretrained=False)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 37)


checkpoint = torch.load('./ckpt.pth')
net.load_state_dict(checkpoint)

net = net.cuda().eval()


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


categroy = open('annotations/names.txt', 'r').read().split('\n')[:-1]
categroy = [line.split(' ') for line in categroy]


txt_list = open('/home/qiyi/PETS/annotations/test.txt').read().split('\n')[:-1]
for i in range(0, len(txt_list), 500):
    image_name_id = txt_list[i].split(' ')[0]

    im = Image.open('/home/qiyi/PETS/images/' + image_name_id + '.jpg')
    inputs = test_transforms(im)
    inputs = inputs.unsqueeze(0).cuda()

    outputs = net(inputs)

    idx = outputs.argmax().item()
    print(image_name_id, categroy[idx][1])
    
    
    
    font = ImageFont.truetype('NotoSansCJK-Bold.ttc',25)
    fillColor = (255,255,255)
    position = (10,10)
    chinese = categroy[idx][1]
    draw = ImageDraw.Draw(im)
    draw.text(position,chinese,font=font,fill=fillColor)
    im.save('sample/' + image_name_id + '.jpg')








