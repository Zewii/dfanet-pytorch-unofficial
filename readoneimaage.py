#!/usr/bin/python
# -*- encoding: utf-8 -*-

from tools.logger import *
# from models.deeplabv3plus import Deeplab_v3plus
from models.dfanet import DFANet
from configs import config_factory

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import logging
import numpy as np
import cv2
from to_color import cloor,cloor2
from PIL import Image
from torchvision import transforms

def MscEval(cfg, net, im):
    with torch.no_grad():
        im = im.cuda()
        out = net(im)
        prob = F.softmax(out, 1)
    pred = prob.argmax(dim=1).cpu().numpy() 
    return pred


def read_one_image(image):
    cfg = config_factory['resnet_mydataset']
    net = DFANet(cfg.n_classes,False)
    save_pth = osp.join(cfg.respth, 'model_final.pth')
    net.load_state_dict(torch.load(save_pth))
    net.cuda()
    net.eval()
    loader = transforms.Compose([transforms.ToTensor(),transforms.Normalize(cfg.mean,cfg.std)])  # transform it into a torch tensor
    time_st = time.time()
    r = MscEval(cfg,net,loader(Image.open(image)).unsqueeze(0))
    time_ed = time.time()
    print("FPS:",str(1./(time_ed-time_st)))
    r = r[0]
    r=cloor2(r)
    r = (r*0.5).astype('uint8')
    i = np.array(cv2.imread(image)*0.5).astype('uint8')
    return r+i

def read_video(video,net,cfg):
    cap = cv2.VideoCapture(video)
    image = None
    loader = transforms.Compose([transforms.ToTensor(),transforms.Normalize(cfg.mean,cfg.std)])
    while cap.grab():
        _, image = cap.retrieve()
        time_st = time.time()
        image = cv2.resize(image,(768,768))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image)
        image = loader(image).unsqueeze(0)
        r = MscEval(cfg,net,image)
        time_ed = time.time()
        print("FPS:",str(1./(time_ed-time_st)))
    return 
        
if __name__=='__main__':
    cfg = config_factory['resnet_frdc']
    net = DFANet(cfg.n_classes,False)
    save_pth = osp.join(cfg.respth, 'model_final.pth')
    net.load_state_dict(torch.load(save_pth))
    net.cuda()
    net.eval()
    read_video("1.m2p",net,cfg)
