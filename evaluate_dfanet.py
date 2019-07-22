#!/usr/bin/python
# -*- encoding: utf-8 -*-

from tools.logger import *
from models.dfanet import DFANet
from loader.cityscapes import CityScapes
from loader.frdc import Loader
from configs import config_factory
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sys
import os
import logging
import numpy as np
# import numba
class MscEval(object):
    def __init__(self, cfg):
        self.cfg = cfg
        ## dataloader
        dsval = Loader(cfg, mode='val')
        sampler = None
        self.dl = DataLoader(dsval, batch_size = cfg.eval_batchsize, sampler = sampler, shuffle = False, num_workers = cfg.eval_n_workers, drop_last = False)

    def __call__(self, net):
        ## evaluate
        hist_size = (self.cfg.n_classes, self.cfg.n_classes)
        hist = np.zeros(hist_size, dtype=np.float32)
        diter = enumerate(self.dl)
        for i, (imgs, label) in diter:
            N, _, H, W = label.shape
            probs = torch.zeros((N, self.cfg.n_classes, H, W))
            probs.requires_grad = False
            for sc in self.cfg.eval_scales:
                new_hw = [int(H*sc), int(W*sc)]
                with torch.no_grad():
                    im = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True)
                    im = im.cuda()
                    out = net(im)
                    out = F.interpolate(out, (H, W), mode='bilinear', align_corners=True)
                    prob = F.softmax(out, 1)
                    probs += prob.cpu()
                    if self.cfg.eval_flip:
                        out = net(torch.flip(im, dims=(3,)))
                        out = torch.flip(out, dims=(3,))
                        out = F.interpolate(out, (H, W), mode='bilinear',
                                align_corners=True)
                        prob = F.softmax(out, 1)
                        probs += prob.cpu()
                    del out, prob
            probs = probs.data.numpy()
            preds = np.argmax(probs, axis=1)
            hist_once = self.compute_hist(preds, label.data.numpy().squeeze(1))
            hist = hist + hist_once
        IOUs = np.diag(hist) / (np.sum(hist, axis=0)+np.sum(hist, axis=1)-np.diag(hist))
        mIOU = np.mean(IOUs)
        return mIOU

    def compute_hist(self, pred, lb):
        n_classes = self.cfg.n_classes
        keep = np.logical_not(lb==self.cfg.ignore_label)
        merge = pred[keep] * n_classes + lb[keep]
        hist = np.bincount(merge, minlength=n_classes**2)
        hist = hist.reshape((n_classes, n_classes))
        return hist


def evaluate():
    ## setup
    cfg = config_factory['resnet_mydataset']
    setup_logger(cfg.respth)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger()

    ## model
    logger.info('setup and restore model')
    net = DFANet(cfg.n_classes,False)
    save_pth = os.path.join(cfg.respth, 'model_final.pth')
    net.load_state_dict(torch.load(save_pth))
    net.cuda()
    net.eval()

    ## evaluator
    logger.info('compute the mIOU')
    evaluator = MscEval(cfg)
    mIOU = evaluator(net)
    logger.info('mIOU is: {:.6f}'.format(mIOU))


if __name__ == "__main__":
    evaluate()
