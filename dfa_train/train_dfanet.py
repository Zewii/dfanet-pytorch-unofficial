#!/usr/bin/python



from tools.logger import *
from models.dfanet import DFANet
from loader.frdc import Loader
from evaluate_dfanet import MscEval
from optimizer.optimizer import Optimizer_enet
from loss.loss import OhemCELoss
from configs import config_factory

import torch
from torch.utils.data import DataLoader

import os
import os.path as osp
import logging
import time
import datetime


cfg = config_factory['resnet_frdc']
if not osp.exists(cfg.respth): os.makedirs(cfg.respth)

def train(verbose=True):
    torch.cuda.set_device(0)
    setup_logger(cfg.respth)
    logger = logging.getLogger()

    ## dataset
    ds = Loader(cfg, mode='train')
    sampler = torch.utils.data.sampler.RandomSampler(ds, False)
    dl = DataLoader(ds,
                    batch_size = cfg.ims_per_gpu,
                    shuffle = False,
                    sampler = sampler,
                    num_workers = cfg.n_workers,
                    pin_memory = True,
                    drop_last = True)

    ## model
    net = DFANet(3,cfg.n_classes,False,'B')
    net.cuda()
    net.train()
    n_min = cfg.ims_per_gpu*cfg.crop_size[0]*cfg.crop_size[1]//16
    criteria = OhemCELoss(thresh=cfg.ohem_thresh, n_min=n_min).cuda()

    ## optimizer
    optim = Optimizer_enet(
            net,
            cfg.lr_start,
            cfg.momentum,
            cfg.weight_decay,
            cfg.warmup_steps,
            cfg.warmup_start_lr,
            cfg.max_iter,
            cfg.lr_power
            )

    ## train loop
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)
    n_epoch = 0
    for it in range(cfg.max_iter):
        try:
            im, lb = next(diter)
            if not im.size()[0]==cfg.ims_per_gpu: continue
        except StopIteration:
            n_epoch += 1
            diter = iter(dl)
            im, lb = next(diter)
        im = im.cuda()
        lb = lb.cuda()

        H, W = im.size()[2:]
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        logits = net(im)
        loss = criteria(logits, lb)
        loss.backward()
        optim.step()

        loss_avg.append(loss.item())
        ## print training log message
        if it%cfg.msg_iter==0 and not it==0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((cfg.max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds = eta))
            msg = ', '.join([
                    'iter: {it}/{max_it}',
                    'lr: {lr:4f}',
                    'loss: {loss:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                ]).format(
                    it = it,
                    max_it = cfg.max_iter,
                    lr = lr,
                    loss = loss_avg,
                    time = t_intv,
                    eta = eta
                )
            logger.info(msg)
            loss_avg = []
            st = ed
        if it%5000==0 and it!=0:
            net.cpu()
            save_pth = osp.join(cfg.respth, 'model_final_'+str(it)+'.pth')
            state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            torch.save(state, save_pth)
            logger.info('training done, model saved to: {}'.format(save_pth))
            net.cuda()
    ## dump the final model and evaluate the result
    if verbose:
        net.cpu()
        save_pth = osp.join(cfg.respth, 'model_final.pth')
        state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
        torch.save(state, save_pth)
        logger.info('training done, model saved to: {}'.format(save_pth))
        logger.info('evaluating the final model')
        net.cuda()
        net.eval()
        evaluator = MscEval(cfg)
        mIOU = evaluator(net)
        logger.info('mIOU is: {}'.format(mIOU))


if __name__ == "__main__":
    train(True)

