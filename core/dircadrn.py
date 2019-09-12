import torch
import torch.nn as nn
import numpy as np
from data_loader import Data_loader
from cadrn import CADRN
import os

gpus = [0]
cuda_gpu = torch.cuda.is_available()

class DIRCADRN():
    def __init__(self, config):
        self.cadrn = CADRN(config['patch_size'], config['kernel_size'])
        if cuda_gpu:
            self.cadrn = torch.nn.DataParallel(self.cadrn, device_ids=gpus).cuda()
        self.config = config
        self.lossfun = nn.MSELoss()

    def train_inside_epoch(self, debug):
        config = self.config
        loss_sum = 0
        self.train_loader.reset()
        for i in range(config['n_iters']):
            sta, mov, disfield = self.train_loader.data_batch()
            if cuda_gpu:
                sta = sta.cuda()
                mov = mov.cuda()
            dis_pred = self.cadrn((sta, mov)) * config['delta']
            if i%20 == 0 and debug:
                print(disfield)
                print(dis_pred)

            loss = self.lossfun(dis_pred, disfield)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print('epoch {:d}, batch {:d}, loss = {:.2f}'.format(self.epoch, i, loss.item()))
            loss_sum += loss.item()
        return loss_sum/config['n_iters']

    def train(self, debug=False):
        config = self.config

        self.train_loader = Data_loader(path=config['data'], batch_size=config['batch_size'], num_data=config['num_data'])

        optim_params = [{'params': self.cadrn.DRN.parameters(), 'lr': config['lr']}]
        self.optimizer = torch.optim.Adam(
            optim_params,
            betas=(self.config['momentum'], self.config['beta']),
            weight_decay=self.config['weight_decay'])

        losses = []
        if not os.path.exists(config['ckpt_path']):
            os.mkdir(config['ckpt_path'])
        for i in range(config['epoch']):
            self.epoch = i
            epoch_loss = self.train_inside_epoch(debug)
            losses.append(epoch_loss)
            torch.save({
                'epoch': i,
                'drn_state_dict': self.cadrn.DRN.state_dict(),
                'loss': epoch_loss
            }, config['ckpt_path'] + 'epoch={:d},loss={:.2f}'.format(i, epoch_loss))


    def evaluate(self):
        pass
