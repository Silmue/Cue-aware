import numpy as np
import torch
import pickle

epsilon = 1e-10
np.random.seed(147)


class Data_loader():
    def __init__(self, path, batch_size, num_data):
        self.path = path + '{:d}.pkl'
        self.batch_size = batch_size
        self.num_data = num_data
        self.lst = np.arange(self.num_data)
        self.reset()

    def reset(self):
        self.cur = 0
        np.random.shuffle(self.lst)

    def data_batch(self):
        sta_batch = []
        mov_batch = []
        dis_batch = []
        for _ in range(self.batch_size):
            with open(self.path.format(self.lst[self.cur]), 'rb') as f:
                x = pickle.load(f)
            sta_batch.append(x['sta'])
            mov_batch.append(x['mov'])
            dis_batch.append(x['dis'])
            self.cur += 1
            self.cur %= self.num_data
        sta_batch = torch.Tensor(np.array(sta_batch))
        mov_batch = torch.Tensor(np.array(mov_batch))
        dis_batch = torch.Tensor(np.array(dis_batch))
        return sta_batch, mov_batch, dis_batch
