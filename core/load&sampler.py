import numpy as np
import torch
import torch.nn as nn
import pickle
from scipy.ndimage.filters import sobel

epsilon = 1e-10
np.random.seed(147)


def pad_(x, p):
    if len(x.shape)==3:
        return np.pad(x, ((p, p), (p, p), (p, p)), 'constant')
    else:
        return np.pad(x, ((p, p), (p, p), (p, p), (0, 0)), 'constant')


def calc_grad(img):
    dx = sobel(img, axis=0, mode='constant')
    dy = sobel(img, axis=1, mode='constant')
    dz = sobel(img, axis=2, mode='constant')
    grad = abs(dx)+abs(dy)+abs(dz)
    grad = grad/grad.max()
    return grad


class Data_loader():
    def __init__(self, path, img_size=220, patch_size=31, batch_size=1, pad=True, omega=0.01, debug=False):
        self.d = {'pair':0, 'w':1, 'h':1, 't':1}
        self.path = path
        self.mpair = 25
        self.img_size = img_size
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.pad = pad
        self.omega = omega
        self.debug = debug
        self.load_data()

    def load_data(self):
        with open(self.path + '{:d}/data.pkl'.format(self.d['pair']), 'rb') as f:
            x = pickle.load(f)
        self.mov = pad_(x['mov'], self.patch_size//2) if self.pad else x['mov']
        self.sta = pad_(x['sta'], self.patch_size//2) if self.pad else x['sta']
        self.dis = pad_(x['forward'], self.patch_size//2) if self.pad else x['forward']
        self.grad = calc_grad(self.mov)

    def prob(self, w, h, t):
        return np.exp(-self.omega/(np.linalg.norm(self.dis[w, h, t])*self.grad[w, h, t] + epsilon))

    def sample_batch(self):
        cnt = 0
        sta_batch = []
        mov_batch = []
        dis_batch = []
        fp = self.patch_size
        d = self.d
        while cnt < self.batch_size:
            if self.debug:
                print(d)
            pr = self.prob(d['w']+fp//2, d['h']+fp//2, d['t']+fp//2)
            sppr = np.random.random()
            print(pr, sppr)
            # if np.random.random() < self.prob(d['w']+fp//2, d['h']+fp//2, d['t']+fp//2):
            if sppr <= pr:
                cnt += 1
                sta_batch.append(self.sta[d['w']:d['w']+fp, d['h']:d['h']+fp, d['t']:d['t']+fp])
                mov_batch.append(self.mov[d['w']:d['w']+fp, d['h']:d['h']+fp, d['t']:d['t']+fp])
                dis_batch.append(self.dis[d['w']:d['w']+fp, d['h']:d['h']+fp, d['t']:d['t']+fp])
            d['t'] += 1
            if d['t'] >= self.img_size:
                d['t'] = 1
                d['h'] += 1
                if d['h'] >= self.img_size:
                    d['h'] = 1
                    d['w'] += 1
                    if d['w'] >= self.img_size:
                        d['w'] = 1
                        d['pair'] += 1
                        d['pair'] %= self.mpair
                        self.d = d
                        self.load_data()
        self.d = d
        sta_batch = torch.Tensor(np.array(sta_batch))
        mov_batch = torch.Tensor(np.array(mov_batch))
        assert sta_batch.shape == [self.batch_size, 31, 31, 31]

        return sta_batch, mov_batch, np.array(dis_batch)

