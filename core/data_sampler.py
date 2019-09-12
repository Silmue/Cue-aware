import numpy as np
import pickle
from scipy.ndimage.filters import sobel
import os

epsilon = 1e-10
np.random.seed(147)


def pad_(x, p):
    if len(x.shape) == 3:
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


class Data_sampler():
    def __init__(self, path, outpath, img_size=[199, 185, 199], patch_size=31, batch_size=1, pad=True, omega=0.01, delta = 7, debug=False):
        self.path = path
        self.outpath = outpath.format(omega)
        self.mpair = 25
        self.img_size = img_size
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.pad = pad
        self.omega = omega
        self.delta = delta
        self.debug = debug

    def load_data(self, pair):
        with open(self.path + '{:d}/data.pkl'.format(pair), 'rb') as f:
            x = pickle.load(f)
        self.mov = pad_(x['mov'], self.patch_size//2) if self.pad else x['mov']
        self.sta = pad_(x['sta'], self.patch_size//2) if self.pad else x['sta']
        self.dis = pad_(x['dvf'], self.patch_size//2) if self.pad else x['dvf']
        self.grad = calc_grad(self.mov)

    def prob(self, w, h, t):
        tmp = np.linalg.norm(self.dis[w, h, t])
        if tmp > self.delta:
            return 0
        return np.exp(-self.omega/(tmp*self.grad[w, h, t] + epsilon))

    def sample_data(self):
        img_size = self.img_size
        cnt = 0
        fp = self.patch_size
        path = self.outpath
        if not os.path.exists(path):
            os.mkdir(path)
        for pair in range(self.mpair):
            self.load_data(pair)
            print('sampling {:d}th pair'.format(pair))
            for w in range(img_size[0]):
                for h in range(img_size[1]):
                    for t in range(img_size[2]):
                        pr = self.prob(w+fp//2, h+fp//2, t+fp//2)
                        sppr = np.random.random()
                        if self.debug:
                            print(pr, sppr)
                        # if np.random.random() < self.prob(d['w']+fp//2, d['h']+fp//2, d['t']+fp//2):
                        if sppr <= pr:
                            tmp = {
                                'sta': self.sta[w:w+fp, h:h+fp, t:t+fp],
                                'mov': self.mov[w:w+fp, h:h+fp, t:t+fp],
                                'dis': self.dis[w+fp//2, h+fp//2, t+fp//2, :]
                            }
                            with open(path+'{:d}.pkl'.format(cnt), 'wb') as f:
                                pickle.dump(tmp, f)
                            cnt += 1
                            if cnt % 1000 == 0:
                                print('pair{:d}, got {:d} samples'.format(pair, cnt))


if __name__=='__main__':
    path = '../../../dataset/LPBA/LPBA40/pair/'
    outpath = '../../../dataset/LPBA/samples/'
    sampler = Data_sampler(path, outpath, omega=10)
    sampler.sample_data()




