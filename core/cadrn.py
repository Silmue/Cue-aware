import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CADRN(nn.Module):
    def __init__(self, patch_size, kernel_size):
        super(CADRN, self).__init__()
        self.patch_size = patch_size
        self.patch_center = (patch_size-1)//2
        self.kernel_size = max(2, kernel_size)
        self.cPool = ChannelPool(self.kernel_size-1)
        self.DRN = DRN()
        self.DRN.init_weight()

    # def init_weight(self):
    #         for m in self.modules():
    #             if isinstance(m, nn.Conv3d):
    #                 nn.init.xavier_normal_(m.weight)
    #                 if m.bias is not None:
    #                     nn.init.zeros_(m.bias)

    def forward(self, input):
        c = self.patch_center
        k = self.kernel_size
        context_cue = None
        for sta, mov in zip(input[0], input[1]):
            tmp = None
            mov = mov.unsqueeze(0).unsqueeze(0)
            sta = sta.unsqueeze(0).unsqueeze(0)
            for s in range(2, k+1):
                x = F.conv3d(mov, sta[:, :, c-s:c+s+1, c-s:c+s+1, c-s:c+s+1], padding=s)
                try:
                    tmp = torch.cat((tmp, x), dim=1)
                except:
                    tmp = x
            try:
                context_cue = torch.cat((context_cue, tmp), dim=0)
            except:
                context_cue = tmp
        pooled_cue = self.cPool(context_cue)
        return self.DRN(torch.cat((input[0].unsqueeze(1), input[1].unsqueeze(1), pooled_cue), dim=1))


class ChannelPool(nn.MaxPool1d):
    def __init__(self, kernel_size):
        super(ChannelPool, self).__init__(kernel_size)
        self.kernel_size = kernel_size

    def forward(self, input):
        n, c, t, w, h = input.size()
        input = input.view(n, c, t*w*h).permute(0, 2, 1) * -1
        pooled = F.max_pool1d(input, kernel_size=self.kernel_size, padding=0) * -1
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, t, w, h)


def downconv(in_chnls, out_chnls, kernel_size):
    return nn.Sequential(
        nn.Conv3d(in_chnls, out_chnls, kernel_size,
                  stride=2, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_chnls, out_chnls, kernel_size,
                  stride=1, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True)
    )


def fclayer(in_chnls, out_chnls, tanh_activation=False):
    return nn.Sequential(
        nn.Linear(in_chnls, out_chnls),
        nn.BatchNorm1d(out_chnls),
        nn.Tanh() if tanh_activation else nn.ReLU()
    )


class DRN(nn.Module):
    def __init__(self):
        super(DRN, self).__init__()
        self.conv1 = downconv(3, 64, 3)
        self.conv2 = downconv(64, 128, 3)
        self.pool1 = nn.MaxPool3d(3, stride=2, padding = 1)
        self.conv3 = downconv(128, 256, 3)
        self.conv4 = downconv(256, 512, 3)
        self.fc1 = fclayer(512, 4096)
        self.fc2 = fclayer(4096, 1024)
        self.fc3 = fclayer(1024, 128)
        self.fc4 = fclayer(128, 3, tanh_activation=True)


    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input):
        conv1_out = self.conv1(input)
        conv2_out = self.conv2(conv1_out)
        pool1_out = self.pool1(conv2_out)
        conv3_out = self.conv3(pool1_out)
        conv4_out = self.conv4(conv3_out).squeeze()
        fc1_out = self.fc1(conv4_out)
        fc2_out = self.fc2(fc1_out)
        fc3_out = self.fc3(fc2_out)
        fc4_out = self.fc4(fc3_out)
        return fc4_out
