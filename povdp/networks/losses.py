import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import povdp.utils as utils


class WeightedPolicyLoss(nn.Module):

    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer('weights', weights)
        self.action_dim = action_dim

    def forward(self, pred, targ, mask=None):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        if mask is None:
            weighted_loss = (loss * self.weights).mean()
        else:
            weighted_loss = (loss * self.weights * mask).mean()
        a0_loss = (loss[:, 0, :self.action_dim] / self.weights[0, :self.action_dim]).mean()
        return weighted_loss, {'a0_loss': a0_loss}


class WeightedPolicyL1(WeightedPolicyLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedPolicyL2(WeightedPolicyLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')


class ValueLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        pass

    def forward(self, pred, targ):
        loss = self._loss(pred, targ).mean()

        if len(pred) > 1:
            corr = np.corrcoef(
                utils.to_np(pred).squeeze(),
                utils.to_np(targ).squeeze()
            )[0,1]
        else:
            corr = np.NaN

        info = {
            'mean_pred': pred.mean(), 'mean_targ': targ.mean(),
            'min_pred': pred.min(), 'min_targ': targ.min(),
            'max_pred': pred.max(), 'max_targ': targ.max(),
            'corr': corr,
        }

        return loss, info


class ValueL1(ValueLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class ValueL2(ValueLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')


PolicyLosses = {
    'l1': WeightedPolicyL1,
    'l2': WeightedPolicyL2,
    'value_l1': ValueL1,
    'value_l2': ValueL2,
}
    

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, C1=1e-4, C2=9e-4, channel=None, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2
        self.channel = channel
        self.size_average = size_average

    def forward(self, img1, img2):
        if self.channel is None:
            self.channel = img1.size(1)
            
        window = self.create_window(self.channel).to(img1.get_device())
        
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(
            img1 * img1, 
            window, 
            padding=self.window_size // 2, 
            groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(
            img2 * img2, 
            window, 
            padding=self.window_size // 2, 
            groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(
            img1 * img2, 
            window, 
            padding=self.window_size // 2, 
            groups=self.channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) /\
              ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)

    def create_window(self, channel):
        def _gaussian(window_size, sigma):
            gauss = torch.from_numpy(
                np.exp(-(np.arange(window_size) - window_size // 2) ** 2 / float(2 * sigma ** 2))
            )
            return gauss / gauss.sum()

        _1D_window = _gaussian(self.window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(
            channel, 1, self.window_size, self.window_size).contiguous()
        return window

