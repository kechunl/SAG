
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import pdb

class SAGloss(nn.Module):
    def __init__(self, loss_type, reduce=True, reduction='mean', *args, **kwargs):
        super(SAGloss, self).__init__()
        self.loss_type = loss_type # mse, or in-out
        self.reduce = reduce
        self.reduction = reduction

    def compute_mse_loss(self, attn_wts, target):
        '''
        Compute frobenius norm loss for attention map
        :params attn_wts: N
        :params target: N
        :return loss
        '''
        loss = ((attn_wts - target) ** 2).sum()
        return loss
    
    def compute_inout_loss(self, attn_wts, target):
        '''
        Compute inclusion & exclusion loss for attention map
        :params attn_wts: N
        :params target: N
        :return loss
        '''
        loss = -(attn_wts * (target > 0)).sum() * 1e-2 + (attn_wts * (target == 0)).sum() * 1e-2
        # loss = (attn_wts * (target == 0)).sum() * 1e-1
        return loss
    
    def compute_loss(self, pred, target):
        if self.loss_type == "mse":
            return self.compute_mse_loss(pred, target)
        if self.loss_type == "in-out":
            return self.compute_inout_loss(pred, target)
        raise ValueError('Unknown loss type')

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        '''
        pred: 1 x N
        target: N
        '''
        if target.sum() == 0:
            return torch.tensor(0.0)
        pred = pred.squeeze(0)
        loss = self.compute_loss(pred, target)
        return loss