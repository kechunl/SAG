
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import pdb

class AttnGuideReg(nn.Module):
    def __init__(self, opts, head_ind=None, reduce=True, reduction='mean', *args, **kwargs):
        super(AttnGuideReg, self).__init__()
        self.attn_layer = opts['attn_layer'] # which attn layer to impose attn guiding loss
        self.select_attn_head = opts['select_attn_head'] # fixed attn head or random attn head
        self.num_heads = opts['attn_head'] # how many heads
        self.head_ind = head_ind
        self.loss_type = opts['attn_loss'] # mse or in-out
        self.reduce = reduce
        self.reduction = reduction
    
    def gather_attn_map(self, attn_wts, bs):
        '''
        Gather attention map for loss calculation
        '''
        attn_wts_gathered = []
        if self.select_attn_head == 'random':
            head_idx = torch.randint(0, 4, (self.num_heads,), device=attn_wts[0][0].device)
        else:
            head_idx = torch.arange(self.num_heads, device=attn_wts[0][0].device) if self.head_ind is None else self.head_ind.to(attn_wts[0][0].device)
        for s in range(len(attn_wts)): # num of scales
            P = attn_wts[s][0].shape[-1]
            attn_map = []
            for l in range(len(attn_wts[s])): # layer in transformer
                attn = torch.index_select(attn_wts[s][l].view(bs, -1, P, P), 1, head_idx) # B x n_h x P x P
                attn = attn.mean(dim=-2) # B x n_h x P 
                attn_map.append(attn)
            attn_wts_gathered.append(attn_map)
        return attn_wts_gathered

    def compute_mse_loss(self, attn_wts, target, bs):
        '''
        Compute mse loss for attention map
        :params attn_wts: [[[attn_map_layer1], [attn_map_layer2], ...], ...] each attn map is [B x n_h x P x P], len=num_scales
        :params target: [[B x sqrt(p) x sqrt(p)], ...] len=num_scales
        :return loss
        '''
        loss = torch.zeros(bs).to(target[0].device)
        # scale
        for s in range(len(attn_wts)):
            # layer
            for l in self.attn_layer:
                loss += ((attn_wts[s][l] - target[s].view(bs,1,-1)) ** 2).sum(dim=(1,2))
        if self.reduce:
            loss = loss.sum()
        return loss
    
    def compute_inout_loss(self, attn_wts, target, bs):
        '''
        Compute in-out loss for attention map
        :params attn_wts: [[[attn_map_layer1], [attn_map_layer2], ...], ...] each attn map is [B x n_h x P x P], len=num_scales
        :params target: [[B x sqrt(p) x sqrt(p)], ...] len=num_scales
        :return loss
        '''
        loss = torch.zeros(bs).to(target[0].device)
        for s in range(len(attn_wts)):
            for l in self.attn_layer:
                loss += -(attn_wts[s][l] * (target[s].view(bs,1,-1) > 0)).sum(dim=(1,2)) * 1e-2
                loss += (attn_wts[s][l] * (target[s].view(bs,1,-1) == 0)).sum(dim=(1,2)) * 1e-2

        if self.reduce:
            loss = loss.sum()
        return loss
    
    def compute_loss(self, pred, target, bs):
        if self.loss_type == "mse":
            return self.compute_mse_loss(pred, target, bs)
        if self.loss_type == "in-out":
            return self.compute_inout_loss(pred, target, bs)
        raise ValueError('Unknown loss type')
    
    def filter_zero(self, pred, target):
        nonzero_index = torch.sum(target[0], dim=(1,2)).nonzero()[:,0]
        target = [t[nonzero_index,...] for t in target]
        pred = [[l[nonzero_index,...] for l in s] for s in pred]
        return pred, target

    def forward(self, pred: Tensor, target: Tensor, bs: int) -> Tensor:
        pred = self.gather_attn_map(pred, bs)
        # some data sample don't have attn_map, the corresponding target will be all 0
        pred, target = self.filter_zero(pred, target)
        bs = target[0].shape[0]
        if bs == 0:
            return torch.tensor(0.0)

        loss = self.compute_loss(pred, target, bs)
        if self.reduction == 'mean':
            loss /= bs
        return loss