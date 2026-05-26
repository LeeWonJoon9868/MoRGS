#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from utils.image_utils import coords_grid

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def mse_loss(network_output, gt):
    return torch.sqrt(torch.sum((network_output - gt) ** 2, dim=0)).mean()

def lp_loss(network_output, gt, p = 4):
    return ((network_output - gt) ** p).mean()

def tv_loss(network_output):
    v_loss = (network_output[...,1:,:]-network_output[...,:-1,:]).pow(2).mean()
    h_loss = (network_output[...,:,1:]-network_output[...,:,:-1]).pow(2).mean()
    return h_loss + v_loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class DepthRelLoss(torch.nn.Module):
    def __init__(self, H, W, pix_diff = 10, num_comp=3, tolerance=0.05):
        super(DepthRelLoss, self).__init__()

        self.H, self.W = H,W
        self.pix_diff, self.num_comp, self.tolerance = pix_diff, num_comp, tolerance
        self.grid = coords_grid(1,H,W, device='cuda').unsqueeze(-1).long()
        self.grid.requires_grad_(False)
        self.zeros = torch.zeros_like(self.grid)
        self.resample_pairs()

    @torch.no_grad()
    def resample_pairs(self):
        size = (1,2,self.H,self.W,self.num_comp)
        pix_shift = torch.randint(1,self.pix_diff+1,size=size).to(self.grid)
        sign = torch.randint(0,2,size=size).to(self.grid)*2-1
        pix_shift = pix_shift*sign
        grid_shift = self.grid+pix_shift # B,2,H,W,num_comp
        grid_shift[:,0] += 2*torch.maximum(-grid_shift[:,0],self.zeros[:,0])
        grid_shift[:,0] -= 2*torch.maximum(grid_shift[:,0]-(self.W-1),self.zeros[:,0])
        grid_shift[:,1] += 2*torch.maximum(-grid_shift[:,1],self.zeros[:,1])
        grid_shift[:,1] -= 2*torch.maximum(grid_shift[:,1]-(self.H-1),self.zeros[:,1])
        self.grid_shift = grid_shift.long()


    def forward(self, pred_depth, gt_depth):
        src_gt = gt_depth[self.grid[0,1,...],self.grid[0,0,...]] # HxWx1
        tgt_gt = gt_depth[self.grid_shift[0,1,...],self.grid_shift[0,0,...]] #HxWxnum_comp
        src_pred = pred_depth[self.grid[0,1,...],self.grid[0,0,...]] # HxWx1
        tgt_pred = pred_depth[self.grid_shift[0,1,...],self.grid_shift[0,0,...]] #HxWxnum_comp
        rel_depth_mask = src_gt/(tgt_gt+1.0e-8) # HxWx1
        rel_depth_mask_pos = rel_depth_mask>=(1+self.tolerance)
        rel_depth_mask_neg = rel_depth_mask<=(1/(1+self.tolerance))
        diff = (src_pred-tgt_pred)  #HxWxnum_comp
        sign = rel_depth_mask_pos.float()-rel_depth_mask_neg.float() # HxWx1
        depth_loss = torch.log(1+torch.exp(-sign[sign!=0]*diff[sign!=0]))
        depth_loss_sim = torch.pow(diff[sign==0],2)
        return depth_loss.mean()+depth_loss_sim.mean()


def get_depth_order_loss(render_depth,gt_depth,mask,method_name="sign_loss",pair_num = 200000,alpha=100,):
    """_summary_

    Args:
        render_depth (_type_): 1,H,W,
        gt_depth (_type_): H,W,
        mask (_type_): H,W,
        method_name (str, optional): _description_. Defaults to "pearson".

    Returns:
        _type_: _description_
    """
    if method_name=="vallina":
        
        raise NotImplementedError


    elif method_name=="SingleImageDepthPerceptionintheWild" or method_name=="DepthRanking": ## https://arxiv.org/pdf/1604.03901.pdf
        loss = 0.0
        threshold = (gt_depth.max()-gt_depth.min())/100 ##    10    ，       
        gt_depth=gt_depth[mask>0]## N,1
        render_depth=render_depth.squeeze(0)[mask>0]## N,1
        index1 = torch.randperm(gt_depth.shape[0])[:pair_num,].to(gt_depth.device)
        index2 = torch.randperm(gt_depth.shape[0])[:pair_num,].to(gt_depth.device)
        gt_depth_diff = gt_depth[index1]-gt_depth[index2]
        gt_oder = torch.sign(gt_depth_diff) ## 1 or -1
        gt_oder[gt_depth_diff.abs()<threshold]=0
        selected_i = render_depth[index1]
        selected_j = render_depth[index2]
        
        # loss += torch.mean(torch.clamp((selected_i-selected_j)[gt_oder==1],min=None,max=0))
        # loss += torch.mean(torch.clamp((selected_j-selected_i)[gt_oder==-1],min=None,max=0))
        pred_depth = render_depth[index1]-render_depth[index2]
        log_loss = torch.mean(torch.log(1 + torch.exp(-gt_oder[gt_oder != 0] * pred_depth[gt_oder != 0])))
        square_loss = torch.mean(torch.square(pred_depth[gt_oder == 0]))
        return log_loss+square_loss
        

    elif method_name=="tanh_threshold":
        # alpha = 100
        ###   Tanh          。
        gt_depth=gt_depth[mask>0]## N,1
        depthmax=gt_depth.max()
        depthmin=gt_depth.min()
        interval = (depthmax-depthmin)/10
        # interval = (depthmax-depthmin)/20
    
        render_depth=render_depth.squeeze(0)[mask>0]## N,1
        index1 = torch.randperm(gt_depth.shape[0])[:pair_num,].to(gt_depth.device)
        index2 = torch.randperm(gt_depth.shape[0])[:pair_num,].to(gt_depth.device)
        threshold_msk  = (torch.abs(gt_depth[index1]-gt_depth[index2])>=interval)
        index1 = index1[threshold_msk]
        index2 = index2[threshold_msk]
        
        gt_oder = torch.sign(gt_depth[index1]-gt_depth[index2])
        # render_oder = torch.sign(render_depth[index1]-render_depth[index2])
        render_diff = render_depth[index1]-render_depth[index2]

        loss = torch.mean(torch.abs(torch.tanh(alpha*render_diff)-gt_oder))
        return loss

        
    else:
        raise NotImplementedError(method_name)

    return loss