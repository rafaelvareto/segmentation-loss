import numpy
import torch

from scipy.ndimage import distance_transform_edt as distance
from torchvision import transforms
from functools import partial
from operator import itemgetter

from .cross_entropy import CrossEntropyLoss
from .label_smooth import LabelSmoothSoftmaxCEV1 as LSSCE


# Tools
def kl_div(a,b): # q,p
    return torch.nn.functional.softmax(b, dim=1) * (torch.nn.functional.log_softmax(b, dim=1) - torch.nn.functional.log_softmax(a, dim=1))   

def one_hot2dist(seg):
    res = numpy.zeros_like(seg)
    for i in range(len(seg)):
        posmask = seg[i].astype(numpy.bool)
        if posmask.any():
            negmask = ~posmask
            res[i] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res

def class2one_hot(seg, C):
    seg = seg.unsqueeze(dim=0) if len(seg.shape) == 2 else seg
    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    return res


class ActiveBoundaryLoss(torch.nn.Module):
    """
    Active Boundary Loss for Semantic Segmentation 
    Paper:  https://ojs.aaai.org/index.php/AAAI/article/view/20139
    GitHub: https://github.com/wangchi95/active-boundary-loss
    ****** AUTHORS' ORIGINAL IMPLEMENTATION ****** 
    """
    def __init__(self, border_factor=0.1, device='cpu', isdetach=True, ignore_label=255, label_smoothing=0.2, max_N_ratio=1/100, weight=None, max_clip_dist=20.):
        super(ActiveBoundaryLoss, self).__init__()
        self.border_factor = border_factor
        self.device = device
        self.ignore_label = ignore_label
        self.label_smoothing = label_smoothing
        self.isdetach=isdetach
        self.max_N_ratio = max_N_ratio

        self.weight_func = lambda w, max_distance=max_clip_dist: torch.clamp(w, max=max_distance) / max_distance

        self.dist_map_transform = transforms.Compose([
            lambda img: img.unsqueeze(0),
            lambda nd: nd.type(torch.int64),
            partial(class2one_hot, C=1),
            itemgetter(0),
            lambda t: t.cpu().numpy(),
            one_hot2dist,
            lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])

        self.target_criterion = CrossEntropyLoss(
                weight=weight,
                ignore_index=ignore_label,
                reduction='none'
            )

        if label_smoothing == 0:
            self.border_criterion = CrossEntropyLoss(
                weight=weight,
                ignore_index=ignore_label,
                reduction='none'
            )
        else:
            self.border_criterion = LSSCE(
                reduction='none',
                ignore_index=ignore_label,
                lb_smooth = label_smoothing
            )

    def slices2boundary(self, logit):
        eps = 1e-5
        _, _, h, w = logit.shape
        max_N = (h*w) * self.max_N_ratio
        kl_ud = kl_div(logit[:, :, 1:, :], logit[:, :, :-1, :]).sum(1, keepdim=True)
        kl_lr = kl_div(logit[:, :, :, 1:], logit[:, :, :, :-1]).sum(1, keepdim=True)
        kl_ud = torch.nn.functional.pad(
            kl_ud, [0, 0, 0, 1, 0, 0, 0, 0], mode='constant', value=0)
        kl_lr = torch.nn.functional.pad(
            kl_lr, [0, 1, 0, 0, 0, 0, 0, 0], mode='constant', value=0)
        kl_combine = kl_lr+kl_ud
        while True: # avoid the case that full image is the same color
            kl_combine_bin = (kl_combine > eps).to(torch.float)
            if kl_combine_bin.sum() > max_N:
                eps *=1.2
            else:
                break
        #dilate
        dilate_weight = torch.ones((1,1,3,3)).to(self.device)
        edge2 = torch.nn.functional.conv2d(kl_combine_bin, dilate_weight, stride=1, padding=1)
        #edge2 = edge2.squeeze(1)  # NCHW->NHW
        kl_combine_bin = (edge2 > 0)
        return kl_combine_bin

    def gtruth2boundary(self, gt, ignore_label=-1):  # gt NHW
        gt_ud = gt[:,:,1:,:]-gt[:,:,:-1,:]  # NHW
        gt_lr = gt[:,:,:,1:]-gt[:,:,:,:-1]
        gt_ud = torch.nn.functional.pad(gt_ud, [0,0,0,1,0,0], mode='constant', value=0) != 0 
        gt_lr = torch.nn.functional.pad(gt_lr, [0,1,0,0,0,0], mode='constant', value=0) != 0
        gt_combine = gt_lr+gt_ud
        del gt_lr
        del gt_ud
        
        # set 'ignore area' to all boundary
        gt_combine += (gt==ignore_label)
        
        return gt_combine > 0

    def get_orientation(self, pred_dist_map, pred_bound, slices):
        # NHW,NHW,NCHW
        eps = 1e-5
        # bound = torch.where(pred_bound)  # 3k
        bound = torch.nonzero(pred_bound*1)
        n,c,x,y = bound.T
        max_dis = 1e5

        slices = slices.permute(0,2,3,1) # NHWC

        pred_dist_map_d = torch.nn.functional.pad(pred_dist_map,(1,1,1,1,0,0),mode='constant', value=max_dis) # NH+2W+2

        slices_d = torch.nn.functional.pad(slices,(0,0,1,1,1,1,0,0),mode='constant') # N(H+2)(W+2)C
        slices_d[:,0,:,:] = slices_d[:,1,:,:] # N(H+2)(W+2)C
        slices_d[:,-1,:,:] = slices_d[:,-2,:,:] # N(H+2)(W+2)C
        slices_d[:,:,0,:] = slices_d[:,:,1,:] # N(H+2)(W+2)C
        slices_d[:,:,-1,:] = slices_d[:,:,-2,:] # N(H+2)(W+2)C
        
        """
        | 4| 0| 5|
        | 2| 8| 3|
        | 6| 1| 7|
        """
        x_range = [1, -1,  0, 0, -1,  1, -1,  1, 0]
        y_range = [0,  0, -1, 1,  1,  1, -1, -1, 0]
        dist_maps = torch.zeros((0,len(x))).to(self.device) # 8k
        kl_maps = torch.zeros((0,len(x))).to(self.device) # 8k

        kl_center = slices[(n,x,y)] # KC

        for dx, dy in zip(x_range, y_range):
            dist_now = pred_dist_map_d[(n,c,x+dx+1,y+dy+1)]
            dist_maps = torch.cat((dist_maps,dist_now.unsqueeze(0)),0)

            if dx != 0 or dy != 0:
                slices_now = slices_d[(n,x+dx+1,y+dy+1)]
                # kl_map_now = torch.kl_div((kl_center+eps).log(), slices_now+eps).sum(2)  # 8KC->8K
                if self.isdetach:
                    slices_now = slices_now.detach()
                kl_map_now = kl_div(kl_center, slices_now)
                
                kl_map_now = kl_map_now.sum(1)  # KC->K
                kl_maps = torch.cat((kl_maps,kl_map_now.unsqueeze(0)),0)
                torch.clamp(kl_maps, min=0.0, max=20.0)

        # direction_gt shound be Nk  (8k->K)
        direction_gt = torch.argmin(dist_maps, dim=0)
        # weight_ce = pred_dist_map[bound]
        weight_ce = pred_dist_map[(n,c,x,y)]
        # print(weight_ce)

        # delete if min is 8 (local position)
        direction_gt_idx = [direction_gt!=8]
        direction_gt = direction_gt[direction_gt_idx]


        kl_maps = torch.transpose(kl_maps,0,1)
        direction_pred = kl_maps[direction_gt_idx]
        weight_ce = weight_ce[direction_gt_idx]

        return direction_gt, direction_pred, weight_ce

    def get_dist_matrix(self, targets):
        targets_detach = targets.clone().detach()
        dist_maps = torch.cat([self.dist_map_transform(targets_detach[i]) for i in range(targets_detach.shape[0])])
        out = -dist_maps
        out = torch.where(out>0, out, torch.zeros_like(out))
        
        return out

    def forward(self, slices, targets):
        eps = 1e-10
        ph, pw = slices.size(2), slices.size(3)
        h, w = targets.size(2), targets.size(3)

        if ph != h or pw != w:
            slices = torch.nn.functional.interpolate(input=slices, size=(h, w), mode='bilinear', align_corners=True)

        pred_boundary = self.slices2boundary(slices)

        if pred_boundary.sum() > 1: # avoid nan
            gt_boundary = self.gtruth2boundary(targets, ignore_label=self.ignore_label)
            dist_maps = self.get_dist_matrix(gt_boundary).to(self.device) # <-- it will slow down the training, you can put it to dataloader.
            direction_gt, direction_pred, weight_ce = self.get_orientation(dist_maps, pred_boundary, slices) # NHW,NHW,NCHW

            border_loss = self.border_criterion(direction_pred, direction_gt).sum() * self.weight_func(weight_ce).sum()  # direction_pred [K,8], direction_gt [K]
        else:
            border_loss = 0
        
        target_loss = self.target_criterion(slices, torch.squeeze(targets,dim=1)).sum()
    
        return (self.border_factor * border_loss) + target_loss


if __name__ == '__main__':
    from torch.backends import cudnn
    import os
    import random
    cudnn.benchmark = False
    cudnn.deterministic = True

    device='cpu'
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    n,c,h,w = 1,2,100,100
    gt = torch.zeros((n,h,w)).to(device)
    gt[0,5] = 1
    gt[0,50] = 1
    slices = torch.randn((n,c,h,w)).to(device)

    abl = ActiveBoundaryLossO(device='cpu')
    print(abl(slices, gt))