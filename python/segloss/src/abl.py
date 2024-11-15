import numpy
import torch

from scipy.ndimage import distance_transform_edt as distance
from torchvision   import transforms
from functools     import partial
from operator      import itemgetter


__all__ = ['ActiveBoundaryLoss']


def kl_div(a, b):
    """
    Calculates the Kullback-Leibler divergence between two probability distributions.

    Args:
        a (torch.Tensor): Tensor of shape (batch_size, num_classes) representing distribution q.
        b (torch.Tensor): Tensor of shape (batch_size, num_classes) representing distribution p.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, num_classes) containing the KL divergence values.
    """

    return torch.nn.functional.softmax(b, dim=1) * (torch.nn.functional.log_softmax(b, dim=1) - torch.nn.functional.log_softmax(a, dim=1))


def one_hot2dist(seg):
    """
    Converts a one-hot encoded segmentation mask to a distance map.

    Args:
        seg (numpy.ndarray): A one-hot encoded segmentation mask of shape (N, H, W, C) or (H, W, C).

    Returns:
        numpy.ndarray: A distance map of the same shape as the input segmentation mask.
    """

    res = numpy.zeros_like(seg)
    for i in range(len(seg)):
        posmask = seg[i].astype(bool)
        if posmask.any():
            negmask = ~posmask
            res[i] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res


def class2one_hot(seg, C):
    """
    Converts a class-based segmentation mask to a one-hot encoded mask.

    Args:
        seg (torch.Tensor): A class-based segmentation mask of shape (N, H, W) or (H, W).
        C (int): The number of classes.

    Returns:
        torch.Tensor: A one-hot encoded mask of shape (N, H, W, C) or (H, W, C).
    """

    seg = seg.unsqueeze(dim=0) if len(seg.shape) == 2 else seg  # Add batch dimension if needed
    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    return res


class ActiveBoundaryLoss(torch.nn.Module):
    """
    ActiveBoundaryLoss: A loss function that emphasizes segmentation boundaries for improved accuracy.

    It combines a standard cross-entropy loss on all pixels with a weighted cross-entropy loss on predicted boundary regions. 
    The boundary loss focuses on correctly classifying boundary pixels and accurately predicting their orientations.
    """
    def __init__(self, border_params=[0.1, 'cpu', True, 0.2, 0.01, 20.], ignore_index=255, weight=None):
        """
        Initializes the ActiveBoundaryLoss class.

        Args:
            border_params (list, optional): List of parameters controlling boundary loss behavior.
                - border_params[0]: border_factor (float): Weighting factor for the boundary loss component.
                - border_params[1]: device (str): Device to use for computations (e.g., 'cpu' or 'cuda').
                - border_params[2]: isdetach (bool): Whether to detach slices for KL divergence calculation.
                - border_params[3]: label_smoothing (float): Label smoothing factor for the border criterion.
                - border_params[4]: max_N_ratio (float): Maximum ratio of boundary pixels to consider.
                - border_params[5]: max_clip_dist (float): Maximum distance value for normalization.
                Default: [0.1, 'cpu', True, 0.2, 0.01, 20.].
            ignore_index (int, optional): Label index to ignore. Default: 255.
            weight (torch.Tensor, optional): Weight tensor for classes. Default: None.
        """
        super(ActiveBoundaryLoss, self).__init__()
        # Extract and store boundary loss parameters
        self.border_factor = float(border_params[0])
        self.device = str(border_params[1])
        self.isdetach = bool(border_params[2])
        self.label_smoothing = float(border_params[3])
        self.max_N_ratio = float(border_params[4])
        self.max_clip_dist = float(border_params[5])
        self.ignore_index = ignore_index
        # Function to normalize weights based on distance
        self.weight_func = lambda w, max_distance=self.max_clip_dist: torch.clamp(w, max=max_distance) / max_distance
        # Transform to generate distance maps from ground truth labels
        self.dist_map_transform = transforms.Compose([
            lambda img: img.unsqueeze(0),  # Add batch dimension
            lambda nd: nd.type(torch.int64),  # Convert to int64
            partial(class2one_hot, C=1),  # Create one-hot encoding
            itemgetter(0),  # Extract the first channel
            lambda t: t.cpu().numpy(),  # Move to CPU and convert to NumPy array
            one_hot2dist,  # Convert one-hot encoding to distance map
            lambda nd: torch.tensor(nd, dtype=torch.float32)  # Convert back to torch.Tensor
        ])
        # Loss criteria for pixel-wise paradigm
        self.target_criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            reduction='none',  # No reduction for element-wise loss
            weight=weight
        )
        # Loss criteria for boundary regions
        self.border_criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction='none',
            weight=weight
        )        

    def slices2boundary(self, logit, eps=1e-5):
        """
        Extracts boundary regions from segmentation predictions.

        Args:
            logit (torch.Tensor): Model's prediction tensor of shape (N, C, H, W).
            eps (float, optional): Threshold for boundary detection

        Returns:
            torch.Tensor: A binary tensor of shape (N, H, W) indicating boundary regions.
        """
        # -- Calculate KL divergence between neighboring pixels --
        _, _, h, w = logit.shape
        max_N = (h * w) * self.max_N_ratio  # Maximum number of boundary pixels
        kl_ud = kl_div(logit[:, :, 1:, :], logit[:, :, :-1, :]).sum(1, keepdim=True)  # Vertical KL divergence
        kl_lr = kl_div(logit[:, :, :, 1:], logit[:, :, :, :-1]).sum(1, keepdim=True)  # Horizontal KL divergence
        # -- Combine KL divergences and handle boundaries --
        kl_ud = torch.nn.functional.pad(kl_ud, [0, 0, 0, 1, 0, 0, 0, 0], mode='constant', value=0)
        kl_lr = torch.nn.functional.pad(kl_lr, [0, 1, 0, 0, 0, 0, 0, 0], mode='constant', value=0)
        kl_combine = kl_lr + kl_ud
        # -- Threshold and adjust for full-image cases --
        while True:
            kl_combine_bin = (kl_combine > eps).to(torch.float)
            if kl_combine_bin.sum() > max_N:
                eps *= 1.2  # Increase threshold if too many boundary pixels
            else:
                break
        # -- Dilate boundaries --
        dilate_weight = torch.ones((1, 1, 3, 3)).to(self.device)
        edge2 = torch.nn.functional.conv2d(kl_combine_bin, dilate_weight, stride=1, padding=1)
        kl_combine_bin = (edge2 > 0)  # Final binary boundary mask
        return kl_combine_bin

    def gtruth2boundary(self, gt, ignore_label=-1):  # gt NHW
        """
        Generates a binary boundary mask from ground truth labels.

        Args:
            gt (torch.Tensor): Ground truth labels of shape (N, H, W).
            ignore_label (int, optional): Label index to consider as boundary. Default: -1

        Returns:
            torch.Tensor: Binary boundary mask of shape (N, H, W).
        """
        # -- Calculate horizontal and vertical boundary differences --
        gt_ud = gt[:, :, 1:, :] - gt[:, :, :-1, :]  # NHW
        gt_lr = gt[:, :, :, 1:] - gt[:, :, :, :-1]
        # -- Pad tensors for boundary handling --
        gt_ud = torch.nn.functional.pad(gt_ud, [0, 0, 0, 1, 0, 0], mode='constant', value=0) != 0
        gt_lr = torch.nn.functional.pad(gt_lr, [0, 1, 0, 0, 0, 0], mode='constant', value=0) != 0
        # -- Combine horizontal and vertical boundaries --
        gt_combine = gt_lr + gt_ud
        del gt_lr
        del gt_ud
        # -- Include ignore areas as boundaries --
        gt_combine += (gt == ignore_label)
        # -- Return final binary boundary mask --
        return gt_combine > 0

    def get_orientation(self, pred_dist_map, pred_bound, slices):
        """
        Calculates the ground truth and predicted boundary orientations.

        Args:
            pred_dist_map (torch.Tensor): Predicted distance map of shape (N, 1, H, W).
            pred_bound (torch.Tensor): Predicted boundary mask of shape (N, 1, H, W).
            slices (torch.Tensor): Model predictions of shape (N, C, H, W).

        Returns:
            tuple: (direction_gt, direction_pred, weight_ce)
                - direction_gt (torch.Tensor): Ground truth boundary directions of shape (K).
                - direction_pred (torch.Tensor): Predicted boundary directions of shape (K, 8).
                - weight_ce (torch.Tensor): Weights for cross-entropy loss of shape (K).
        """
        # Get indices of predicted boundary pixels
        bound = torch.nonzero(pred_bound * 1)  # Get indices of non-zero elements
        n, c, x, y = bound.T

        # Pad tensors for boundary handling
        slices = slices.permute(0, 2, 3, 1)  # NHWC
        pred_dist_map_d = torch.nn.functional.pad(pred_dist_map, (1, 1, 1, 1, 0, 0), mode='constant', value=1e5) # NH+2W+2
        slices_d = torch.nn.functional.pad(slices, (0, 0, 1, 1, 1, 1, 0, 0), mode='constant') # N(H+2)(W+2)C
        slices_d[:, 0,:,:] = slices_d[:, 1,:,:] # N(H+2)(W+2)C
        slices_d[:,-1,:,:] = slices_d[:,-2,:,:] # N(H+2)(W+2)C
        slices_d[:,:, 0,:] = slices_d[:,:, 1,:] # N(H+2)(W+2)C
        slices_d[:,:,-1,:] = slices_d[:,:,-2,:] # N(H+2)(W+2)C
        
        """
        | 4| 0| 5|
        | 2| 8| 3|
        | 6| 1| 7|
        """
        # Calculate distances and KL divergences for neighboring pixels
        dist_maps = torch.zeros((0, len(x))).to(self.device)  # 8K
        kl_maps = torch.zeros((0, len(x))).to(self.device)  # 8K
        x_range = [1, -1, 0, 0, -1, 1, -1, 1, 0]
        y_range = [0, 0, -1, 1, 1, 1, -1, -1, 0]

        kl_center = slices[(n,x,y)] # KC

        for dx, dy in zip(x_range, y_range):
            dist_now = pred_dist_map_d[(n,c,x+dx+1,y+dy+1)]
            dist_maps = torch.cat((dist_maps,dist_now.unsqueeze(0)),0)
            if dx != 0 or dy != 0:
                slices_now = slices_d[(n,x+dx+1,y+dy+1)]
                if self.isdetach:
                    slices_now = slices_now.detach()
                kl_map_now = kl_div(kl_center, slices_now)
                kl_map_now = kl_map_now.sum(1)  # KC->K
                kl_maps = torch.cat((kl_maps,kl_map_now.unsqueeze(0)),0)
                torch.clamp(kl_maps, min=0.0, max=20.0)

        # Determine ground truth and predicted directions
        direction_gt = torch.argmin(dist_maps, dim=0)  # Find direction with minimum distance
        direction_gt_idx = [direction_gt != 8]  # Exclude local position (8)
        direction_gt = direction_gt[direction_gt_idx]

        weight_ce = pred_dist_map[(n, c, x, y)]  # Weights based on predicted distances
        weight_ce = weight_ce[direction_gt_idx]
        
        kl_maps = torch.transpose(kl_maps, 0, 1)  # K8
        direction_pred = kl_maps[direction_gt_idx]  # Predicted directions

        return direction_gt, direction_pred, weight_ce

    def get_dist_matrix(self, targets):
        """
        Generates distance maps from ground truth segmentation labels.

        Args:
            targets (torch.Tensor): Ground truth segmentation labels.

        Returns:
            torch.Tensor: A tensor of distance maps.
        """
        # Detach targets to avoid gradients during distance map calculation
        targets_detach = targets.clone().detach()
        # Generate distance maps for each batch element
        dist_maps = torch.cat([self.dist_map_transform(targets_detach[i]) for i in range(targets_detach.shape[0])])
        # Invert distances and ensure non-negative values (0 for boundaries)
        out = -dist_maps
        out = torch.where(out > 0, out, torch.zeros_like(out))
        return out

    def forward(self, slices, targets):
        """
        Calculates the ActiveBoundaryLoss.

        Args:
            slices (torch.Tensor): Model's prediction tensor of shape (N, C, H, W).
            targets (torch.Tensor): Ground truth segmentation labels of shape (N, H, W).

        Returns:
            torch.Tensor: The calculated ActiveBoundaryLoss value.
        """
        # -- Resize predictions if dimensions don't match --
        ph, pw = slices.size(2), slices.size(3)
        h, w = targets.size(2), targets.size(3)
        if ph != h or pw != w:
            slices = torch.nn.functional.interpolate(input=slices, size=(h, w), mode='bilinear', align_corners=True)
        # -- Generate boundary masks --
        pred_boundary = self.slices2boundary(slices)
        # -- Calculate boundary loss if predicted boundary pixels exist --
        if pred_boundary.sum() > 1:  # Avoid NaN
            gt_boundary = self.gtruth2boundary(targets, ignore_label=self.ignore_index)
            dist_maps = self.get_dist_matrix(gt_boundary).to(self.device)  # Note: Potential performance impact
            direction_gt, direction_pred, weight_ce = self.get_orientation(dist_maps, pred_boundary, slices)
            border_loss = (self.border_criterion(direction_pred, direction_gt).sum() *
                        self.weight_func(weight_ce).sum())
        else:
            border_loss = 0
        # -- Calculate standard cross-entropy loss on all pixels --
        target_loss = self.target_criterion(slices, torch.squeeze(targets, dim=1)).sum()
        # -- Combine boundary loss and standard loss --
        return (self.border_factor * border_loss) + target_loss
    

if __name__ == '__main__':
    torch.manual_seed(2022)

    loss  =  ActiveBoundaryLoss(border_params=[0.1,'cpu',True,0.2,1/100,20.])
    image = torch.rand([1, 3, 14, 14])
    label = torch.zeros([1, 1, 14, 14], dtype=torch.long)
    
    for idy in range(image.shape[2]):
        for idx in range(image.shape[3]):
            if idy > idx+1: label[:,:,idy,idx] = 1
            if idy < idx-1: label[:,:,idy,idx] = 2
    print(image)
    print(label)

    print(loss(image,label))