import torch


def one_hot(label, n_classes, requires_grad=True):
    """
    Converts a label tensor to a one-hot encoded tensor.

    Args:
        label (torch.Tensor): Long tensor of shape (N, H, W) or (N, 1, H, W) containing integer class labels.
        n_classes (int): Number of classes.
        requires_grad (bool, optional): Whether the resulting one-hot tensor should have gradients enabled. Default: True.

    Returns:
        torch.Tensor: One-hot encoded tensor of shape (N, C, H, W) with C = n_classes.
    """
    # Create a one-hot encoding matrix using torch.eye
    one_hot_label = torch.eye(
        n_classes, 
        device=label.device, 
        requires_grad=requires_grad
    )[label]
    # Reshape the tensor to match the expected output shape
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)
    return one_hot_label


class BoundaryScoreLoss(torch.nn.Module):
    """
    BoundaryScoreLoss: A loss function that emphasizes segmentation boundaries by 
    combining a standard cross-entropy loss with a weighted boundary loss.

    The boundary loss focuses on correctly classifying boundary pixels,
    using a weighted cross-entropy loss based on the distance to the nearest
    ground truth boundary.
    """
    def __init__(self, border_params=[3, 5, 0.5], ignore_index=255, reduction='mean', weight=None):
        """
        Initializes the BoundaryScoreLoss class.

        Args:
            border_params (list, optional): List of parameters for boundary loss configuration. Default: [3, 5, 0.5]
            ignore_index (int, optional): Label index to ignore. Default: 255
            reduction (str, optional): Reduction mode for the loss. Default: 'mean'
            weight (torch.Tensor, optional): Weights for each class. Default: None
        """
        super(BoundaryScoreLoss, self).__init__()
        self.border_theta0 = int(border_params[0])
        self.border_theta = int(border_params[1])
        self.border_weight = float(border_params[2])

        self.ignore_index = ignore_index
        self.reduction = reduction
        self.weight = weight

        self.target_criterion = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction,
            weight=weight
        )

    def forward(self, slices, targets):
        """
        Input:
            - slices: the output from model (before softmax) shape (N, C, H, W)
            - targets: ground truth map shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """
        n, c, _, _ = slices.shape
        # Handle ignored indexes and extra dimension
        targets = torch.squeeze(targets, dim=1)
        targets[targets==self.ignore_index] = 0
        # Apply softmax to predictions for probability distribution and Create one-hot encoded ground truth
        slices = torch.softmax(slices, dim=1)
        one_hot_gt = one_hot(targets, c)
        # Create boundary maps for ground-truth and predictions
        gt_b = torch.nn.functional.max_pool2d(1 - one_hot_gt, kernel_size=self.border_theta0, stride=1, padding=(self.border_theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt
        pred_b = torch.nn.functional.max_pool2d(1 - slices, kernel_size=self.border_theta0, stride=1, padding=(self.border_theta0 - 1) // 2)
        pred_b -= 1 - slices
        # Create extended boundary maps for ground-truth and predictions
        gt_b_ext = torch.nn.functional.max_pool2d(gt_b, kernel_size=self.border_theta, stride=1, padding=(self.border_theta - 1) // 2)
        pred_b_ext = torch.nn.functional.max_pool2d(pred_b, kernel_size=self.border_theta, stride=1, padding=(self.border_theta - 1) // 2)
        # Reshape for precision and recall calculations
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)
        # Calculate precision and recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)
        # Calculate Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)
        # Combine boundary loss and standard cross-entropy loss
        border_loss = torch.mean(1 - BF1)
        target_loss = self.target_criterion(slices, targets)
        return target_loss + (self.border_weight * border_loss)


if __name__ == '__main__':
    torch.manual_seed(2022)

    loss =  BoundaryScoreLoss(border_params=[3,5,0.5], reduction='sum')
    image = torch.rand([1, 3, 14, 14])
    label = torch.zeros([1, 1, 14, 14], dtype=torch.long)
    
    for idy in range(image.shape[2]):
        for idx in range(image.shape[3]):
            if idy > idx+1: label[:,:,idy,idx] = 1
            if idy < idx-1: label[:,:,idy,idx] = 2
    print(image)
    print(label)

    print(loss(image,label))
