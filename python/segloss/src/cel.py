import torch


__all__ = ['CrossEntropyLoss']


class CrossEntropyLoss:
    """
    A customized CrossEntropyLoss class with optional label smoothing and flexible reduction options.
    """
    def __init__(self, border_params=[0.0], ignore_index=255, reduction='none', weight=None):
        """
        Initializes the CrossEntropyLoss class.

        Args:
            border_params (list, optional): List containing label smoothing factor (default: [0.0]).
            ignore_index (int, optional): Label index to ignore (default: 255).
            reduction (str, optional): Reduction mode for the loss (default: 'none').
            weight (torch.Tensor, optional): Weights for each class (default: None).
        """
        self.label_smoothing = float(border_params[0])
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.weight = weight

    def __call__(self, slices, targets):
        """
        Calculates the CrossEntropyLoss.

        Args:
            slices (torch.Tensor): Model predictions of shape (N, C, H, W).
            targets (torch.Tensor): Ground truth segmentation labels of shape (N, H, W) or (N, 1, H, W).

        Returns:
            torch.Tensor: The calculated CrossEntropyLoss value.
        """
        if len(targets.shape) > 3:
            targets = torch.squeeze(targets, dim=1)  # Handle extra dimension in targets

        return torch.nn.functional.cross_entropy(
            slices,
            targets,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
            weight=self.weight
        )


if __name__ == '__main__':
    torch.manual_seed(2022)

    loss =  CrossEntropyLoss(reduction='sum')
    image = torch.rand([1, 3, 14, 14])
    label = torch.zeros([1, 1, 14, 14], dtype=torch.long)
    
    for idy in range(image.shape[2]):
        for idx in range(image.shape[3]):
            if idy > idx+1: label[:,:,idy,idx] = 1
            if idy < idx-1: label[:,:,idy,idx] = 2
    print(image)
    print(label)

    print(loss(image,label))
