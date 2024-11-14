import torch


class CrossEntropyLoss:
    def __init__(self, ignore_index=255, label_smoothing=0.0, reduction='none', weight=None):
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.weight = weight
        

    def __call__(self, slices, targets):
        if len(targets.shape) > 3:
            targets = torch.squeeze(targets, dim=1)
        return torch.nn.functional.cross_entropy(slices, targets, ignore_index=self.ignore_index, label_smoothing=self.label_smoothing, reduction=self.reduction, weight=self.weight)
