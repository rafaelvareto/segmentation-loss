import torch


def one_hot(label, n_classes, requires_grad=True):
    """
    Return One Hot Label
    """
    one_hot_label = torch.eye(
        n_classes, 
        device=label.device, 
        requires_grad=requires_grad
    )[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label


class BoundaryScoreLoss(torch.nn.Module):
    """
    Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, border_params=[3,5,0.3], ignore_index=255, reduction='mean', weight=None):
        super(BoundaryScoreLoss, self).__init__()
        self.border_theta0 = border_params[0]
        self.border_theta  = border_params[1]
        self.border_weight = border_params[2]

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
            - slices: the output from model (before softmax)
                    shape (N, C, H, W)
            - targets: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, c, _, _ = slices.shape
        
        # handling unwanted indexes and extra dimension
        targets = torch.squeeze(targets, dim=1)
        targets[targets==self.ignore_index] = 0

        # softmax so that predicted map can be distributed in [0, 1]
        slices = torch.softmax(slices, dim=1)

        # one-hot vector of ground truth
        one_hot_gt = one_hot(targets, c)

        # boundary map
        gt_b = torch.nn.functional.max_pool2d(
            1 - one_hot_gt, 
            kernel_size=self.border_theta0, 
            stride=1, 
            padding=(self.border_theta0 - 1) // 2
        )
        gt_b -= 1 - one_hot_gt

        pred_b = torch.nn.functional.max_pool2d(
            1 - slices, 
            kernel_size=self.border_theta0, 
            stride=1, 
            padding=(self.border_theta0 - 1) // 2
        )
        pred_b -= 1 - slices

        # extended boundary map
        gt_b_ext = torch.nn.functional.max_pool2d(
            gt_b, 
            kernel_size=self.border_theta, 
            stride=1, 
            padding=(self.border_theta - 1) // 2
        )

        pred_b_ext = torch.nn.functional.max_pool2d(
            pred_b, 
            kernel_size=self.border_theta, 
            stride=1, 
            padding=(self.border_theta - 1) // 2
        )

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        border_loss = torch.mean(1 - BF1)
        target_loss = self.target_criterion(slices, targets)

        return target_loss + (self.border_weight * border_loss)


# for debug
if __name__ == "__main__":
    from torchvision.models import segmentation

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img = torch.randn(8, 3, 224, 224).to(device)
    gt = torch.randint(0, 10, (8, 224, 224)).to(device)

    print(img.shape, gt.shape)

    model = segmentation.fcn_resnet50(num_classes=10).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = BoundaryScoreLoss()

    y = model(img)

    loss = criterion(y['out'], gt)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss)
