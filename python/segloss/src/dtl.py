import torch


__all__ = ['DistanceTransformLoss']


class DistanceTransformLoss(torch.nn.Module):
    """
    DistanceTransformLoss: A loss function that emphasizes boundary segmentation by
    combining a standard cross-entropy loss with a weighted loss based on the distance
    transform of boundary pixels.

    The distance transform loss focuses on correctly classifying pixels near boundaries
    by assigning higher penalties to misclassified pixels closer to ground truth boundaries.
    """    
    def __init__(self, border_params=[5,0.5,1.0], ignore_index=255, reduction='sum', weight=None):
        """
        Initializes the DistanceTransformLoss class.

        Args:
            border_params (list, optional): List of parameters for border loss configuration:
                - border_params[0]: Dilation factor for distance transform (default: 5)
                - border_params[1]: Power to raise the border loss (default: 0.5)
                - border_params[2]: Weighting factor for the border loss (default: 1.0)
            ignore_index (int, optional): Label index to ignore (default: 255)
            reduction (str, optional): Reduction mode for the loss (default: 'sum')
            weight (torch.Tensor, optional): Weights for each class (default: None)
        """
        super(DistanceTransformLoss, self).__init__()
        self.border_dilate =   int(border_params[0])
        self.border_power  = float(border_params[1])
        self.border_weight = float(border_params[2])
        
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.weight = weight
        
        self.target_criterion = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction, 
            weight=weight
        )

    def get_class_boundaries(self, matrix, target_id=None):
        """
        Creates a one-hot encoded boundary mask for the specified class in the given matrix.

        Args:
            matrix (torch.Tensor): Input tensor of shape (N, C, H, W).
            target_id (int, optional): ID of the class for which to create boundaries.
                                       If None, boundaries for all classes are generated.

        Returns:
            torch.Tensor: One-hot encoded boundary mask of shape (N, 1, H, W).
        """
        matrix_ = (matrix == target_id)
        border_tb = matrix_[:,:,1:,:] ^ matrix_[:,:,:-1,:] 
        border_lr = matrix_[:,:,:,1:] ^ matrix_[:,:,:,:-1]
        border_tb  = torch.nn.functional.pad(input=border_tb, pad=[0,0,0,1,0,0], mode='constant', value=0) 
        border_lr  = torch.nn.functional.pad(input=border_lr, pad=[1,0,0,0,0,0], mode='constant', value=0)
        border_onehot = (border_tb + border_lr) != 0
        return border_onehot

    def get_distance_matrix(self, matrix, dist_transform=True):
        """
        Calculates the distance transform of a boundary mask.

        Args:
            matrix (torch.Tensor): Input tensor of shape (N, C, H, W) containing boundary mask.
            dist_transform (bool, optional): If True, performs distance transform.
                                             If False, returns the original matrix.

        Returns:
            torch.Tensor: Distance transform of the input matrix.
        """
        _, _, height, width = matrix.shape
        # Initialize distance matrix with max penalty
        penalty = torch.tensor(matrix.shape).sum()
        dist_matrix = torch.ones_like(matrix).float() * penalty
        # Replace border values with zero
        pos_indices = torch.nonzero(matrix == 1, as_tuple=True)
        dist_matrix[pos_indices] = 0
        # Iterate over matrix rows to create distance mapping
        for index in range(1, height): 
            dist_matrix[:,:,index,:] = torch.min(dist_matrix[:,:,index,:], dist_matrix[:,:,index-1,:]+1)
        for index in reversed(range(0, height-1)): 
            dist_matrix[:,:,index,:] = torch.min(dist_matrix[:,:,index,:], dist_matrix[:,:,index+1,:]+1)
        # Iterate over matrix columns to create distance mapping
        for index in range(1, width): 
            dist_matrix[:,:,:,index] = torch.min(dist_matrix[:,:,:,index], dist_matrix[:,:,:,index-1]+1)
        for index in reversed(range(0, width-1)): 
            dist_matrix[:,:,:,index] = torch.min(dist_matrix[:,:,:,index], dist_matrix[:,:,:,index+1]+1)
        dist_matrix = torch.nn.functional.relu(dist_matrix - self.border_dilate)
        return dist_matrix

    def border_criterion(self, slices, targets):
        """
        Calculates the border loss based on distance transform.

        Args:
            slices (torch.Tensor): Model predictions of shape (N, C, H, W).
            targets (torch.Tensor): Ground truth segmentation labels of shape (N, H, W).

        Returns:
            torch.Tensor: The calculated border loss value.
        """
        border_penalty = 0.
        slices_indices = torch.argmax(slices, dim=1, keepdim=True)
        slices_softmax = torch.nn.functional.softmax(slices, dim=1)
        # Perform multi-class border penalization
        for target_id in torch.unique(targets):
            if target_id != self.ignore_index:
                slices_borders  = self.get_class_boundaries(slices_indices, target_id)
                target_borders  = self.get_class_boundaries(targets, target_id)
                target_borders  = self.get_distance_matrix(target_borders, dist_transform=True)
                border_penalty += slices_softmax[:,target_id,:,:] * slices_borders * target_borders
        # Ruduce penalty though mean or sum of all batch samples
        if   self.reduction == 'mean': return border_penalty[border_penalty != 0].mean()
        elif self.reduction ==  'sum': return border_penalty[border_penalty != 0].sum()
        else:                          return border_penalty[border_penalty != 0]
         
    def forward(self, slices, targets):
        """
        Calculates the DistanceTransformLoss.

        Args:
            slices (torch.Tensor): Model predictions of shape (N, C, H, W).
            targets (torch.Tensor): Ground truth segmentation labels of shape (N, H, W).

        Returns:
            torch.Tensor: The calculated DistanceTransformLoss value.
        """
        target_loss = self.target_criterion(slices, torch.squeeze(targets, dim=1))
        border_loss = self.border_criterion(slices, targets)
        if (self.reduction is None) or (self.reduction == 'none'):
            return (target_loss, border_loss)
        return target_loss + (self.border_weight * border_loss**self.border_power)


if __name__ == '__main__':
    torch.manual_seed (2022)

    loss =  DistanceTransformLoss(border_params=[1,0.5,1.0])
    image = torch.rand([1, 3, 14, 14])
    label = torch.zeros([1, 1, 14, 14], dtype=torch.long)
    
    for idy in range(image.shape[2]):
        for idx in range(image.shape[3]):
            if idy > idx+1: label[:,:,idy,idx] = 1
            if idy < idx-1: label[:,:,idy,idx] = 2
    print(image)
    print(label)

    print(loss(image,label))
