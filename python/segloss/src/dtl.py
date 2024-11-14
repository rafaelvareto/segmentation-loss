import torch


class DistanceTransformLoss(torch.nn.Module):
    def __init__(self, border_params=[5,0.5,1.0], ignore_index=255, reduction='sum', weight=None):
        super(DistanceTransformLoss, self).__init__()
        self.border_dilate = border_params[0]
        self.border_power  = border_params[1]
        self.border_weight = border_params[2]
        
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.weight = weight
        
        self.target_criterion = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction, 
            weight=weight
        )

    def get_class_boundaries(self, matrix, target_id=None):
        matrix_ = (matrix == target_id)
        border_tb = matrix_[:,:,1:,:] ^ matrix_[:,:,:-1,:] 
        border_lr = matrix_[:,:,:,1:] ^ matrix_[:,:,:,:-1]
        border_tb  = torch.nn.functional.pad(input=border_tb, pad=[0,0,0,1,0,0], mode='constant', value=0) 
        border_lr  = torch.nn.functional.pad(input=border_lr, pad=[1,0,0,0,0,0], mode='constant', value=0)
        border_onehot = (border_tb + border_lr) != 0
        return border_onehot

    def get_distance_matrix(self, matrix, dist_transform=True):
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
        target_loss = self.target_criterion(slices, torch.squeeze(targets, dim=1))
        border_loss = self.border_criterion(slices, targets)
        if (self.reduction is None) or (self.reduction == 'none'):
            return (target_loss, border_loss)
        # print(target_loss.item(), 'border:', border_loss.item(), (self.border_weight * border_loss**self.border_power))
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
