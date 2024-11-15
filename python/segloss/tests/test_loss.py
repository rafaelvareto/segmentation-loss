import torch

torch.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


import segloss


def test_abl():
    image = torch.rand([1, 3, 64, 64])
    label = torch.zeros([1, 1, 64, 64], dtype=torch.long)
    
    for idy in range(image.shape[2]):
        for idx in range(image.shape[3]):
            if idy > idx+1: label[:,:,idy,idx] = 1
            if idy < idx-1: label[:,:,idy,idx] = 2

    criterion  = segloss.ActiveBoundaryLoss(border_params=[0.1,'cpu',True,0.2,1/100,20.])
    loss_score = criterion(image, label)
    loss_score.backward()

    assert round(loss_score.item(), 3) == None # Replace value


def test_bfl():
    image = torch.rand([1, 3, 64, 64])
    label = torch.zeros([1, 1, 64, 64], dtype=torch.long)
    
    for idy in range(image.shape[2]):
        for idx in range(image.shape[3]):
            if idy > idx+1: label[:,:,idy,idx] = 1
            if idy < idx-1: label[:,:,idy,idx] = 2

    criterion  = segloss.BoundaryScoreLoss(border_params=[3,5,0.5], reduction='sum')
    loss_score = criterion(image, label)
    loss_score.backward()

    assert round(loss_score.item(), 3) == None # Replace value


def test_cel():
    image = torch.rand([1, 3, 64, 64])
    label = torch.zeros([1, 1, 64, 64], dtype=torch.long)
    
    for idy in range(image.shape[2]):
        for idx in range(image.shape[3]):
            if idy > idx+1: label[:,:,idy,idx] = 1
            if idy < idx-1: label[:,:,idy,idx] = 2

    criterion  = segloss.CrossEntropyLoss(reduction='sum')
    loss_score = criterion(image, label)
    loss_score.backward()

    assert round(loss_score.item(), 3) == None # Replace value



def test_dtl():
    image = torch.rand([1, 3, 64, 64])
    label = torch.zeros([1, 1, 64, 64], dtype=torch.long)
    
    for idy in range(image.shape[2]):
        for idx in range(image.shape[3]):
            if idy > idx+1: label[:,:,idy,idx] = 1
            if idy < idx-1: label[:,:,idy,idx] = 2

    criterion  = segloss.DistanceTransformLoss(border_params=[1,0.5,1.0])
    loss_score = criterion(image, label)
    loss_score.backward()

    assert round(loss_score.item(), 3) == None # Replace value
