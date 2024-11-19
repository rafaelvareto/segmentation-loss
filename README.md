# SEGLOSS: Semantic Segmentation Loss Functions

This project encompasses a list of three cost functions designed for semantic segmentation, also available through the **SegLoss Package** [[PyPi link](https://pypi.org/project/segloss/)] so that it can be easily installed in your environment:

* ```segloss.ActiveBoundaryLoss(border_params=[0.1, 'cpu', True, 0.2, 0.01, 20.], ignore_index=255, weight=None)```
* ```segloss.BoundaryLoss(border_params=[3, 5, 0.5], ignore_index=255, reduction='mean', weight=None)``` 
* ```segloss.DistanceTransformLoss(border_params=[5,0.5,1.0], ignore_index=255, reduction='sum', weight=None)```

Our most recent contribution, the Distance Transform Loss (DTL), punishes deep networks when class boundaries are misclassified in exchange for more accurate contour delineations, an important aspect in the geological field.
DTL consists of four key steps: contour detection, distance transform mapping, pixel-wise multiplication, and the summation of all grid elements.

The proposed functions' API models the conventional cost functions available under the PyTorch framework so that it can be invoked according to the code block below:
```python
    import segloss
    import torch

    # Generate 12 samples randomly
    num_classes, num_samples, num_dims = 4, 12, 64
    images = torch.rand([num_samples, 3, num_dims, num_dims])
    labels = torch.randint(0, num_classes, (num_samples, num_dims, num_dims))

    # Feed criterion with both images and labels and compute loss
    criterion = segloss.DistanceTransformLoss(border_params=[1,0.5,1.0])
    loss_score = criterion(images, labels)
    loss_score.backward()
```

There is a notebook tutorial under construction, being implemented with the ```PyTorch``` framework as well as the ```SEGYIO``` library demonstrating how to use DTL in semantic segmentation applications. 
The tutorial showcases an experiment with the Netherlands' F3 Block volume, a publicly available seismic dataset that has been annotated by Yalaudah for the purpose of machine learning-based facies classification. This dataset is considered a valuable resource for researchers working in the field of geophysical data analysis and machine learning.
If you have further questions about the proposed approach, you can reach the corresponding author though the contact form available at [www.vareto.com.br](https://vareto.com.br/index.html#contact).
