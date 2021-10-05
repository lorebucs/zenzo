import torch
from torch.distributions import uniform

def binary_classification_dataset(f, n_samples, scale=1):
    """Make a 2D dataset for binary classification.
    
    The input function ``f`` is used to split the 2D input space into 2 regions. 
    The input function ``f`` constitutes the actual boundary between classes.
    """
    X = torch.rand(n_samples, 2)
    y = [0 if y<f(x) else 1 for x,y in X]
    y = torch.tensor(y).float().unsqueeze(1)
    X = (scale * X) - torch.tensor([scale / 2, scale / 2])
    return X, y