import numpy as np

def create_dataset(func, x_range, n_samples, mu, sigma, equispaced=True):
    """Creates ``n_samples`` samples using function ``func``
    and adding noise from Gaussian with mean ``mu`` and 
    variance ``sigma``.
    """
    if equispaced:
        xs = np.linspace(x_range[0], x_range[1], n_samples)
    else:
        xs = np.random.uniform(x_range[0], x_range[1], size=n_samples)
    ts = func(xs) + np.random.normal(mu, sigma, n_samples)
    return xs, ts

def augment_dataset(xs, deg):
    """Augments monodimentional list of samples.
    """
    assert(xs.ndim == 1)    
    xs = np.expand_dims(xs, 1)
    xs_augmented = np.ones(xs.shape)
    for i in range(1, deg+1):
        xs_augmented = np.hstack((xs_augmented, xs**i))
    return xs_augmented

def polynomial(x, coeffs):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x**0`` to ``x**deg``).
    """
    y = 0
    for i in range(len(coeffs)):
        y += coeffs[i]*x**i
    return y