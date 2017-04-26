import numpy as np


def binarise_saliency_map(saliency_map, method='adaptive', threshold=0.5):
    # check if input is a numpy array
    if type(saliency_map).__module__ != np.__name__:
        raise ValueError('Expected numpy array')

    # check if input is 2-D
    if len(saliency_map.shape) != 2:
        print('Saliency map must be 2-D')
        return None

    if 'fixed' == method:
        return saliency_map > threshold

    elif 'adaptive' == method:
        adaptive_threshold = 2.0 * saliency_map.mean()
        return saliency_map > adaptive_threshold

    elif 'clustering' == method:
        NotImplementedError('Clustering is not yet implemented')
        return None

    else:
        raise NotImplementedError('Method must be one of fixed, adaptive or clustering')
