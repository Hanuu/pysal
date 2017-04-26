import math
# import sys
# import operator
# import networkx as nx
# import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from skimage.io import imread as skimage_imread
from skimage.util import img_as_float
from skimage.color import rgb2lab
from numba import jit
# from skimage.segmentation import slic
# from scipy.optimize import minimize
# import pdb


@jit
def _raster_scan(img, l, u, d):  # called by mbd method
    n_rows = len(img)
    n_cols = len(img[0])

    for x in range(1, n_rows - 1):
        for y in range(1, n_cols - 1):
            ix = img[x][y]
            d_ = d[x][y]

            u1 = u[x - 1][y]
            l1 = l[x - 1][y]

            u2 = u[x][y - 1]
            l2 = l[x][y - 1]

            b1 = max(u1, ix) - min(l1, ix)
            b2 = max(u2, ix) - min(l2, ix)

            if d_ <= b1 and d_ <= b2:
                continue
            elif b1 < d_ and b1 <= b2:
                d[x][y] = b1
                u[x][y] = max(u1, ix)
                l[x][y] = min(l1, ix)
            else:
                d[x][y] = b2
                u[x][y] = max(u2, ix)
                l[x][y] = min(l2, ix)


@jit
def _raster_scan_inv(img, l, u, d):  # called by mbd method
    n_rows = len(img)
    n_cols = len(img[0])

    for x in range(n_rows - 2, 1, -1):
        for y in range(n_cols - 2, 1, -1):

            ix = img[x][y]
            d_ = d[x][y]

            u1 = u[x + 1][y]
            l1 = l[x + 1][y]

            u2 = u[x][y + 1]
            l2 = l[x][y + 1]

            b1 = max(u1, ix) - min(l1, ix)
            b2 = max(u2, ix) - min(l2, ix)

            if d_ <= b1 and d_ <= b2:
                continue
            elif b1 < d_ and b1 <= b2:
                d[x][y] = b1
                u[x][y] = max(u1, ix)
                l[x][y] = min(l1, ix)
            else:
                d[x][y] = b2
                u[x][y] = max(u2, ix)
                l[x][y] = min(l2, ix)


@jit
def mbd(img, num_iters):
    if len(img.shape) != 2:
        print('did not get 2d np array to fast mbd')
        return None
    if img.shape[0] <= 3 or img.shape[1] <= 3:
        print('image is too small')
        return None

    l = np.copy(img)
    u = np.copy(img)
    d = np.empty_like(img)
    d.fill(np.inf)

    d[(0, -1), :] = 0
    d[:, (0, -1)] = 0

    # unfortunately, iterating over numpy arrays is very slow
    img_list = img.tolist()
    l_list = l.tolist()
    u_list = u.tolist()
    d_list = d.tolist()

    for x in range(num_iters):
        if x % 2 == 1:
            _raster_scan(img_list, l_list, u_list, d_list)
        else:
            _raster_scan_inv(img_list, l_list, u_list, d_list)

    return np.array(d_list)


def get_saliency_mbd(img, method='b'):
    # Saliency map calculation based on: Minimum Barrier Salient Object Detection at 80 FPS

    # we get either a file name or a list of file names
    if isinstance(img, str):
        img_list = (skimage_imread(img), )
    elif isinstance(img, list):
        if isinstance(img[0], str):
            img_list = [skimage_imread(im) for im in img]
        else:
            img_list = img
    else:
        img_list = (img, )

    result = []

    for img in img_list:
        img_mean = np.mean(img, axis=2)
        sal = mbd(img_mean, 3)

        if 'b' == method:  # get the background map
            # paper uses 30px for an image of size 300px, so we use 10%
            n_rows, n_cols = img.shape[:2]
            img_size = math.sqrt(n_rows * n_cols)
            border_thickness = int(img_size * 0.1)

            img_lab = img_as_float(rgb2lab(img))

            px_left = img_lab[0:border_thickness, :, :]
            px_right = img_lab[n_rows - border_thickness - 1:-1, :, :]

            px_top = img_lab[:, 0:border_thickness, :]
            px_bottom = img_lab[:, n_cols - border_thickness - 1:-1, :]

            px_mean_left = np.mean(px_left, axis=(0, 1))
            px_mean_right = np.mean(px_right, axis=(0, 1))
            px_mean_top = np.mean(px_top, axis=(0, 1))
            px_mean_bottom = np.mean(px_bottom, axis=(0, 1))

            px_left = px_left.reshape((n_cols * border_thickness, 3))
            px_right = px_right.reshape((n_cols * border_thickness, 3))

            px_top = px_top.reshape((n_rows * border_thickness, 3))
            px_bottom = px_bottom.reshape((n_rows * border_thickness, 3))

            cov_left = np.cov(px_left.T)
            cov_right = np.cov(px_right.T)

            cov_top = np.cov(px_top.T)
            cov_bottom = np.cov(px_bottom.T)

            cov_left = np.linalg.inv(cov_left + np.eye(cov_left.shape[1]) * 1e-12)
            cov_right = np.linalg.inv(cov_right + np.eye(cov_right.shape[1]) * 1e-12)

            cov_top = np.linalg.inv(cov_top + np.eye(cov_top.shape[1]) * 1e-12)
            cov_bottom = np.linalg.inv(cov_bottom + np.eye(cov_bottom.shape[1]) * 1e-12)

            # u_left = np.zeros(sal.shape)
            # u_right = np.zeros(sal.shape)
            # u_top = np.zeros(sal.shape)
            # u_bottom = np.zeros(sal.shape)
            #
            # u_final = np.zeros(sal.shape)
            img_lab_unrolled = img_lab.reshape(img_lab.shape[0] * img_lab.shape[1], 3)

            px_mean_left_2 = np.zeros((1, 3))
            px_mean_left_2[0, :] = px_mean_left

            u_left = cdist(img_lab_unrolled, px_mean_left_2, 'mahalanobis', VI=cov_left)
            u_left = u_left.reshape((img_lab.shape[0], img_lab.shape[1]))

            px_mean_right_2 = np.zeros((1, 3))
            px_mean_right_2[0, :] = px_mean_right

            u_right = cdist(img_lab_unrolled, px_mean_right_2, 'mahalanobis', VI=cov_right)
            u_right = u_right.reshape((img_lab.shape[0], img_lab.shape[1]))

            px_mean_top_2 = np.zeros((1, 3))
            px_mean_top_2[0, :] = px_mean_top

            u_top = cdist(img_lab_unrolled, px_mean_top_2, 'mahalanobis', VI=cov_top)
            u_top = u_top.reshape((img_lab.shape[0], img_lab.shape[1]))

            px_mean_bottom_2 = np.zeros((1, 3))
            px_mean_bottom_2[0, :] = px_mean_bottom

            u_bottom = cdist(img_lab_unrolled, px_mean_bottom_2, 'mahalanobis', VI=cov_bottom)
            u_bottom = u_bottom.reshape((img_lab.shape[0], img_lab.shape[1]))

            max_u_left = np.max(u_left)
            max_u_right = np.max(u_right)
            max_u_top = np.max(u_top)
            max_u_bottom = np.max(u_bottom)

            u_left = u_left / max_u_left
            u_right = u_right / max_u_right
            u_top = u_top / max_u_top
            u_bottom = u_bottom / max_u_bottom

            u_max = np.maximum(np.maximum(np.maximum(u_left, u_right), u_top), u_bottom)

            u_final = (u_left + u_right + u_top + u_bottom) - u_max

            u_max_final = np.max(u_final)
            sal_max = np.max(sal)
            sal = sal / sal_max + u_final / u_max_final

        # postprocessing

        # apply centeredness map
        sal /= np.max(sal)

        # s = np.mean(sal)
        # alpha = 50.0
        # delta = alpha * math.sqrt(s)

        xv, yv = np.meshgrid(np.arange(sal.shape[1]), np.arange(sal.shape[0]))
        w2, h2 = np.array(sal.shape) / 2

        c = 1 - np.sqrt(np.square(xv - h2) + np.square(yv - w2)) / math.sqrt(np.square(w2) + np.square(h2))
        sal *= c

        # increase bg/fg contrast

        def f(x):
            b = 10.0
            return 255.0 / (1.0 + math.exp(-b * (x - 0.5)))

        fv = np.vectorize(f)

        sal /= np.max(sal)
        sal = fv(sal)
        result.append(sal)

    if len(result) is 1:
        return result[0]
    return result
