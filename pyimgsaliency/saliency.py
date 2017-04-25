import math
# import sys
# import operator
import networkx as nx
# import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance
import scipy.signal
from skimage.io import imread as skimage_imread
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.color import rgb2gray, gray2rgb, rgb2lab
# from scipy.optimize import minimize
# import pdb


def _func_s(x1, x2, geodesic, sigma_clr=10):  # called by rbd method
    return math.exp(-pow(geodesic[x1, x2], 2) / (2 * sigma_clr * sigma_clr))


def _compute_saliency_cost(smoothness, w_bg, w_ctr):  # called by rbd method
    n = len(w_bg)
    arr = np.zeros((n, n))
    b = np.zeros(n)

    for x in range(0, n):
        arr[x, x] = 2 * w_bg[x] + 2 * w_ctr[x]
        b[x] = 2 * w_ctr[x]
        for y in range(0, n):
            arr[x, x] += 2 * smoothness[x, y]
            arr[x, y] -= 2 * smoothness[x, y]

    x = np.linalg.solve(arr, b)
    return x


def _path_length(path, graph):  # called by rbd method
    return np.sum([graph[path[i - 1]][path[i]]['weight'] for i in range(1, len(path))])


def _make_graph(grid):  # called by rbd method
    # get unique labels
    vertices = np.unique(grid)

    # map unique labels to [1,...,num_labels]
    reverse_dict = dict(zip(vertices, np.arange(len(vertices))))
    grid = np.array([reverse_dict[x] for x in grid.flat]).reshape(grid.shape)

    # create edges
    down = np.c_[grid[:-1, :].ravel(), grid[1:, :].ravel()]
    right = np.c_[grid[:, :-1].ravel(), grid[:, 1:].ravel()]
    all_edges = np.vstack([right, down])
    all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1], :]
    all_edges = np.sort(all_edges, axis=1)
    num_vertices = len(vertices)
    edge_hash = all_edges[:, 0] + num_vertices * all_edges[:, 1]
    # find unique connections
    edges = np.unique(edge_hash)
    # undo hashing
    edges = [[vertices[int(x % num_vertices)],
              vertices[int(x / num_vertices)]] for x in edges]

    return vertices, edges


def get_saliency_rbd(img):
    # Saliency map calculation based on:
    # Saliency Optimization from Robust Background Detection, Wangjiang Zhu, Shuang Liang, Yichen Wei and Jian Sun,
    # IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014

    if isinstance(img, str):  # img is img_path string
        img = skimage_imread(img)

    if len(img.shape) != 3:  # got a grayscale image
        img = gray2rgb(img)

    img_lab = img_as_float(rgb2lab(img))

    img_rgb = img_as_float(img)

    img_gray = img_as_float(rgb2gray(img))

    segments_slic = slic(img_rgb, n_segments=250, compactness=10, sigma=1, enforce_connectivity=False)

    # num_segments = len(np.unique(segments_slic))

    nrows, ncols = segments_slic.shape
    max_dist = math.sqrt(nrows * nrows + ncols * ncols)

    grid = segments_slic

    vertices, edges = _make_graph(grid)

    gridx, gridy = np.mgrid[:grid.shape[0], :grid.shape[1]]

    centers = dict()
    colors = dict()
    # distances = dict()
    boundary = dict()

    for v in vertices:
        centers[v] = [gridy[grid == v].mean(), gridx[grid == v].mean()]
        colors[v] = np.mean(img_lab[grid == v], axis=0)

        x_pix = gridx[grid == v]
        y_pix = gridy[grid == v]

        if np.any(x_pix == 0) or np.any(y_pix == 0) or np.any(x_pix == nrows - 1) or np.any(y_pix == ncols - 1):
            boundary[v] = 1
        else:
            boundary[v] = 0

    graph = nx.Graph()

    # build the graph
    for edge in edges:
        pt1 = edge[0]
        pt2 = edge[1]
        color_distance = scipy.spatial.distance.euclidean(colors[pt1], colors[pt2])
        graph.add_edge(pt1, pt2, weight=color_distance)

    # add a new edge in graph if edges are both on boundary
    for v1 in vertices:
        if boundary[v1] == 1:
            for v2 in vertices:
                if boundary[v2] == 1:
                    color_distance = scipy.spatial.distance.euclidean(colors[v1], colors[v2])
                    graph.add_edge(v1, v2, weight=color_distance)

    geodesic = np.zeros((len(vertices), len(vertices)), dtype=float)
    spatial = np.zeros((len(vertices), len(vertices)), dtype=float)
    smoothness = np.zeros((len(vertices), len(vertices)), dtype=float)
    adjacency = np.zeros((len(vertices), len(vertices)), dtype=float)

    sigma_clr = 10.0
    sigma_bndcon = 1.0
    sigma_spa = 0.25
    mu = 0.1

    all_shortest_paths_color = nx.shortest_path(graph, source=None, target=None, weight='weight')

    for v1 in vertices:
        for v2 in vertices:
            if v1 == v2:
                geodesic[v1, v2] = 0
                spatial[v1, v2] = 0
                smoothness[v1, v2] = 0
            else:
                geodesic[v1, v2] = _path_length(all_shortest_paths_color[v1][v2], graph)
                spatial[v1, v2] = scipy.spatial.distance.euclidean(centers[v1], centers[v2]) / max_dist
                smoothness[v1, v2] = math.exp(- (geodesic[v1, v2] * geodesic[v1, v2]) / (2.0 * sigma_clr * sigma_clr)) + mu

    for edge in edges:
        pt1 = edge[0]
        pt2 = edge[1]
        adjacency[pt1, pt2] = 1
        adjacency[pt2, pt1] = 1

    for v1 in vertices:
        for v2 in vertices:
            smoothness[v1, v2] = adjacency[v1, v2] * smoothness[v1, v2]

    area = dict()
    len_bnd = dict()
    bnd_con = dict()
    w_bg = dict()
    ctr = dict()
    w_ctr = dict()

    for v1 in vertices:
        area[v1] = 0
        len_bnd[v1] = 0
        ctr[v1] = 0
        for v2 in vertices:
            d_app = geodesic[v1, v2]
            d_spa = spatial[v1, v2]
            w_spa = math.exp(- (d_spa * d_spa) / (2.0 * sigma_spa * sigma_spa))
            area_i = _func_s(v1, v2, geodesic)
            area[v1] += area_i
            len_bnd[v1] += area_i * boundary[v2]
            ctr[v1] += d_app * w_spa
        bnd_con[v1] = len_bnd[v1] / math.sqrt(area[v1])
        w_bg[v1] = 1.0 - math.exp(- (bnd_con[v1] * bnd_con[v1]) / (2 * sigma_bndcon * sigma_bndcon))

    for v1 in vertices:
        w_ctr[v1] = 0
        for v2 in vertices:
            d_app = geodesic[v1, v2]
            d_spa = spatial[v1, v2]
            w_spa = math.exp(- (d_spa * d_spa) / (2.0 * sigma_spa * sigma_spa))
            w_ctr[v1] += d_app * w_spa * w_bg[v2]

    # normalise value for w_ctr

    min_value = min(w_ctr.values())
    max_value = max(w_ctr.values())

    # minVal = [key for key, value in w_ctr.iteritems() if value == min_value]
    # maxVal = [key for key, value in w_ctr.iteritems() if value == max_value]

    for v in vertices:
        w_ctr[v] = (w_ctr[v] - min_value) / (max_value - min_value)

    sal = img_gray.copy()

    x = _compute_saliency_cost(smoothness, w_bg, w_ctr)

    for v in vertices:
        sal[grid == v] = x[v]

    sal_max = np.max(sal)
    sal_min = np.min(sal)
    sal = 255 * ((sal - sal_min) / (sal_max - sal_min))  # normalize to [0, 255[

    return sal


def get_saliency_ft(img):
    # Saliency map calculation based on:

    if isinstance(img, str):  # img is img_path string
        img = skimage_imread(img)

    img_rgb = img_as_float(img)

    img_lab = rgb2lab(img_rgb)

    mean_val = np.mean(img_rgb, axis=(0, 1))

    kernel_h = (1.0 / 16.0) * np.array([[1, 4, 6, 4, 1]])
    kernel_w = kernel_h.transpose()

    blurred_l = scipy.signal.convolve2d(img_lab[:, :, 0], kernel_h, mode='same')
    blurred_a = scipy.signal.convolve2d(img_lab[:, :, 1], kernel_h, mode='same')
    blurred_b = scipy.signal.convolve2d(img_lab[:, :, 2], kernel_h, mode='same')

    blurred_l = scipy.signal.convolve2d(blurred_l, kernel_w, mode='same')
    blurred_a = scipy.signal.convolve2d(blurred_a, kernel_w, mode='same')
    blurred_b = scipy.signal.convolve2d(blurred_b, kernel_w, mode='same')

    im_blurred = np.dstack([blurred_l, blurred_a, blurred_b])

    sal = np.linalg.norm(mean_val - im_blurred, axis=2)
    sal_max = np.max(sal)
    sal_min = np.min(sal)
    sal = 255 * ((sal - sal_min) / (sal_max - sal_min))  # normalize to [0, 255[
    return sal
