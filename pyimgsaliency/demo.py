import numpy as np
import pyimgsaliency as psal
import cv2

if '__main__' == __name__:
    # path to the image
    filename = 'bird.jpg'

    # get the saliency maps using the 3 implemented methods
    rbd = psal.get_saliency_rbd(filename).astype(np.uint8)

    ft = psal.get_saliency_ft(filename).astype(np.uint8)

    mbd = psal.get_saliency_mbd(filename).astype(np.uint8)

    # often, it is desirable to have a binary saliency map
    binary_sal = psal.binarise_saliency_map(mbd, method='adaptive')

    img = cv2.imread(filename)

    cv2.imshow('img', img)
    cv2.imshow('rbd', rbd)
    cv2.imshow('ft', ft)
    cv2.imshow('mbd', mbd)

    # OpenCV cannot display numpy type 0, so convert to uint8 and scale
    cv2.imshow('binary', 255 * binary_sal.astype(np.uint8))

    cv2.waitKey(0)
