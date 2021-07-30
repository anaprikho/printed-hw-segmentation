"""
This file contains various helper function for image processing
"""
import cv2
import numpy as np
from skimage import img_as_float, img_as_ubyte
from skimage.filters import threshold_sauvola
from skimage.filters import threshold_otsu
from skimage.filters import threshold_minimum
from skimage.filters import threshold_local
from skimage.filters import threshold_niblack
from skimage.filters import threshold_mean
from skimage.filters import roberts
from skimage.color import rgb2gray

def getbinim(image):
    if len(image.shape) >= 3:
        image = rgb2gray(image)
    thresh_sauvola = threshold_sauvola(image)
    return img_as_float(image < thresh_sauvola)

def mybin(image):
    if len(image.shape) >= 3:
        image = rgb2gray(image)
    # image = cv2.GaussianBlur(image, (5, 5), 0)
    # image = cv2.medianBlur(image, 5)
    # img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # img = threshold_local(image, 15, 'mean')
    # img = threshold_local(image, 5, 'median')
    # img = threshold_minimum(image)
    img = threshold_sauvola(image, window_size=9, k=0.1)
    # img = threshold_sauvola(image, window_size=7)
    # img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    # thresh, img_binary = cv2.threshold(image, 155, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # return img_as_float(image < img_binary)
    return img_as_float(image < img)

# adapted from https://www.pyimagesearch.com/2015/09/28/implementing-the-max-rgb-filter-in-opencv/
def max_rgb_filter(image):
    image = image[:, :, ::-1]
    image = img_as_ubyte(image)
    # split the image into its BGR components
    (B, G, R) = cv2.split(image)
    # find the maximum pixel intensity values for each
    # (x, y)-coordinate,, then set all pixel values less
    # than M to zero
    M = np.maximum(np.maximum(R, G), B)
    R[R < M] = 0
    G[G < M] = 0
    B[B < M] = 0

    # merge the channels back together and return the image
    image = cv2.merge([B, G, R])
    image = img_as_float(image)
    image = image[:, :, ::-1]
    return np.ceil(image)


# Util functions to manipulate masks


def rgb2mask(mask):
    result = np.zeros((mask.shape))
    result[:, :][np.where((mask[:, :] == [255, 0, 0]).all(axis=2))] = [0, 0, 1]
    result[:, :][np.where((mask[:, :] == [0, 255, 0]).all(axis=2))] = [0, 0, 2]
    result[:, :][np.where((mask[:, :] == [0, 0, 255]).all(axis=2))] = [0, 0, 4]
    result[:, :][np.where((mask[:, :] == [1, 0, 0]).all(axis=2))] = [0, 0, 1]
    result[:, :][np.where((mask[:, :] == [0, 1, 0]).all(axis=2))] = [0, 0, 2]
    result[:, :][np.where((mask[:, :] == [0, 0, 1]).all(axis=2))] = [0, 0, 4]
    return result

def mask2rgb(mask):
    result = np.zeros((mask.shape))
    result[:, :][np.where((mask[:, :] == [0, 0, 1]).all(axis=2))] = [1, 0, 0]
    result[:, :][np.where((mask[:, :] == [0, 0, 2]).all(axis=2))] = [0, 1, 0]
    result[:, :][np.where((mask[:, :] == [0, 0, 4]).all(axis=2))] = [0, 0, 1]
    return result

def getclass(n, mask):
    result = np.zeros((mask.shape))
    if n == 1: result[:, :][np.where((mask[:, :] == [0, 0, 1]).all(axis=2))] = [0, 0, 1]
    if n == 2: result[:, :][np.where((mask[:, :] == [0, 0, 2]).all(axis=2))] = [0, 0, 2]
    result[:, :][np.where((result[:, :] == [0, 0, 0]).all(axis=2))] = [0, 0, 4]
    return result

def getBinclassImg(n, mask): # hier ist 100% richtig so

    result = np.zeros((mask.shape))
    # print(f'mask: {mask}')
    if n == 1: result[:, :][np.where(np.logical_or((mask[:, :] == [255, 0, 0]),(mask[:, :] == [0, 0, 1])).all(axis=2))] = [1, 1, 1]
    if n == 2: result[:, :][np.where(np.logical_or((mask[:, :] == [0, 255, 0]), (mask[:, :] == [0, 0, 2])).all(axis=2))] = [1, 1, 1]
    if n == 3: result[:, :][np.where(np.logical_or((mask[:, :] == [0, 0, 255]), (mask[:, :] == [0, 0, 4])).all(axis=2))] = [1, 1, 1]

    # if n == 1: result[:, :][np.where((mask[:, :] == [0, 0, 1]).all(axis=2))] = [1, 1, 1]
    # if n == 2: result[:, :][np.where((mask[:, :] == [0, 0, 2]).all(axis=2))] = [1, 1, 1]
    # if n == 3: result[:, :][np.where((mask[:, :] == [0, 0, 4]).all(axis=2))] = [1, 1, 1]
    print(f'-----------------getBinclassImg: {result}------------------')
    print(f'-----------------getBinclassImg: {result[:,:,0]}------------------')
    return result[:,:,0]


def get_IoU(prediction, target):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    print("target in getIoU: ", target)
    print(f'prediction in getIoU: {prediction}')
    print('intersection in getIoU: ', intersection, 'union: ', union)
    print("=============================================================")
    # print(f'that what getIoU returns: {float(np.sum(intersection)) / float(np.sum(union))}')
    if float(np.sum(union)) == 0:
        return float(np.sum(intersection))
    else:
        return float(np.sum(intersection)) / float(np.sum(union))
