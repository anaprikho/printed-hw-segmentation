import os
import skimage.io as io
from keras.layers import *
from keras.models import *
from keras.utils import to_categorical
from skimage import img_as_float
from skimage.color import gray2rgb
#  from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from fcn_helper_function import *
from img_utils import getbinim
import pickle
import cv2


"""
This file defines the model fcn-light-2. It is a fully convolutional model based on the FCN-8 architecture
"""

# import warnings
# warnings.filterwarnings("ignore", message=r"Passing",category=FutureWarning)
# np.random.seed(123)

output_folder = 'C:/Users/prikha/Downloads/BA/test/'
# feature vectors
X_train = []
X_valid = []

# labels
y_train = []
y_valid = []

print("Reading images...")

#path = os.getcwd()
os.chdir('../')
inputs_train = io.imread_collection("train/data/*.png")
inputs_valid = io.imread_collection("val/data/*.png")

masks_train = io.imread_collection("train/gt/*.png")
masks_valid = io.imread_collection("val/gt/*.png")

def mask2rgb(mask):
    # print(mask)
    result = np.zeros((mask.shape))  # crestin a blank black mask with shape = 3: ...[0. 0. 0.]...
    #print(result)
    result[:, :][np.where((mask[:, :] == [255, 0, 0]).all(axis=2))] = [1, 0, 0]  # red where input image pixels are ??? (machine-printed)
    result[:, :][np.where((mask[:, :] == [0, 255, 0]).all(axis=2))] = [0, 1, 0]  # green where was yellow (handwritten)
    result[:, :][np.where((mask[:, :] == [0, 0, 255]).all(axis=2))] = [0, 0, 1]  # blue (background)
    print(result)
    # cv2.imshow('mask', result)
    # cv2.waitKey()
    return result

# for im_in,im_mask in zip(inputs_train, masks_train):
#     X_train.append(img_as_float(gray2rgb(getbinim(im_in))))
#     y_train.append(mask2rgb(im_mask))

# print(list(zip(inputs_train, masks_train)))

for im_in, im_mask in zip(inputs_train, masks_train):
    X_train.append(img_as_float(gray2rgb(getbinim(im_in))))
    mask2rgb(im_mask)
    print("===============")
    # io.imsave(output_folder + 'out.png', im_mask)
    y_train.append(mask2rgb(im_mask))

# print(inputs_train[2])
# print(getbinim(inputs_train[0]))


# for im_in,im_mask in zip(inputs_valid, masks_valid):
#     X_valid.append(img_as_float(gray2rgb(getbinim(im_in))))
#     y_valid.append(mask2rgb(im_mask))

# print('dumping x_valid')
# pickle.dump(X_valid, open("models/x_valid.sav", "wb"))
# print('done x_valid')
# del X_valid
# print("dumping y_valid")
# pickle.dump(y_valid, open("models/y_valid.sav", "wb"))
# print("done")
# del y_valid
# print('dumping x_train')
# pickle.dump(X_train, open("models/x_train.sav", "wb"))
# print('done')
# del X_train
# print('dumping y_train')
# pickle.dump(y_train, open("models/y_train.sav", "wb"))
# exit()
