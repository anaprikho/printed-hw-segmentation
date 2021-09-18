import os
import skimage.io as io
from keras.layers import *
from keras.models import *
from skimage import img_as_float
from skimage.color import gray2rgb
from fcn_helper_function import *
from img_utils import getbinim
import pickle
import pandas as pd

"""
This file defines the model fcn-light-2. It is a fully convolutional model based on the FCN-8 architecture
"""


np.random.seed(123)
#'''



output_folder = 'C:/Users/prikha/Downloads/BA/test/'
# feature vectors
X_train = []
X_valid = []

# labels
y_train = []
y_valid = []

print("Reading images...")

# path = os.getcwd()
# os.chdir('../')

inputs_train = io.imread_collection("D:/Uni/BA/Datasets/HTSNet_data_synthesis/wgm-cvl_jottueset_subset/model_input/train/syn/*")
inputs_valid = io.imread_collection("D:/Uni/BA/Datasets/HTSNet_data_synthesis/wgm-cvl_jottueset_subset/model_input/validation/syn/*")

masks_train = io.imread_collection("D:/Uni/BA/Datasets/HTSNet_data_synthesis/wgm-cvl_jottueset_subset/model_input/train/label/*")
masks_valid = io.imread_collection("D:/Uni/BA/Datasets/HTSNet_data_synthesis/wgm-cvl_jottueset_subset/model_input/validation/label/*")


def mask2rgb(mask):
    result = np.zeros((mask.shape))  # create a blank black mask with a shape = 3: ...[0. 0. 0.]...
    result[:, :][np.where((mask[:, :] == [255, 0, 0]).all(axis=2))] = [1, 0, 0]
    result[:, :][np.where((mask[:, :] == [0, 255, 0]).all(axis=2))] = [0, 1, 0]
    result[:, :][np.where((mask[:, :] == [0, 0, 255]).all(axis=2))] = [0, 0, 1]
    print(result)
    return result


for im_in, im_mask in zip(inputs_train, masks_train):
    X_train.append(img_as_float(gray2rgb(getbinim(im_in))))
    y_train.append(mask2rgb(im_mask))


for im_in, im_mask in zip(inputs_valid, masks_valid):
    X_valid.append(img_as_float(gray2rgb(getbinim(im_in))))
    y_valid.append(mask2rgb(im_mask))

os.chdir('../')

print('dumping x_valid')
pickle.dump(X_valid, open("FCN/models/x_valid.sav", "wb"))
print('done x_valid')
del X_valid

print("dumping y_valid")
pickle.dump(y_valid, open("FCN/models/y_valid.sav", "wb"))
print("done")
del y_valid

print('dumping x_train')
pickle.dump(X_train, open("FCN/models/x_train.sav", "wb"))
print('done')
del X_train

print('dumping y_train')
pickle.dump(y_train, open("FCN/models/y_train.sav", "wb"))
exit()
#'''

os.chdir('../')

X_valid = pickle.load(open("FCN/models/x_valid.sav", "rb"))
y_valid = pickle.load(open("FCN/models/y_valid.sav", "rb"))
X_train = pickle.load(open("FCN/models/x_train.sav", "rb"))
y_train = pickle.load(open("FCN/models/y_train.sav", "rb"))

print('done reading')
X_valid = np.array(X_valid)
X_valid = (X_valid - X_valid.mean()) / X_valid.std()
print('done valid std norm')
X_train = np.array(X_train)
X_train = (X_train - X_train.mean()) / X_train.std()

y_train = np.array(y_train)
y_valid = np.array(y_valid)

print("Done!")

print('Number of training samples:' + str(len(X_train)))
print('Number of validation samples:' + str(len(y_valid)))


def FCN(nClasses, input_height=256, input_width=256):
    IMAGE_ORDERING = "channels_last"

    img_input = Input(shape=(input_height, input_width, 3))

    x = Conv2D(32, (3, 3), activation='relu', padding='same',
               data_format=IMAGE_ORDERING)(img_input)
    skip_1_in = Conv2D(32, (3, 3), activation='relu',
                       padding='same', data_format=IMAGE_ORDERING)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), data_format=IMAGE_ORDERING)(skip_1_in)

    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', data_format=IMAGE_ORDERING)(x)
    skip_2_in = Conv2D(64, (3, 3), activation='relu',
                       padding='same', data_format=IMAGE_ORDERING)(x)

    x = Dropout(0.05)(skip_2_in)

    x = MaxPooling2D((2, 2), strides=(2, 2), data_format=IMAGE_ORDERING)(x)

    x = Conv2D(128, (3, 3), activation='relu',
               padding='same', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), activation='relu',
               padding='same', data_format=IMAGE_ORDERING)(x)

    x = Dropout(0.1)(x)

    x = (Conv2D(54, (1, 1), activation='relu',
                padding='same', data_format=IMAGE_ORDERING))(x)

    x = Dropout(0.2)(x)

    skip_reduce_2 = Conv2D(nClasses, (1, 1), activation='relu', padding='same')(skip_2_in)

    x = Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(
        2, 2), use_bias=False, data_format=IMAGE_ORDERING)(x)

    skip_2_out = Add()([x, skip_reduce_2])

    skip_reduce_1 = Conv2D(nClasses, (1, 1), activation='relu', padding='same')(skip_1_in)

    x = Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(
        2, 2), use_bias=False, data_format=IMAGE_ORDERING)(skip_2_out)

    skip_1_out = Add()([x, skip_reduce_1])

    o = (Activation('softmax'))(skip_1_out)

    model = Model(img_input, o)

    return model


model = FCN(nClasses=3,
            input_height=256,
            input_width=256)

from contextlib import redirect_stdout

with open('modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

print(model.summary())

model.compile(loss=[weighted_categorical_crossentropy([0.4, 0.5, 0.1])],
              optimizer='adam',
              metrics=[IoU])

history = model.fit(x=X_train, y=y_train, epochs=15, batch_size=16, validation_data=(X_valid, y_valid))

hist_df = pd.DataFrame(history.history)
hist_json_file = 'history_fcnn_wgm-cvl_jottueset_subset.json'
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# hist_csv_file = 'history.csv'
# with open(hist_csv_file, mode='w') as f:
#     hist_df.to_csv(f)

model.save('FCN/models/fcnn_wgm-cvl_jottueset_subset.h5')
