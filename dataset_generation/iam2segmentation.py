import skimage.io as io
import os
import numpy as np

from img_utils import *

XML_DATA_PATH = 'D:/Uni/BA/Datasets/IAM/xml/xml'
IM_DATA_PATH = 'D:/Uni/BA/Datasets/IAM/formsE-H/formsE-H/'

IM_GT_OUT_PATH = 'D:/Uni/BA/Datasets/IAM/USED/Z/'
IM_OUT_PATH = 'D:/Uni/BA/Datasets/IAM/USED/E-H/crops/'


def iam2segmentation(image, name, y_lo=645, y_up=2215+570):
    """
    Takes an image file and its name from IAM dataset
    and outputs a cropped original image and an RGB image with pixel-wise labels:
    R : printed
    G : handwritten
    B : Background / noise
    :param y_up: end of a hw part
    :param y_lo: y coordinate of the separation line between hw and printed text parts
    :param name: output name (without extension)
    :param image: input image file name
    :return: pixel-wise annotated image
    """

    image = ndimage.filters.median_filter(image, 3)

    #  crop an image
    image = image[:y_up, :]
    io.imsave(IM_OUT_PATH + name + '.png', image)

    # binarize an image
    bin_im = getbinim(image)
    bin_im = gray2rgb(bin_im)
    mask = bin_im

    # --------------USING XML----------------

    # Use this part when reading from IAM XML

    # try:
    #     doc = xmltodict.parse(ground_truth.read())
    # except ExpatError:
    #     print('XML file malformated: ' + name + '.xml' + ' skipping..')
    #     return

    # asy_value = int(doc['form']['handwritten-part']['line'][0]['@asy'])

    # --------------USING XML----------------

    mask[0:y_lo, :][np.where(
        (bin_im[0:y_lo, :] == [1, 1, 1]).all(axis=2))] = [1, 0, 0]

    mask[y_lo:y_up, :][np.where(
        (bin_im[y_lo:y_up, :] == [1, 1, 1]).all(axis=2))] = [0, 1, 0]

    mask[:, :][np.where(
        (bin_im[:, :] == [0, 0, 0]).all(axis=2))] = [0, 0, 1]

    io.imsave(IM_GT_OUT_PATH + name + '.png', mask)


if __name__ == '__main__':
    input_files = io.imread_collection(IM_DATA_PATH + '/*')
    names = os.listdir(IM_DATA_PATH)

    # --------------USING XML----------------

    # Use this part when reading from IAM XML

    # xml_files = os.listdir(XML_DATA_PATH)
    # for xml, img in zip(xml_files, input_files):
    #     name = os.path.splitext(xml)[0]
    #     with open(XML_DATA_PATH + xml, 'r') as xml_doc:
    #         xml2segmentation(img, xml_doc, name)

    # --------------USING XML----------------

    # Without XML by setting fixed coordinates
    for img, file in zip(input_files, names):
        name = os.path.splitext(file)[0]
        iam2segmentation(img, name)