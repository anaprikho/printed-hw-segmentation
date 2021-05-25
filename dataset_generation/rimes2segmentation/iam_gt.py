import skimage.io as io
import cv2
import os
import xmltodict
import matplotlib.pyplot as plt
import numpy as np
from pyexpat import ExpatError
import json

from img_utils import *

XML_DATA_PATH = 'C:/Users/prikha/Downloads/BA/Datasets/IAM handwriting database/xml/xml/'     # 'C:/Users/prikha/Downloads/BA/Datasets/IAM handwriting database/xml/xml/'
IM_DATA_PATH = 'C:/Users/prikha/Downloads/BA/Datasets/IAM handwriting database/formsE-H/formsE-H'

IM_OUT_PATH = 'C:/Users/prikha/Downloads/BA/Datasets/IAM handwriting database/iam_gt_output_dsy/'


def xml2segmentation(image, ground_truth, name, y_lo=645, y_up=2215, x_lim=20):
    """
    Takes an image file and groundtruth file from the RIMES dataset
    and outputs an RGB image with pixel-wise labels:
    R : printed
    G : handwritten
    B : Background / noise
    :param name: output name (without extension)
    :param img: input image file name
    :param xml: input xml file name
    :return: pixel-wise annotated image
    """
    # orgim = np.copy(image)
    # image = gray2rgb(getbinim(image))
    #
    # mask = image
    # try:
    #     doc = xmltodict.parse(ground_truth.read())
    # except ExpatError:
    #     print('XML file malformated: ' + name + '.xml' + ' skipping..')
    #     return

#----------------------------------------------------------------------
    #from data_gen.py (iam2segmentation)

    orgim = np.copy(image)
    image = ndimage.filters.median_filter(image, 3)
    image = image[:y_up+550, :]
    #image = image[:y_up, x_lim:]
    bin_im = getbinim(image)
    bin_im = gray2rgb(bin_im)
    mask = bin_im

    try:
        doc = xmltodict.parse(ground_truth.read())
    except ExpatError:
        print('XML file malformated: ' + name + '.xml' + ' skipping..')
        return


    # bin_im[0:y_lo, :][np.where((bin_im[0:y_lo, :] == [1, 1, 1]).all(axis=2))] = [
    #     1, 0, 0]
    # bin_im[y_lo:y_up, :][np.where(
    #     (bin_im[y_lo:y_up, :] == [1, 1, 1]).all(axis=2))] = [0, 1, 0]
    # bin_im[:, :][np.where((bin_im[:, :] == [0, 0, 0]).all(axis=2))] = [
    #     0, 0, 1]
#-----------------------------------------------------------------------
   #dict = json.loads(json.dumps(doc))
    #my_dict = doc['form']['handwritten-part']['line'][0]['asy']
    #print(my_dict)


    # bboxes = doc['form']
    #
    # for boxes in bboxes:
    #     y_upper_boundary = int(doc['form']['handwritten-part']['line'][0]['@asy'])
    #
    #     mask[0:y_upper_boundary, :][np.where(
    #         (bin_im[0:y_upper_boundary, :] == [1, 1, 1]).all(axis=2))] = [1, 0, 0]
    #
    #     mask[y_upper_boundary:, :][np.where(
    #         (bin_im[y_upper_boundary:y_up, :] == [1, 1, 1]).all(axis=2))] = [0, 1, 0]

    #     # x1, y1 = int(boxes['@top_left_x']), int(boxes['@top_left_y'])
    #     # x2, y2 = int(boxes['@bottom_right_x']), int(boxes['@bottom_right_y'])
    #     # type = boxes['type']
    #
    #     if 'DactylographiÃ©' in boxes:
    #         mask[0:y_lo, :][np.where(
    #             (bin_im[0:y_lo, :] == [1, 1, 1]).all(axis=2))] = [1, 0, 0]
    #     if 'Manuscrit' in type:
    #         mask[y_lo:y_up, :][np.where(
    #              (bin_im[y_lo:y_up, :] == [1, 1, 1]).all(axis=2))] = [0, 1, 0]

    #y_upper_boundary = int(doc['form']['handwritten-part']['line'][0]['@asy'])-150
    #print(doc['form']['handwritten-part']['line'][0]['word'][0]['cmp'][0]['@height'])
    #y_upper_boundary = int(doc['form']['handwritten-part']['line'][0]['@asy']) - \
    #                   int(doc['form']['handwritten-part']['line'][0]['@threshold'])

    difference = int(doc['form']['handwritten-part']['line'][0]['@dsy']) - int(doc['form']['handwritten-part']['line'][0]['@uby'])
    #y_upper_boundary = int(doc['form']['handwritten-part']['line'][0]['@asy']) - difference
    y_upper_boundary = 660
    print("asy value: ", y_upper_boundary)
    print("difference: ", difference)
    #print(int(doc['form']['handwritten-part']['line'][0]['@dsy'])-int(doc['form']['handwritten-part']['line'][0]['@asy']))

    mask[0:y_upper_boundary, :][np.where(
        (bin_im[0:y_upper_boundary, :] == [1, 1, 1]).all(axis=2))] = [1, 0, 0]

    mask[y_upper_boundary:, :][np.where(
        (bin_im[y_upper_boundary:, :] == [1, 1, 1]).all(axis=2))] = [0, 1, 0]

    mask[:, :][np.where(
        (bin_im[:, :] == [0, 0, 0]).all(axis=2))] = [0, 0, 1]

    io.imsave(IM_OUT_PATH + name + '.png', mask)
    #io.imsave('data/input' + name + '.png', orgim)


if __name__ == '__main__':
    input_files = io.imread_collection(IM_DATA_PATH + '/*')
    xml_files = os.listdir(XML_DATA_PATH)
    for xml, img in zip(xml_files, input_files):
        name = os.path.splitext(xml)[0]
        with open(XML_DATA_PATH + xml, 'r') as xml_doc:
            xml2segmentation(img, xml_doc, name)