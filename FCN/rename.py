import glob
import random

import cv2
import os
from os.path import splitext
from PIL import Image

import skimage.io as io
import cv2
import os
import xmltodict
import numpy as np
from pyexpat import ExpatError

# SOURCE = 'C:/Users/prikha/Downloads/BA/Datasets/HTSNet_data_synthesis/FINAL/VoTT_color_input_binarized_images_FINAL/model_input/evaluation/syn/'
SOURCE = 'C:/Users/prikha/Downloads/BA/Datasets/CVL_pages/pages_hw/'
# SOURCE = 'C:/Users/prikha/Downloads/BA/Datasets/CVL_pages/original_tif/'

XML_DATA_PATH = 'C:/Users/prikha/Downloads/BA/Datasets/CVL_pages/xml/'

SAVE_DIR = 'C:/Users/prikha/Downloads/BA/Datasets/CVL_pages/cropps/'
# SAVE_DIR = 'C:/Users/prikha/Downloads/BA/Datasets/CVL_pages/original_png/'

img_collection = glob.glob('C:/Users/prikha/Downloads/BA/Datasets/CVL_pages/pages_hw/*')

def convert2png():
    for count, file in enumerate(os.listdir(SOURCE)):
        filename, extension = splitext(file)
        print(filename, extension)
        dst = filename + str(count) + '.png'
        # cv2.imwrite(SAVE_DIR + dst, file)

        os.rename(os.path.join(SOURCE, file), os.path.join(SAVE_DIR, dst))
    # print(len(os.listdir(SOURCE)))

def data_generation():
    xml_files = os.listdir(XML_DATA_PATH)
    for xml, img in zip(xml_files, img_collection):
        name = os.path.splitext(xml)[0]
        with open(XML_DATA_PATH + xml, 'r') as xml_doc:
            cropp_cvl(img, xml_doc, name)
        print('===================================')



def cropp_cvl(image, ground_truth, name, y_up=2215+570):
    print(image)
    try:
        doc = xmltodict.parse(ground_truth.read())
    except ExpatError:
        print('XML file malformated: ' + name + '.xml' + ' skipping..')
        return


    y = int(doc['PcGts']['Page']['dkTextLines']['@cropY'])
    h = int(doc['PcGts']['Page']['dkTextLines']['@cropH'])
    # dist = int(float(doc['PcGts']['Page']['dkTextLines']['textlines'][0]['@yS']))
    # dist2 = int(doc['PcGts']['Page']['AttrRegion']['minAreaRect']['Point'][0]['@y'])
    print('y= :', y)
    # print('dist2= :', dist2)

    img = cv2.imread(image)
    # print(img)
    # color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # pil_image = Image.fromarray(color_coverted)
    # img = Image.open(pil_image)

    img_cropped = img[y+200:h, :]
    # if dist2 > 540:
    #     img_cropped = img[y+400:y_up, :]
    # else:
    #     img_cropped = img[y+dist2:y_up, :]
    print(img_cropped.size)
#     cv2.imwrite("C:/Users/prikha/Downloads/BA/Datasets/CVL_pages/cropps/' + name + '_cropped_hw.png", img_cropped)
    # color_coverted = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
    # pil_image = Image.fromarray(color_coverted)

    if (img_cropped.size != 0):
        cv2.imwrite("C:/Users/prikha/Downloads/BA/Datasets/CVL_pages/cropps/" + name + "_cropped_hw.PNG", img_cropped)

    # io.imsave(SAVE_DIR + name + '.png', color_coverted)


def select_img():
    path_syn = 'C:/Users/prikha/Downloads/BA/Datasets/HTSNet_data_synthesis/cvl_jottueset/syn'
    save_dir_syn = 'C:/Users/prikha/Downloads/BA/Datasets/HTSNet_data_synthesis/cvl_jottueset/model_input/test/syn'

    path_label = 'C:/Users/prikha/Downloads/BA/Datasets/HTSNet_data_synthesis/cvl_jottueset/label'
    save_dir_label = 'C:/Users/prikha/Downloads/BA/Datasets/HTSNet_data_synthesis/cvl_jottueset/model_input/test/label'

    files = os.listdir(path_syn)
    print(len(files))

    sampled_list = random.sample(files, 1704)
    print(len(sampled_list))

    for count, file in enumerate(sampled_list):
        filename, extension = splitext(file)
        print(filename, extension)
        dst = filename + '.png'

        os.rename(os.path.join(path_syn, file), os.path.join(save_dir_syn, dst))
        os.rename(os.path.join(path_label, file), os.path.join(save_dir_label, dst))


if __name__ == '__main__':
    # convert2png()
    # data_generation()
    select_img()