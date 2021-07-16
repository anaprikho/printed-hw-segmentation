import glob
import cv2
import os
from os.path import splitext
from PIL import Image,  ImageDraw

import skimage.io as io
import cv2
import os
import xmltodict
import matplotlib.pyplot as plt
import numpy as np
from pyexpat import ExpatError
import json

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
        # dst = filename + ".png"

        # # rename all the files
        os.rename(os.path.join(SOURCE, file), os.path.join(SAVE_DIR, dst))
    # print(len(os.listdir(SOURCE)))

def cropp():
    # for count, file in enumerate(os.listdir(SOURCE)):
    #     img = Image.open(SOURCE + file)
    #     # img.save(SAVE_DIR + str(count) + '_cropped.png', 'JPEG')
    #     # io.imsave(SAVE_DIR + str(count) + '_cropped.png', img)
    #     # img_cropped = img.crop((x_left, y_top, x_right, y_bottom))
    #     print(file, count)

    i = 0
    for img in img_collection:
        img = cv2.imread(img)
        print(img)
        cv2.imwrite(SAVE_DIR + 'cvl_' + str(i) + '_cropped.png', img)
        i = i + 1

    # # print(os.listdir(SAVE_DIR))
    # img = Image.open(SOURCE + "0001-21.png")
    # img.show()
    # x_left, y_top, x_right, y_bottom = 100, 300, 800, 1500
    # img_cropped = img.crop((x_left, y_top, x_right, y_bottom))
    # # img.save('C:/Users/prikha/Downloads/BA/Datasets/CVL_pages/test/cropped.png', "PNG")
    # cv2.imwrite('C:/Users/prikha/Downloads/BA/Datasets/CVL_pages/cropped.jpg', img)


def cropp_cvl(image, ground_truth, name, y_lo=645, y_up=2215+570, x_lim=20):
    print(image)
    try:
        doc = xmltodict.parse(ground_truth.read())
        # print(doc)
    except ExpatError:
        print('XML file malformated: ' + name + '.xml' + ' skipping..')
        return

    # x = int(doc['PcGts']['Page']['AttrRegion']['minAreaRect']['Point'][0]['@x'])
    # y = int(doc['PcGts']['Page']['AttrRegion']['minAreaRect']['Point'][0]['@y'])
    y = int(doc['PcGts']['Page']['dkTextLines']['@cropY'])
    h = int(doc['PcGts']['Page']['dkTextLines']['@cropH'])
    print('y= :', y)

    # print('before')

    img = cv2.imread(image)
    # print(img)
    # color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # pil_image = Image.fromarray(color_coverted)
    # img = Image.open(pil_image)

    # img = img[:y_up, :]
    # if img is not empt:
    img_cropped = img[y:y_up, :]
    print(img_cropped.size)
#     cv2.imwrite("C:/Users/prikha/Downloads/BA/Datasets/CVL_pages/cropps/' + name + '_cropped_hw.png", img_cropped)
    # color_coverted = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
    # pil_image = Image.fromarray(color_coverted)
    # cv2.imshow("cropped", img_cropped)
    # cv2.waitKey(0)
    # print('mid')
    # print(img)
    # img_cropped = img.crop((x, y, x_lim, y_lo))

    # print('after')
    # try:
    if (img_cropped.size != 0):
        cv2.imwrite("C:/Users/prikha/Downloads/BA/Datasets/CVL_pages/cropps/" + name + "_cropped_hw.PNG", img_cropped)
    # except cv2.error as e:
    #     print(e)

    # io.imsave(SAVE_DIR + name + '.png', color_coverted)



# Driver Code
if __name__ == '__main__':
    # cropp()
    # main()

    input_files = os.listdir(SOURCE)
    xml_files = os.listdir(XML_DATA_PATH)
    for xml, img in zip(xml_files, img_collection):
        name = os.path.splitext(xml)[0]
        with open(XML_DATA_PATH + xml, 'r') as xml_doc:
            cropp_cvl(img, xml_doc, name)
        print('===================================')