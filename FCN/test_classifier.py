"""
This file can be used to evaluate models. You will only need the test/ folder of the
dataset printed-hw-seg which can be downloaded in the thesis.
Run python test_classifier.py for help
"""
import argparse
import sys
import warnings

import numpy as np
import skimage.io as io
from fcn_helper_function import weighted_categorical_crossentropy, IoU
from img_utils import max_rgb_filter, get_IoU, getBinclassImg, mask2rgb, rgb2mask, getbinim
from keras.engine.saving import load_model
from post import crf
from skimage import img_as_float
from skimage.color import gray2rgb
from tqdm import tqdm

if not sys.warnoptions:
    warnings.simplefilter("ignore")

BOXWDITH = 256
STRIDE = BOXWDITH - 10


def classify(imgdb):
    result_imgs = []
    print("classifying " + str(len(imgdb)) + ' images')
    for image in imgdb:
        model = load_model('models/fcnn_wgm-cvl_jottueset.h5', custom_objects={
            'loss': weighted_categorical_crossentropy([0.4, 0.5, 0.1]), 'IoU': IoU})
        orgim = np.copy(image)
        image = img_as_float(gray2rgb(getbinim(image)))
        maskw = int((np.ceil(image.shape[1] / BOXWDITH) * BOXWDITH)) + 1
        maskh = int((np.ceil(image.shape[0] / BOXWDITH) * BOXWDITH))
        mask = np.ones((maskh, maskw, 3))
        mask2 = np.zeros((maskh, maskw, 3))
        mask[0:image.shape[0], 0:image.shape[1]] = image
        print("going into the 1st for-loop...")
        for y in tqdm(range(0, mask.shape[0], STRIDE), unit='batch'):
            x = 0
            if (y + BOXWDITH > mask.shape[0]):
                break
            while (x + BOXWDITH) < mask.shape[1]:
                input = mask[y:y + BOXWDITH, x:x + BOXWDITH]
                std = input.std() if input.std() != 0 else 1
                mean = input.mean()
                mask2[y:y + BOXWDITH, x:x + BOXWDITH] = model.predict(
                    np.array([(input - mean) / std]))[0]
                x = x + STRIDE
        result_imgs.append(mask2[0:image.shape[0], 0:image.shape[1]])
        print("end of the 1st for-loop...")
    return result_imgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-label', "--ground_truth_folder", help="ground truth folder")
    parser.add_argument('-syn', "--test_folder", help="test folder")
    args = parser.parse_args()

    image_col = io.imread_collection(args.test_folder + '*')
    # mask_col = io.imread_collection(args.ground_truth_folder + '*')
    # crf_orgim = io.imread_collection('C:/Users/prikha/Downloads/BA/Datasets/printed-hw-seg/printed-hw-seg_test/test/' + 'crf-orgim/*')
    out_raw = classify(image_col)
    print("end of classifying...")
    out_crf = [crf(inim, outim) for inim, outim in zip(image_col, out_raw)]
    print("after out_crf...")

    results_folder = 'D:/Uni/BA/Datasets/HTSNet_data_synthesis/eval/wgm-cvl_jottueset/Microfilm/'

    IoUs_hw_old = []
    IoUs_hw_new = []
    IoUs_printed_old = []
    IoUs_printed_new = []
    IoUs_bg_old = []
    IoUs_bg_new = []
    IoUs_mean_old = []
    IoUs_mean_new = []

    k = 0
    for i, (out, crf_res) in enumerate(zip(out_raw, out_crf)):
        print('inside of the 2nd for-loop...')
        # IoU_printed_old = get_IoU(getBinclassImg(1, rgb2mask(max_rgb_filter(out))), getBinclassImg(1, gt))
        # IoU_hw_old = get_IoU(getBinclassImg(2, rgb2mask(max_rgb_filter(out))), getBinclassImg(2, gt))
        # IoU_bg_old = get_IoU(getBinclassImg(3, rgb2mask(max_rgb_filter(out))), getBinclassImg(3, gt))
        # IoU_mean_old = np.array([IoU_printed_old, IoU_hw_old, IoU_bg_old]).mean()
        #
        # # try:
        # IoU_printed_new = get_IoU(getBinclassImg(1, crf_res), getBinclassImg(1, gt))
        # IoU_hw_new = get_IoU(getBinclassImg(2, crf_res), getBinclassImg(2, gt))
        # IoU_bg_new = get_IoU(getBinclassImg(3, crf_res), getBinclassImg(3, gt))
        # IoU_mean_new = np.array([IoU_printed_new, IoU_hw_new, IoU_bg_new]).mean()
        # # except ZeroDivisionError:
        # #     print('Cannot devide by zero.')
        #
        # IoUs_hw_old.append(IoU_hw_old)
        # IoUs_hw_new.append(IoU_hw_new)
        # IoUs_printed_new.append(IoU_printed_new)
        # IoUs_printed_old.append(IoU_printed_old)
        # IoUs_bg_old.append(IoU_bg_old)
        # IoUs_bg_new.append(IoU_bg_new)
        # IoUs_mean_old.append(IoU_mean_old)
        # IoUs_mean_new.append(IoU_mean_new)
        #
        # print("--------------- IoU test results for image " + str(i) + " ---------------")
        # print("Format:   <Class Name>  | [old IoU]-->[new IoU]")
        # print("           printed      | [{:.2f}]-->[{:.2f}]".format(IoU_printed_old, IoU_printed_new))
        # print("           handwritten  | [{:.2f}]-->[{:.2f}]".format(IoU_hw_old, IoU_hw_new))
        # print("           background   | [{:.2f}]-->[{:.2f}]".format(IoU_bg_old, IoU_bg_new))
        # print("-------------------------------------------------")
        # print("           mean         | [{:.2f}]-->[{:.2f}]".format(IoU_mean_old, IoU_mean_new))
        # print("\n")

        io.imsave(results_folder + 'fcn_out_' + str(k) + '.png', max_rgb_filter(out))
        print('saved fcn_out_' + str(i) + '.png')
        io.imsave(results_folder + 'fcn_out_crf_' + str(k) + '.png', mask2rgb(crf_res))
        print('saved fcn_out_crf_' + str(i) + '.png')

        print("\n")
        k = k + 1

    # IoUs_hw_old = np.array(IoUs_hw_old)
    # IoUs_hw_new = np.array(IoUs_hw_new)
    # IoUs_printed_old = np.array(IoUs_printed_old)
    # IoUs_printed_new = np.array(IoUs_printed_new)
    # IoUs_bg_old = np.array(IoUs_bg_old)
    # IoUs_bg_new = np.array(IoUs_bg_new)
    # IoUs_mean_old = np.array(IoUs_mean_old)
    # IoUs_mean_new = np.array(IoUs_mean_new)
    #
    # print('\n')
    # print("--------------- IoU mean test results ---------------")
    # print("Format:   <Class Name>  | [old IoU mean]-->[new IoU mean]")
    # print("           printed      | [{:.2f}]-->[{:.2f}]".format(IoUs_printed_old.mean(), IoUs_printed_new.mean()))
    # print("           handwritten  | [{:.2f}]-->[{:.2f}]".format(IoUs_hw_old.mean(), IoUs_hw_new.mean()))
    # print("           background   | [{:.2f}]-->[{:.2f}]".format(IoUs_bg_old.mean(), IoUs_bg_new.mean()))
    # print("-------------------------------------------------")
    # print("           mean         | [{:.2f}]-->[{:.2f}]".format(IoUs_mean_old.mean(), IoUs_mean_new.mean()))
    # print("\n")
