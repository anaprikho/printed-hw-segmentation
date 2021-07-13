import glob
import cv2
import os
from os.path import splitext


# SOURCE = 'C:/Users/prikha/Downloads/BA/Datasets/HTSNet_data_synthesis/FINAL/VoTT_color_input_binarized_images_FINAL/model_input/evaluation/syn/'
SOURCE = 'C:/Users/prikha/Downloads/BA/Datasets/HTSNet_data_synthesis/FINAL/merged_model_input/test/syn_jpg/'

SAVE_DIR = 'C:/Users/prikha/Downloads/BA/Datasets/HTSNet_data_synthesis/FINAL/merged_model_input/test/syn/'


def main():
    for count, file in enumerate(os.listdir(SOURCE)):
        filename, extension = splitext(file)
        print(filename)
        # dst = "color" + str(count) + ".png"
        dst = filename + ".png"

        # # rename all the files
        os.rename(os.path.join(SOURCE, file), os.path.join(SAVE_DIR, dst))


# Driver Code
if __name__ == '__main__':
    main()
