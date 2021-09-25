import json
from PIL import Image
import cv2
import numpy as np

# define a path of JSON data with tagged regions
JSON_DATA_PATH = "C:/Users/prikha/Downloads/BA/Datasets/wgm-subset/output VoTT/output_photo_StAFF F 196-10_110/vott-json-export/wgm_photo_110-export.json"
# define a path of original images
JPG_ORIGIM_PATH = "C:/Users/prikha/Downloads/BA/Datasets/wgm-subset/output VoTT/output_photo_StAFF F 196-10_110/vott-json-export/"

# define paths to store crops of regions labeled as stamp, signature, handwritten, machineprinted, mixed
IM_OUT_PATH_STAMP = 'C:/Users/prikha/Downloads/BA/Datasets/wgm-subset/output VoTT/output_photo_StAFF F 196-10_110/cropped_stamp/'
IM_OUT_PATH_SIGNATURE = 'C:/Users/prikha/Downloads/BA/Datasets/wgm-subset/output VoTT/output_photo_StAFF F 196-10_110/cropped_signature/'
IM_OUT_PATH_HANDWRITTEN = 'C:/Users/prikha/Downloads/BA/Datasets/wgm-subset/output VoTT/output_photo_StAFF F 196-10_110/cropped_handwritten/'
IM_OUT_PATH_MACHINEPRINTED = 'C:/Users/prikha/Downloads/BA/Datasets/wgm-subset/output VoTT/output_photo_StAFF F 196-10_110/cropped_machineprinted/'
IM_OUT_PATH_MIXED = 'C:/Users/prikha/Downloads/BA/Datasets/wgm-subset/output VoTT/output_photo_StAFF F 196-10_110/cropped_mixed/'


def json2crops(data):
    '''
    Takes a JSON file and saves crops of tagged regions into separate folders according to tags as JPEG image files.
    :param data: a JSON file
    '''

    data = data['assets']
    assets = list(data.keys())
    for asset in range(len(assets)):
        name_im = data[assets[asset]]['asset']['name']  # get an image name
        regions = data[assets[asset]]['regions']  # get an asset containing all tagged regions of an image

        for region in regions:

            region_id = region['id']
            print('id: ', region_id)

            region_type = region['type']
            print("type: ", region_type)

            region_tag = region['tags'][0]
            print('tags: ', region_tag)

            region_points = region['points']
            print('points: ', region_points)

            try:

                if region_type == 'POLYGON':
                    # ------------------------CROPPING POLYGON AREA FROM AN IMAGE------------------------

                    polygon_corners = []
                    # convert dict values to tuple ('x', 'y')
                    for i in range(len(region_points)):
                        d = region_points[i]
                        coord = tuple([d[field] for field in ['x', 'y']])
                        polygon_corners.append(coord)  # array with coordinates of polygon corners

                    # with cv2 (with WHITE MASK)
                    img = cv2.imread(JPG_ORIGIM_PATH + name_im)
                    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
                    # define a region of interest
                    roi_corners = np.array([polygon_corners], dtype=np.int32)
                    cv2.fillPoly(mask, roi_corners, 255)
                    # apply the mask
                    res = cv2.bitwise_and(img, img, mask=mask)
                    # create the white background of the same size of original image
                    wbg = np.ones_like(img, np.uint8) * 255
                    cv2.bitwise_not(wbg, wbg, mask=mask)
                    # overlap the resulted cropped image on the white background
                    dst = wbg + res

                    if region_tag == 'stamp':
                        cv2.imwrite(IM_OUT_PATH_STAMP + region_id + '_cropped.jpg', dst)
                    elif region_tag == 'signature':
                        cv2.imwrite(IM_OUT_PATH_SIGNATURE + region_id + '_cropped.jpg', dst)

                else:

                    # ------------------------CROPPING RECTANGULAR AREA FROM AN IMAGE------------------------

                    x_left, x_right = region_points[0]['x'], region_points[1]['x']
                    y_top, y_bottom = region_points[0]['y'], region_points[2]['y']

                    img = Image.open(JPG_ORIGIM_PATH + name_im)
                    img_cropped = img.crop((x_left, y_top, x_right, y_bottom))

                    if region_tag == 'stamp':
                        img_cropped.save(IM_OUT_PATH_STAMP + region_id + '_cropped.jpg', 'JPEG')
                    elif region_tag == 'signature':
                        img_cropped.save(IM_OUT_PATH_SIGNATURE + region_id + '_cropped.jpg', 'JPEG')
                    elif region_tag == 'handwritten':
                        img_cropped.save(IM_OUT_PATH_HANDWRITTEN + region_id + '_cropped.jpg', 'JPEG')
                    elif region_tag == 'machineprinted':
                        img_cropped.save(IM_OUT_PATH_MACHINEPRINTED + region_id + '_cropped.jpg', 'JPEG')
                    elif region_tag == 'mixed':
                        img_cropped.save(IM_OUT_PATH_MIXED + region_id + '_cropped.jpg', 'JPEG')

                    print("------------------END OF A REGION----------------")

            except SystemError:
                print("Trying to crop a region beyond the dimension of the image.")


if __name__ == '__main__':
    with open(JSON_DATA_PATH) as json_file:
        output_VoTT = json.load(json_file)
        json2crops(output_VoTT)
