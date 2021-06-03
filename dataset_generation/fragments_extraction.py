import json
from PIL import Image, ImageDraw
import cv2
import numpy as np

JSON_DATA_PATH = "C:/Users/prikha/Downloads/BA/Datasets/wgm-subset/output VoTT/output_photo_StAFF F 196-10_107/vott-json-export/wgm_photo_107-export.json"
JPG_ORIGIM_PATH = "C:/Users/prikha/Downloads/BA/Datasets/wgm-subset/output VoTT/output_photo_StAFF F 196-10_107/vott-json-export/"

IM_OUT_PATH_STAMP = 'C:/Users/prikha/Downloads/BA/Datasets/wgm-subset/output VoTT/output_photo_StAFF F 196-10_107/cropped_stamp/'
IM_OUT_PATH_SIGNATURE = 'C:/Users/prikha/Downloads/BA/Datasets/wgm-subset/output VoTT/output_photo_StAFF F 196-10_107/cropped_signature/'
IM_OUT_PATH_HANDWRITTEN = 'C:/Users/prikha/Downloads/BA/Datasets/wgm-subset/output VoTT/output_photo_StAFF F 196-10_107/cropped_handwritten/'
IM_OUT_PATH_MACHINEPRINTED = 'C:/Users/prikha/Downloads/BA/Datasets/wgm-subset/output VoTT/output_photo_StAFF F 196-10_107/cropped_machineprinted/'
IM_OUT_PATH_NOISE = 'C:/Users/prikha/Downloads/BA/Datasets/wgm-subset/output VoTT/output_photo_StAFF F 196-10_107/cropped_noise/'


def json2fragments(data):
    # print(json.dumps(data, indent=2))
    # print(len(data.keys()))
    # print(len(data))

    keys_arr = list(data.keys())
    for key in range(len(keys_arr)):
        name_im = data[keys_arr[key]]['asset']['name']
        # img = Image.open(JPG_ORIGIM_PATH + name_im)
        asset = data[keys_arr[key]]['regions']
        # print(asset)

        for region in asset:
            region_id = region['id']
            print('id: ', region_id)
            region_type = region['type']
            print("type: ", region_type)
            region_tag = region['tags'][0]
            print('tags: ', region_tag)
            region_points = region['points']
            print('points: ', region_points)

            # try:
            #     if tags == 'stamp':
            #         img2.save(IM_OUT_PATH_STAMP + id + '_cropped.jpg', 'JPEG')
            #     elif tags == 'signature':
            #         img2.save(IM_OUT_PATH_SIGNATURE + id + '_cropped.jpg', 'JPEG')
            #     elif tags == 'handwritten':
            #         img2.save(IM_OUT_PATH_HANDWRITTEN + id + '_cropped.jpg', 'JPEG')
            #     elif tags == 'machineprinted':
            #         img2.save(IM_OUT_PATH_MACHINEPRINTED + id + '_cropped.jpg', 'JPEG')
            #     elif tags == 'noise':
            #         img2.save(IM_OUT_PATH_NOISE + id + '_cropped.jpg', 'JPEG')
            # except SystemError:
            #     print("Trying to crop a region beyond the dimension of the image.")

            try:

                if region_type == 'POLYGON':
                    polygon_corners = []
                    # convert dict values to tuple ('x', 'y')
                    for i in range(len(region_points)):
                        # print(points[i])
                        d = region_points[i]
                        coord = tuple([d[field] for field in ['x', 'y']])
                        polygon_corners.append(coord)

                    # ------------------------------------------------------------------------
                    # with cv2 (with BLACK MASK)

                    # image = cv2.imread(JPG_ORIGIM_PATH + name_im, -1)
                    # mask = np.zeros(image.shape, dtype=np.uint8)
                    # #mask = np.zeros(image.shape[0:2], dtype=np.uint8)
                    # roi_corners = np.array([polygon_corners], dtype=np.int32)
                    #
                    # channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
                    # ignore_mask_color = (255,) * channel_count
                    # cv2.fillPoly(mask, roi_corners, ignore_mask_color)
                    # # from Masterfool: use cv2.fillConvexPoly if you know it's convex
                    #
                    # # apply the mask
                    # masked_image = cv2.bitwise_and(image, mask)
                    #
                    # # save the result
                    # cv2.imwrite(IM_OUT_PATH_STAMP + id + '_cropped.jpg', masked_image)

                    # with PIL (with WHITE MASK)
                    # # read image as RGB (without alpha)
                    # img = Image.open(JPG_ORIGIM_PATH + name_im).convert("RGB")
                    #
                    # # convert to numpy (for convenience)
                    # img_array = numpy.asarray(img)
                    #
                    # # create mask
                    # polygon = polygon_corners
                    #
                    # # create new image ("1-bit pixels, black and white", (width, height), "default color")
                    # mask_img = Image.new('1', (img_array.shape[1], img_array.shape[0]), 0)
                    #
                    # ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
                    # mask = numpy.array(mask_img)
                    #
                    # # assemble new image (uint8: 0-255)
                    # new_img_array = numpy.empty(img_array.shape, dtype='uint8')
                    #
                    # # copy color values (RGB)
                    # new_img_array[:, :, :3] = img_array[:, :, :3]
                    #
                    # # filtering image by mask
                    # new_img_array[:, :, 0] = new_img_array[:, :, 0] * mask
                    # new_img_array[:, :, 1] = new_img_array[:, :, 1] * mask
                    # new_img_array[:, :, 2] = new_img_array[:, :, 2] * mask
                    #
                    # # back to Image from numpy
                    # newIm = Image.fromarray(new_img_array, "RGB")
                    # newIm.save(IM_OUT_PATH_STAMP + id + '_cropped.jpg')

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
                    cv2.imwrite(IM_OUT_PATH_STAMP + region_id + '_cropped.jpg', dst)

                    if region_tag == 'stamp':
                        cv2.imwrite(IM_OUT_PATH_STAMP + region_id + '_cropped.jpg', dst)
                    elif region_tag == 'signature':
                        cv2.imwrite(IM_OUT_PATH_SIGNATURE + region_id + '_cropped.jpg', dst)
                    elif region_tag == 'noise':
                        cv2.imwrite(IM_OUT_PATH_NOISE + region_id + '_cropped.jpg', dst)

                else:
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
                    elif region_tag == 'noise':
                        img_cropped.save(IM_OUT_PATH_NOISE + region_id + '_cropped.jpg', 'JPEG')

                    print("------------------END OF A REGION----------------")

            except SystemError:
                print("Trying to crop a region beyond the dimension of the image.")


if __name__ == '__main__':
    with open(JSON_DATA_PATH) as json_file:
        output_VoTT = json.load(json_file)
        output_VoTT = output_VoTT['assets']
        json2fragments(output_VoTT)
