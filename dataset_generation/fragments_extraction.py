import json
from PIL import Image

JSON_DATA_PATH = "C:/Users/prikha/Downloads/BA/Datasets/wgm-subset/output VoTT//output_photo_StAFF F 196-10_104/wgm_subset_export.json"
JPG_ORIGIM_PATH = "C:/Users/prikha/Downloads/BA/Datasets/wgm-subset/output VoTT/output_photo_StAFF F 196-10_104/vott-json-export/"

IM_OUT_PATH_STAMP = 'C:/Users/prikha/Downloads/BA/Datasets/wgm-subset/output VoTT/output_photo_StAFF F 196-10_104/cropped_stamp/'
IM_OUT_PATH_SIGNATURE = 'C:/Users/prikha/Downloads/BA/Datasets/wgm-subset/output VoTT/output_photo_StAFF F 196-10_104/cropped_signature/'
IM_OUT_PATH_HANDWRITTEN = 'C:/Users/prikha/Downloads/BA/Datasets/wgm-subset/output VoTT/output_photo_StAFF F 196-10_104/cropped_handwritten/'
IM_OUT_PATH_MACHINEPRINTED = 'C:/Users/prikha/Downloads/BA/Datasets/wgm-subset/output VoTT/output_photo_StAFF F 196-10_104/cropped_machineprinted/'
IM_OUT_PATH_NOISE = 'C:/Users/prikha/Downloads/BA/Datasets/wgm-subset/output VoTT/output_photo_StAFF F 196-10_104/cropped_noise/'


def json2fragments(data):

    #print(json.dumps(data, indent=2))
    #print(len(data.keys()))
    #print(len(data))

    keys_arr = list(data.keys())
    for key in range(len(keys_arr)):
        name_im = data[keys_arr[key]]['asset']['name']
        asset = data[keys_arr[key]]['regions'][0]
        print(asset)

        id = asset['id']
        print('id: ', id)
        #path = asset['path']
        #print('path: ', path)
        tags = asset['tags'][0]
        print('tags: ', tags)
        points = asset['points']
        print('points: ', points)
        print(len(points))
    #
        x_left, x_right = points[0]['x'], points[1]['x']
        y_top, y_bottom = points[0]['y'], points[2]['y']

        img = Image.open(JPG_ORIGIM_PATH + name_im)
        img2 = img.crop((x_left, y_top, x_right, y_bottom))

        if tags == 'stamp':
           img2.save(IM_OUT_PATH_STAMP + id + '_cropped.jpg', 'JPEG')
        elif tags == 'signature':
            img2.save(IM_OUT_PATH_SIGNATURE + id + '_cropped.jpg', 'JPEG')
        elif tags == 'handwritten':
            img2.save(IM_OUT_PATH_HANDWRITTEN + id + '_cropped.jpg', 'JPEG')
        elif tags == 'machineprinted':
            img2.save(IM_OUT_PATH_MACHINEPRINTED + id + '_cropped.jpg', 'JPEG')
        elif tags == 'noise':
            img2.save(IM_OUT_PATH_NOISE + id + '_cropped.jpg', 'JPEG')

        print("------------------END OF A LOOP----------------")


if __name__ == '__main__':

    with open(JSON_DATA_PATH) as json_file:
        data = json.load(json_file)
        data = data['assets']
        json2fragments(data)

