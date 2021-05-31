import json
from PIL import Image
import skimage.io as io

JSON_DATA_PATH = 'C:/Users/prikha/Downloads/BA/Datasets/wgm-subset/output VoTT/output_photo_StAFF F 196-10_104/0ec33ccc8c6a72fefa3d0cdf271757ee-asset.json'
JPG_INPUT_PATH = 'C:/Users/prikha/Downloads/BA/Datasets/wgm-subset/output VoTT/output_photo_StAFF F 196-10_104/StAF%20F%20196-1_104_0033.jpg'
IM_OUT_PATH = 'C:/Users/prikha/Downloads/BA/Datasets/wgm-subset/output VoTT/output_photo_StAFF F 196-10_104/cropped/'

with open(JSON_DATA_PATH) as json_file:
    data = json.load(json_file)
    data = data['regions']

#print(json.dumps(data, indent=2))

print(len(data))

for asset in range(0, 1):  #len(data)
    id = data[asset]['id']
    print('id: ', id)
    #path = data[asset]['path']
    #print('path: ', path)
    tags = data[asset]['tags']
    print('tags: ', tags)
    points = data[asset]['points']
    print('points: ', points)
    print(len(points))

    for point in points:
        print(point)

        img = Image.open(JPG_INPUT_PATH)
        img2 = img.crop((0, 0, 150, 150))
        img2.save(IM_OUT_PATH + id + '_cropped.jpg', 'JPEG')


