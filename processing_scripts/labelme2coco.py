# our desired format is as follows
# annotation{
#     "id": int, 
#     "image_id": int, 
#     "category_id": int, 
#     "segmentation": RLE or [polygon], 
#     "area": float, 
#     "bbox": [x,y,width,height], 
#     "iscrowd": 0 or 1,
# }

# categories[{
#     "id": int, 
#     "name": str, 
#     "supercategory": str,
#     }]


import os
import json
import numpy as np
import glob
import shutil
import cv2
from sklearn.model_selection import train_test_split

np.random.seed(41)

# classname_to_id = {
#     # our own class and IDs
#     "CIDA001":1
# }

classname_to_id = {
    # our own class and IDs
    "1":1,
    "3":2,
    "7":3,
    "8":4,
}


class Lableme2CoCo:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent=2 更加美观显示

    # json to coco
    def to_coco(self, json_path_list):
        self._init_categories()
        for json_path in json_path_list:
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']
            for shape in shapes:
                annotation = self._annotation(shape)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # const categories
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # image from coco
    def _image(self, obj, path):
        image = {}
        from labelme import utils
        img_x = utils.img_b64_to_arr(obj['imageData'])
        h, w = img_x.shape[:-1]
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace(".json", ".jpg")
        return image

    # anno from coco
    def _annotation(self, shape):
        # print('shape', shape)
        label = shape['label']
        points = shape['points']
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation

    # read json file data
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    # COCO format： [x1,y1,w,h] COCO bbox 
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]


if __name__ == '__main__':
    labelme_path = "./newBatch/"
    saved_coco_path = "./newBatch/"
    print('reading...')
    # create files
    if not os.path.exists("%scoco/annotations/" % saved_coco_path):
        os.makedirs("%scoco/annotations/" % saved_coco_path)
    if not os.path.exists("%scoco/images/train2017/" % saved_coco_path):
        os.makedirs("%scoco/images/train2017" % saved_coco_path)
    if not os.path.exists("%scoco/images/val2017/" % saved_coco_path):
        os.makedirs("%scoco/images/val2017" % saved_coco_path)
    # obtain all json files from image path
    print(labelme_path + "/*.json")
    json_list_path = glob.glob(labelme_path + "/*.json")
    print('json_list_path: ', len(json_list_path))
    # train test split, all images in images dir
    train_path, val_path = train_test_split(json_list_path, test_size=0.1, train_size=0.9)
    print("train_n:", len(train_path), 'val_n:', len(val_path))

    # train set all labelme convert to coco format
    l2c_train = Lableme2CoCo()
    train_instance = l2c_train.to_coco(train_path)
    l2c_train.save_coco_json(train_instance, '%scoco/annotations/instances_train2017.json' % saved_coco_path)
    for file in train_path:
        # shutil.copy(file.replace("json", "jpg"), "%scoco/images/train2017/" % saved_coco_path)
        img_name = file.replace('json', 'jpg')
        temp_img = cv2.imread(img_name)
        try:
            cv2.imwrite("{}coco/images/train2017/{}".format(saved_coco_path, img_name), temp_img)
        except Exception as e:
            print(e)
            print('Wrong Image:', img_name )
            continue
        print(img_name + '-->', img_name.replace('jpg', 'jpg'))

    for file in val_path:
        # shutil.copy(file.replace("json", "jpg"), "%scoco/images/val2017/" % saved_coco_path)
        img_name = file.replace('json', 'jpg')
        temp_img = cv2.imread(img_name)
        try:
            cv2.imwrite("{}coco/images/val2017/{}".format(saved_coco_path, img_name), temp_img)
        except Exception as e:
            print(e)
            print('Wrong Image:', img_name)
            continue
        print(img_name + '-->', img_name.replace('jpg', 'jpg'))

    # val set to coco format
    l2c_val = Lableme2CoCo()
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, '%scoco/annotations/instances_val2017.json' % saved_coco_path)