import os
import re
import json
import cv2
# some simple helper functions
def modify_label(item, new_label_value):
    if "label" in item:
        item["label"] = new_label_value

def find_unique_elements(input_sequence):
    unique_elements = set(input_sequence)
    unique_elements_list = list(unique_elements)
    return unique_elements_list

def split_elements_by_separator(input_list, separator=";"):
    split_elements_0 = [item.split(separator)[0] for item in input_list]
    split_elements_1 = [item.split(separator)[1] for item in input_list]
    return split_elements_0, split_elements_1

def remove_before_semicolon(input_string):
    semicolon_index = input_string.find(';')
    if semicolon_index != -1:
        return input_string[semicolon_index + 1:]
    else:
        return input_string
     

dir_path = './backupNew'
count_labels = 0
count_files  = 0
save_path = './cls_imgs'

list_labels = []
# converting all objects in to label class 1
for filename in os.listdir(dir_path):
    if filename.endswith('.json'):
        file_path = os.path.join(dir_path, filename)
        with open(file_path, 'r') as file:
            json_data = json.load(file)
            new_shapes = [item for item in json_data["shapes"] if item["label"] != ""]
            json_data["shapes"] = new_shapes
            
            img_name = filename.replace('json', 'jpg')
            main_img = cv2.imread(os.path.join(dir_path, img_name))
            
            image_height, image_width, _ = main_img.shape
            
            
            for cur_idx, item in enumerate(json_data["shapes"]):
                # print(item["label"])
                # item['label'] = "CIDA001"
                # old_val = item['label']
                # new_val = remove_before_semicolon(item['label'])
                # item['label'] = new_val
                # print(f'original label {old_val} replaced with {new_val}')
                # remove_empty_label_dicts()
                # read xs ys and the label
                
                list_labels.append(item['label'])
                cur_label = item["label"]
                cur_points = item["points"]
                
                assert len(cur_points) == 2
                cur_point1 = cur_points[0]
                cur_point2 = cur_points[1]
                x1 = int(cur_point1[0])
                x2 = int(cur_point2[0])
                y1 = int(cur_point1[1])
                y2 = int(cur_point2[1])

                newname = filename.removesuffix(".json",)
                newname = newname + f"_{cur_idx}" + f"_{cur_label}" + ".jpg"

                sub_img = main_img[y1:y2, x1:x2]
                save_file = os.path.join(save_path, newname)
                cv2.imwrite(save_file, sub_img)
                count_labels += 1
            
            if new_shapes != []:
                with open(file_path, 'w') as modified_file:
                    json.dump(json_data, modified_file, indent = 4)
                    
                count_files += 1
            else:        
                jpg_name = os.path.splitext(filename)[0] + '.jpg'
                jpg_path = os.path.join(dir_path, jpg_name)
                os.remove(file_path)
                os.remove(jpg_path)
                
                

print(f"we have {count_files} files, and a total number of {count_labels} objects!")
uniq_labels = find_unique_elements(list_labels)
print(f"for unique labels, we have a total of {len(uniq_labels)} unique labels: \n {uniq_labels}")


