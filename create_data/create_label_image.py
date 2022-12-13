import pprint
import os
import cv2
import numpy as np
from tqdm import tqdm

label_coordinate_list = []
with open("../data/annotation_text_files/221127_160821_overlay_only_copy.txt",'r') as f:
    for line in f:
        if line.strip().split(',')[1] != '0':
            image_name = line.strip().split(',')[0]
            y = line.strip().split(',')[-2]
            x = line.strip().split(',')[-1]
            label_coordinate_list.append([image_name, y, x])

# pprint.pprint(label_coordinate_list[:5])

if not os.path.exists('./dataset'): 
    os.mkdir('./dataset')
    os.mkdir('./dataset/image')
    os.mkdir('./dataset/annotation')

for i in tqdm(range(len(label_coordinate_list))):
    image_id = label_coordinate_list[i][0].split('_')[0]
    original_image_name = image_id + '_original.png'
    original_image_path = f'./all/{original_image_name}'
    y, x = int(label_coordinate_list[i][1]), int(label_coordinate_list[i][2])

    original_image = cv2.imread(original_image_path)
    # cv2.imshow(original_image_name, original_image)
    # cv2.waitKey(0)

    h, w, c = original_image.shape

    label_array = np.zeros((h, w))
    label_array[y][x] = 1
    # cv2.imshow('annotation', label_array)
    # cv2.waitKey(0)

    cv2.imwrite(f'./dataset/image/{image_id}.png',original_image)
    cv2.imwrite(f'./dataset/annotation/{image_id}.png',label_array)