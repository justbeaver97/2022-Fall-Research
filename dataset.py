"""
reference:
    mask augmentation:
        https://albumentations.ai/docs/getting_started/mask_augmentation/
        https://medium.com/pytorch/multi-target-in-albumentations-16a777e9006e
    num_workers:
        https://jjeamin.github.io/posts/gpus/
        https://jjdeeplearning.tistory.com/32
    cv2.fillconvexpoly:
        https://deep-learning-study.tistory.com/105
        https://copycoding.tistory.com/150
    cv2.error: OpenCV(4.6.0) /io/opencv/modules/imgproc/src/drawing.cpp:2374: error: (-215:Assertion failed) points.checkVector(2, CV_32S) >= 0 in function 'fillConvexPoly'
        https://stackoverflow.com/questions/50376393/how-does-opencv-function-cv2-fillpoly-work
"""


import cv2
import csv
import torch
import pandas as pd
import numpy as np
import albumentations as A

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from scipy import ndimage


class CustomDataset(Dataset):
    def __init__(self, df, args, transform=None):
        super().__init__()
        self.args = args
        self.df = df.reset_index()
        self.dataset_path = args.padded_image
        self.image_resize = args.image_resize
        self.delete_method = args.delete_method
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        self.df = self.df.fillna(0)

        image_dir, num_of_pixels = self.df['image'][idx], self.df['data'][idx]
        label_0_y, label_0_x = self.df['label_0_y'][idx], self.df['label_0_x'][idx]
        label_1_y, label_1_x = self.df['label_1_y'][idx], self.df['label_1_x'][idx]
        label_2_y, label_2_x = self.df['label_2_y'][idx], self.df['label_2_x'][idx]
        label_3_y, label_3_x = self.df['label_3_y'][idx], self.df['label_3_x'][idx]
        label_4_y, label_4_x = self.df['label_4_y'][idx], self.df['label_4_x'][idx]
        label_5_y, label_5_x = self.df['label_5_y'][idx], self.df['label_5_x'][idx]
        label_list = [
            label_0_y, label_0_x, label_1_y, label_1_x, label_2_y, label_2_x,
            label_3_y, label_3_x, label_4_y, label_4_x, label_5_y, label_5_x,
        ]

        if self.delete_method == "letter":
            letter_y,  letter_x  = self.df['letter_y'][idx],  self.df['letter_x'][idx]
        elif self.delete_method == "box":
            box_0_y, box_0_x, box_1_y, box_1_x, box_2_y, box_2_x, box_3_y, box_3_x = 0,0,0,0,0,0,0,0
            box_4_y, box_4_x, box_5_y, box_5_x, box_6_y, box_6_x, box_7_y, box_7_x = 0,0,0,0,0,0,0,0
            box_8_y, box_8_x, box_9_y, box_9_x, box_10_y, box_10_x, box_11_y, box_11_x = 0,0,0,0,0,0,0,0
            box_12_y, box_12_x, box_13_y, box_13_x, box_14_y, box_14_x, box_15_y, box_15_x = 0,0,0,0,0,0,0,0
            if num_of_pixels >= 10:
                box_0_y, box_0_x = self.df['box_0_y'][idx], self.df['box_0_x'][idx]
                box_1_y, box_1_x = self.df['box_1_y'][idx], self.df['box_1_x'][idx]
                box_2_y, box_2_x = self.df['box_2_y'][idx], self.df['box_2_x'][idx]
                box_3_y, box_3_x = self.df['box_3_y'][idx], self.df['box_3_x'][idx]
            if num_of_pixels >= 14:
                box_4_y, box_4_x = self.df['box_4_y'][idx], self.df['box_4_x'][idx]
                box_5_y, box_5_x = self.df['box_5_y'][idx], self.df['box_5_x'][idx]
                box_6_y, box_6_x = self.df['box_6_y'][idx], self.df['box_6_x'][idx]
                box_7_y, box_7_x = self.df['box_7_y'][idx], self.df['box_7_x'][idx]
            if num_of_pixels >= 18:
                box_8_y, box_8_x = self.df['box_8_y'][idx], self.df['box_8_x'][idx]
                box_9_y, box_9_x = self.df['box_9_y'][idx], self.df['box_9_x'][idx]
                box_10_y, box_10_x = self.df['box_10_y'][idx], self.df['box_10_x'][idx]
                box_11_y, box_11_x = self.df['box_11_y'][idx], self.df['box_11_x'][idx]
            if num_of_pixels >= 22:
                box_12_y, box_12_x = self.df['box_12_y'][idx], self.df['box_12_x'][idx]
                box_13_y, box_13_x = self.df['box_13_y'][idx], self.df['box_13_x'][idx]
                box_14_y, box_14_x = self.df['box_14_y'][idx], self.df['box_14_x'][idx]
                box_15_y, box_15_x = self.df['box_15_y'][idx], self.df['box_15_x'][idx]
            
            box_list = [
                box_0_y, box_0_x, box_1_y, box_1_x, box_2_y, box_2_x, box_3_y, box_3_x,
                box_4_y, box_4_x, box_5_y, box_5_x, box_6_y, box_6_x, box_7_y, box_7_x,
                box_8_y, box_8_x, box_9_y, box_9_x, box_10_y, box_10_x, box_11_y, box_11_x,
                box_12_y, box_12_x, box_13_y, box_13_x, box_14_y, box_14_x, box_15_y, box_15_x
            ]

        image_path = f'{self.dataset_path}/{image_dir}'
        image = np.array(Image.open(image_path).convert("RGB"))

        mask0 = np.zeros([self.image_resize, self.image_resize])
        mask1 = np.zeros([self.image_resize, self.image_resize])
        mask2 = np.zeros([self.image_resize, self.image_resize])
        mask3 = np.zeros([self.image_resize, self.image_resize])
        mask4 = np.zeros([self.image_resize, self.image_resize])
        mask5 = np.zeros([self.image_resize, self.image_resize])
        if self.delete_method == "letter":
            mask6 = np.zeros([self.image_resize, self.image_resize])

        mask0 = dilate_pixel(mask0, label_0_y, label_0_x, self.args)
        mask1 = dilate_pixel(mask1, label_1_y, label_1_x, self.args)
        mask2 = dilate_pixel(mask2, label_2_y, label_2_x, self.args)
        mask3 = dilate_pixel(mask3, label_3_y, label_3_x, self.args)
        mask4 = dilate_pixel(mask4, label_4_y, label_4_x, self.args)
        mask5 = dilate_pixel(mask5, label_5_y, label_5_x, self.args)
        if self.delete_method == "letter":
            mask6 = dilate_pixel(mask6, letter_y, letter_x, self.args)

        if self.delete_method == "box":
            image = delete_unnecessary_boxes(image, box_list)

        if self.transform:
            if self.delete_method == "letter":
                augmentations = self.transform(image=image, masks=[mask0, mask1, mask2, mask3, mask4, mask5, mask6])
            else: 
                augmentations = self.transform(image=image, masks=[mask0, mask1, mask2, mask3, mask4, mask5])
            image = augmentations["image"]
            mask0 = augmentations["masks"][0]
            mask1 = augmentations["masks"][1]
            mask2 = augmentations["masks"][2]
            mask3 = augmentations["masks"][3]
            mask4 = augmentations["masks"][4]
            mask5 = augmentations["masks"][5]
            if self.delete_method == "letter":
                mask6 = augmentations["masks"][6]


        # reference: https://sanghyu.tistory.com/85
        if self.delete_method == "letter":
            masks = torch.stack([mask0, mask1, mask2, mask3, mask4, mask5, mask6], dim=0)
        else:
            masks = torch.stack([mask0, mask1, mask2, mask3, mask4, mask5], dim=0)

        return image, masks, image_dir, label_list


def dilate_pixel(mask, label_y, label_x, args):
    mask[label_y][label_x] = 1.0
    struct = ndimage.generate_binary_structure(2, 1)
    dilated_mask = ndimage.binary_dilation(mask, structure=struct, iterations=args.dilate).astype(mask.dtype)

    return dilated_mask


def delete_unnecessary_boxes(image, box_list):
    for i in range(len(box_list)//8):
        if box_list[(8*i)+2] == 0 and box_list[(8*i)+3] == 0:
            break
        else:
            ## because of y, x -> x, y flip
            pts = np.array([
                [box_list[(8*i)+1],box_list[(8*i)+0]],[box_list[(8*i)+3],box_list[(8*i)+2]],
                [box_list[(8*i)+5],box_list[(8*i)+4]],[box_list[(8*i)+7],box_list[(8*i)+6]]
            ],'int32')
            color = (0,0,0)
            image = cv2.fillConvexPoly(image, pts, color)

    ## todo: the boxes are moves towards right side a little bit. 
    return image


def load_data(args):
    print("---------- Starting Loading Dataset ----------")
    IMAGE_RESIZE = args.image_resize
    BATCH_SIZE = args.batch_size

    dataset_df = pd.read_csv(args.dataset_csv_path)
    split_point = int((len(dataset_df)*args.dataset_split)/10)
    train_df = dataset_df[:split_point]
    val_df = dataset_df[split_point:]

    ## other augmentations that I can try
    ## InvertImg, pixeldropout
    train_transform = A.Compose([
        A.Resize(height=IMAGE_RESIZE, width=IMAGE_RESIZE),
        # A.Rotate(limit=15, p=0.5),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(height=IMAGE_RESIZE, width=IMAGE_RESIZE),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    train_dataset = CustomDataset(
        train_df, args, train_transform
    )
    val_dataset = CustomDataset(
        val_df, args, val_transform
    )
    print('len of train dataset: ', len(train_dataset))
    print('len of val dataset: ', len(val_dataset))

    num_workers = 4 * torch.cuda.device_count()
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=1, num_workers=num_workers
    )

    print("---------- Loading Dataset Done ----------")

    return train_loader, val_loader


# def create_dataset(args):
#     """
#     Annotation Dataset Approach 1
#     After getting the annotation points from original image, 
#     make an annotation image that has exactly the same size as the original image
#     """
#     print("---------- Starting Creating Dataset ----------")

#     label_coordinate_list = []
#     with open(f"{args.annotation_text_path}/{args.annotation_text_name}",'r') as f:
#         for line in f:
#             if line.strip().split(',')[1] != '0':
#                 image_name = line.strip().split(',')[0]
#                 y = line.strip().split(',')[-2]
#                 x = line.strip().split(',')[-1]
#                 label_coordinate_list.append([image_name, y, x])

#     # pprint.pprint(label_coordinate_list[:5])

#     if not os.path.exists(f'{args.dataset_path}'):
#         os.mkdir(f'{args.dataset_path}')
#         os.mkdir(f'{args.dataset_path}/image')
#         os.mkdir(f'{args.dataset_path}/annotation')

#     for i in tqdm(range(len(label_coordinate_list))):
#         image_id = label_coordinate_list[i][0].split('_')[0]
#         original_image_name = image_id + '_original.png'
#         original_image_path = f'{args.overlaid_image}/{original_image_name}'
#         y, x = int(label_coordinate_list[i][1]), int(label_coordinate_list[i][2])

#         original_image = cv2.imread(original_image_path)
#         # cv2.imshow(original_image_name, original_image)
#         # cv2.waitKey(0)

#         h, w, c = original_image.shape

#         label_array = np.zeros((h, w))

#         directions_4 = [[0,1],[0,-1],[1,0],[-1,0],[0,0]]
#         directions_8 = [[0,1],[0,-1],[1,0],[-1,0],[1,1],[1,-1],[-1,-1],[-1,1],[0,0]]
#         directions_8_2 = [[0,2],[0,-2],[2,0],[-2,0],[1,1],[1,-1],[-1,-1],[-1,1],[0,0]]
#         for four in directions_4:
#             tmp_y, tmp_x = y + four[0], x+four[1]
#             for eight_2 in directions_8_2:
#                 tmp_y_2, tmp_x_2 = tmp_y + eight_2[0], tmp_x + eight_2[1]
#                 for eight in directions_8:
#                     label_array[tmp_y_2+eight[0]][tmp_x_2+eight[1]] = 1

#         # label_array[y][x] = 1
#         # cv2.imshow('annotation', label_array)
#         # cv2.waitKey(0)

#         cv2.imwrite(f'{args.dataset_path}/image/{image_id}.png',original_image)
#         cv2.imwrite(f'{args.dataset_path}/annotation/{image_id}.png',label_array)

#     print("---------- Creating Dataset Done ----------\n")


def create_dataset(args):
    """
    Annotation Dataset Approach 2
    After getting the annotation points from original image as text file, 
    create a csv file that resizes the values into resized image size
    """
    print("---------- Starting Creating Dataset ----------")
    if args.delete_method == "letter":
        annotation_file = f'{args.annotation_text_path}/annotation_label_letters.txt'
        num_of_labels = 7
    elif args.delete_method == "box":
        create_box_dataset(args)
        return 
    else:
        annotation_file = f'{args.annotation_text_path}/{args.annotation_text_name}'
        num_of_labels = 6
    
    label_coordinate_list = []
    with open(annotation_file, 'r') as f:
        for line in tqdm(f):
            if line.strip().split(',')[1] != '0':
                image_name = line.strip().split(',')[0]
                image_num = line.strip().split(',')[0].split('_')[0]
                num_of_pixels = int(line.strip().split(',')[1])

                image = cv2.imread(f'{args.padded_image}/{image_name}')
                resize_value = args.image_resize / image.shape[0]
                tmp = []

                for i in range(num_of_labels):
                    y = int(line.strip().split(',')[(2*i)+2])
                    x = int(line.strip().split(',')[(2*i)+3])

                    # save the resized coordinates
                    tmp.append([round(y*resize_value), round(x*resize_value)])

                if not args.delete_method:
                    label_coordinate_list.append([
                        f'{image_num}_pad.png',num_of_pixels,
                        tmp[0][0], tmp[0][1], tmp[1][0], tmp[1][1], tmp[2][0], tmp[2][1],
                        tmp[3][0], tmp[3][1], tmp[4][0], tmp[4][1], tmp[5][0], tmp[5][1]
                    ])
                elif args.delete_method=="letter":
                    label_coordinate_list.append([
                        f'{image_num}_pad.png',num_of_pixels,
                        tmp[0][0], tmp[0][1], tmp[1][0], tmp[1][1], tmp[2][0], tmp[2][1],
                        tmp[3][0], tmp[3][1], tmp[4][0], tmp[4][1], tmp[5][0], tmp[5][1],
                        tmp[6][0], tmp[6][1]
                    ])
    if args.delete_method=="letter":
        fields = ['image','data',
                'label_0_y', 'label_0_x', 'label_1_y', 'label_1_x', 'label_2_y', 'label_2_x',
                'label_3_y', 'label_3_x', 'label_4_y', 'label_4_x', 'label_5_y', 'label_5_x',
                'letter_y', 'letter_x'
                ]
    else:
        fields = ['image','data',
                'label_0_y', 'label_0_x', 'label_1_y', 'label_1_x', 'label_2_y', 'label_2_x',
                'label_3_y', 'label_3_x', 'label_4_y', 'label_4_x', 'label_5_y', 'label_5_x'
                ]
    

    with open(f'{args.dataset_csv_path}', 'w', newline='') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(label_coordinate_list)



    print("---------- Creating Dataset Done ----------\n")


def create_box_dataset(args):
    """
    Annotation Dataset Approach 2-1
    After getting the box points from original image as text file, 
    create a csv file that resizes the values into resized image size
    """
    print("---------- Starting Creating Box Dataset ----------")
    annotation_file = f'{args.annotation_text_path}/annotation_label_boxes.txt'

    label_coordinate_list = []
    with open(annotation_file, 'r') as f:
        for line in tqdm(f):
            if line.strip().split(',')[1] != '0':
                image_name = line.strip().split(',')[0]
                image_num = line.strip().split(',')[0].split('_')[0]
                num_of_pixels = int(line.strip().split(',')[1])

                image = cv2.imread(f'{args.padded_image}/{image_name}')
                resize_value = args.image_resize / image.shape[0]
                tmp = []

                for i in range(num_of_pixels):
                    y = int(line.strip().split(',')[(2*i)+2])
                    x = int(line.strip().split(',')[(2*i)+3])

                    # save the resized coordinates
                    if i < 6:
                        tmp.append([round(y*resize_value), round(x*resize_value)])
                    else:
                        tmp.append([y, x])

                if num_of_pixels == 10:
                    label_coordinate_list.append([
                        f'{image_num}_pad.png',num_of_pixels,
                        tmp[0][0], tmp[0][1], tmp[1][0], tmp[1][1], tmp[2][0], tmp[2][1],
                        tmp[3][0], tmp[3][1], tmp[4][0], tmp[4][1], tmp[5][0], tmp[5][1],
                        tmp[6][0], tmp[6][1], tmp[7][0], tmp[7][1], tmp[8][0], tmp[8][1], tmp[9][0], tmp[9][1], 
                    ])
                elif num_of_pixels == 14:
                    label_coordinate_list.append([
                        f'{image_num}_pad.png',num_of_pixels,
                        tmp[0][0], tmp[0][1], tmp[1][0], tmp[1][1], tmp[2][0], tmp[2][1],
                        tmp[3][0], tmp[3][1], tmp[4][0], tmp[4][1], tmp[5][0], tmp[5][1],
                        tmp[6][0], tmp[6][1], tmp[7][0], tmp[7][1], tmp[8][0], tmp[8][1], tmp[9][0], tmp[9][1], 
                        tmp[10][0], tmp[10][1], tmp[11][0], tmp[11][1], tmp[12][0], tmp[12][1], tmp[13][0], tmp[13][1],
                    ])
                elif num_of_pixels == 18:
                    label_coordinate_list.append([
                        f'{image_num}_pad.png',num_of_pixels,
                        tmp[0][0], tmp[0][1], tmp[1][0], tmp[1][1], tmp[2][0], tmp[2][1],
                        tmp[3][0], tmp[3][1], tmp[4][0], tmp[4][1], tmp[5][0], tmp[5][1],
                        tmp[6][0], tmp[6][1], tmp[7][0], tmp[7][1], tmp[8][0], tmp[8][1], tmp[9][0], tmp[9][1], 
                        tmp[10][0], tmp[10][1], tmp[11][0], tmp[11][1], tmp[12][0], tmp[12][1], tmp[13][0], tmp[13][1],
                        tmp[14][0], tmp[14][1], tmp[15][0], tmp[15][1], tmp[16][0], tmp[16][1], tmp[17][0], tmp[17][1]
                    ])
                elif num_of_pixels == 22:
                    label_coordinate_list.append([
                        f'{image_num}_pad.png',num_of_pixels,
                        tmp[0][0], tmp[0][1], tmp[1][0], tmp[1][1], tmp[2][0], tmp[2][1],
                        tmp[3][0], tmp[3][1], tmp[4][0], tmp[4][1], tmp[5][0], tmp[5][1],
                        tmp[6][0], tmp[6][1], tmp[7][0], tmp[7][1], tmp[8][0], tmp[8][1], tmp[9][0], tmp[9][1], 
                        tmp[10][0], tmp[10][1], tmp[11][0], tmp[11][1], tmp[12][0], tmp[12][1], tmp[13][0], tmp[13][1],
                        tmp[14][0], tmp[14][1], tmp[15][0], tmp[15][1], tmp[16][0], tmp[16][1], tmp[17][0], tmp[17][1],
                        tmp[18][0], tmp[18][1], tmp[19][0], tmp[19][1], tmp[20][0], tmp[20][1], tmp[21][0], tmp[21][1],
                    ])
                
    fields = ['image','data',
        'label_0_y', 'label_0_x', 'label_1_y', 'label_1_x', 'label_2_y', 'label_2_x',
        'label_3_y', 'label_3_x', 'label_4_y', 'label_4_x', 'label_5_y', 'label_5_x',
        'box_0_y', 'box_0_x', 'box_1_y', 'box_1_x', 'box_2_y', 'box_2_x', 'box_3_y', 'box_3_x', 
        'box_4_y', 'box_4_x', 'box_5_y', 'box_5_x', 'box_6_y', 'box_6_x', 'box_7_y', 'box_7_x',
        'box_8_y', 'box_8_x', 'box_9_y', 'box_9_x', 'box_10_y', 'box_10_x', 'box_11_y', 'box_11_x',
        'box_12_y', 'box_12_x', 'box_13_y', 'box_13_x', 'box_14_y', 'box_14_x', 'box_15_y', 'box_15_x',
    ]

    with open(f'./xlsx/dataset.csv', 'w', newline='') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(label_coordinate_list)

    print("---------- Creating Box Dataset Done ----------\n")