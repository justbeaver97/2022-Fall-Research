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
        image_path = f'{self.dataset_path}/{image_dir}'
        image = np.array(Image.open(image_path).convert("RGB"))

        if self.args.output_channel == 6:
            label_0_y, label_0_x = self.df['label_0_y'][idx], self.df['label_0_x'][idx]
            label_1_y, label_1_x = self.df['label_1_y'][idx], self.df['label_1_x'][idx]
            label_2_y, label_2_x = self.df['label_3_y'][idx], self.df['label_3_x'][idx]
            label_3_y, label_3_x = self.df['label_4_y'][idx], self.df['label_4_x'][idx]
            label_4_y, label_4_x = self.df['label_6_y'][idx], self.df['label_6_x'][idx]
            label_5_y, label_5_x = self.df['label_7_y'][idx], self.df['label_7_x'][idx]
            label_list = [
                label_0_y, label_0_x, label_1_y, label_1_x, label_2_y, label_2_x,
                label_3_y, label_3_x, label_4_y, label_4_x, label_5_y, label_5_x,
            ]

            # label_0_y, label_0_x = self.df['label_0_y'][idx], self.df['label_0_x'][idx]
            # label_1_y, label_1_x = self.df['label_1_y'][idx], self.df['label_1_x'][idx]
            # label_2_y, label_2_x = self.df['label_2_y'][idx], self.df['label_2_x'][idx]
            # label_3_y, label_3_x = self.df['label_3_y'][idx], self.df['label_3_x'][idx]
            # label_4_y, label_4_x = self.df['label_4_y'][idx], self.df['label_4_x'][idx]
            # label_5_y, label_5_x = self.df['label_5_y'][idx], self.df['label_5_x'][idx]
            # label_list = [
            #     label_0_y, label_0_x, label_1_y, label_1_x, label_2_y, label_2_x,
            #     label_3_y, label_3_x, label_4_y, label_4_x, label_5_y, label_5_x,
            # ]

            mask0 = np.zeros([self.image_resize, self.image_resize])
            mask1 = np.zeros([self.image_resize, self.image_resize])
            mask2 = np.zeros([self.image_resize, self.image_resize])
            mask3 = np.zeros([self.image_resize, self.image_resize])
            mask4 = np.zeros([self.image_resize, self.image_resize])
            mask5 = np.zeros([self.image_resize, self.image_resize])

            mask0 = dilate_pixel(mask0, label_0_y, label_0_x, self.args)
            mask1 = dilate_pixel(mask1, label_1_y, label_1_x, self.args)
            mask2 = dilate_pixel(mask2, label_2_y, label_2_x, self.args)
            mask3 = dilate_pixel(mask3, label_3_y, label_3_x, self.args)
            mask4 = dilate_pixel(mask4, label_4_y, label_4_x, self.args)
            mask5 = dilate_pixel(mask5, label_5_y, label_5_x, self.args)

            if self.transform:
                augmentations = self.transform(image=image, masks=[
                    mask0, mask1, mask2, mask3, mask4, mask5
                ])
                
                image = augmentations["image"]
                mask0 = augmentations["masks"][0]
                mask1 = augmentations["masks"][1]
                mask2 = augmentations["masks"][2]
                mask3 = augmentations["masks"][3]
                mask4 = augmentations["masks"][4]
                mask5 = augmentations["masks"][5]

            masks = torch.stack([mask0, mask1, mask2, mask3, mask4, mask5], dim=0)

        # elif self.args.output_channel == 8:
        else:
            label_0_y, label_0_x = self.df['label_0_y'][idx], self.df['label_0_x'][idx]
            label_1_y, label_1_x = self.df['label_1_y'][idx], self.df['label_1_x'][idx]
            label_2_y, label_2_x = self.df['label_2_y'][idx], self.df['label_2_x'][idx]
            label_3_y, label_3_x = self.df['label_3_y'][idx], self.df['label_3_x'][idx]
            label_4_y, label_4_x = self.df['label_4_y'][idx], self.df['label_4_x'][idx]
            label_5_y, label_5_x = self.df['label_5_y'][idx], self.df['label_5_x'][idx]
            label_6_y, label_6_x = self.df['label_6_y'][idx], self.df['label_6_x'][idx]
            label_7_y, label_7_x = self.df['label_7_y'][idx], self.df['label_7_x'][idx]
            label_list = [
                label_0_y, label_0_x, label_1_y, label_1_x, label_2_y, label_2_x,
                label_3_y, label_3_x, label_4_y, label_4_x, label_5_y, label_5_x,
                label_6_y, label_6_x, label_7_y, label_7_x
            ]

            mask0 = np.zeros([self.image_resize, self.image_resize])
            mask1 = np.zeros([self.image_resize, self.image_resize])
            mask2 = np.zeros([self.image_resize, self.image_resize])
            mask3 = np.zeros([self.image_resize, self.image_resize])
            mask4 = np.zeros([self.image_resize, self.image_resize])
            mask5 = np.zeros([self.image_resize, self.image_resize])
            mask6 = np.zeros([self.image_resize, self.image_resize])
            mask7 = np.zeros([self.image_resize, self.image_resize])

            mask0 = dilate_pixel(mask0, label_0_y, label_0_x, self.args)
            mask1 = dilate_pixel(mask1, label_1_y, label_1_x, self.args)
            mask2 = dilate_pixel(mask2, label_2_y, label_2_x, self.args)
            mask3 = dilate_pixel(mask3, label_3_y, label_3_x, self.args)
            mask4 = dilate_pixel(mask4, label_4_y, label_4_x, self.args)
            mask5 = dilate_pixel(mask5, label_5_y, label_5_x, self.args)
            mask6 = dilate_pixel(mask6, label_6_y, label_6_x, self.args)
            mask7 = dilate_pixel(mask7, label_7_y, label_7_x, self.args)

            if self.transform:
                augmentations = self.transform(image=image, masks=[
                    mask0, mask1, mask2, mask3, mask4, mask5, mask6, mask7
                ])
                
                image = augmentations["image"]
                mask0 = augmentations["masks"][0]
                mask1 = augmentations["masks"][1]
                mask2 = augmentations["masks"][2]
                mask3 = augmentations["masks"][3]
                mask4 = augmentations["masks"][4]
                mask5 = augmentations["masks"][5]
                mask6 = augmentations["masks"][6]
                mask7 = augmentations["masks"][7]

            masks = torch.stack([mask0, mask1, mask2, mask3, mask4, mask5, mask6, mask7], dim=0)

        return image, masks, image_dir, label_list


def dilate_pixel(mask, label_y, label_x, args):
    if label_y == 512: label_y -= 1
    if label_x == 512: label_x -= 1
    mask[label_y][label_x] = 1.0
    struct = ndimage.generate_binary_structure(2, 1)
    dilated_mask = ndimage.binary_dilation(mask, structure=struct, iterations=args.dilate).astype(mask.dtype)

    return dilated_mask


def load_data(args):
    print("---------- Starting Loading Dataset ----------")
    IMAGE_RESIZE = args.image_resize
    BATCH_SIZE = args.batch_size

    train_val_df = pd.read_csv(args.dataset_csv_path)
    split_point = int((len(train_val_df)*args.dataset_split)/10)

    train_df = train_val_df[:split_point]
    val_df = train_val_df[split_point:]
    test_df = pd.read_csv(args.test_dataset_csv_path)

    if args.augmentation:
        train_transform = A.Compose([
            A.Resize(height=IMAGE_RESIZE, width=IMAGE_RESIZE),
            A.Rotate(limit=5, p=0.3),
            A.InvertImg(p=0.3),
            # A.HorizontalFlip(p=0.05),
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
    else:
        train_transform = A.Compose([
            A.Resize(height=IMAGE_RESIZE, width=IMAGE_RESIZE),
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
    test_dataset = CustomDataset(
        test_df, args, val_transform
    )
    print('len of train dataset: ', len(train_dataset))
    print('len of val dataset: ', len(val_dataset))
    print('len of test dataset: ', len(test_dataset))

    num_workers = 4 * torch.cuda.device_count()
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=1, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False, batch_size=1, num_workers=num_workers
    )

    print("---------- Loading Dataset Done ----------")

    return train_loader, val_loader, test_loader


def text_to_csv(args, annotation_file, file_type):    
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

                for i in range(args.output_channel):
                    y = int(line.strip().split(',')[(2*i)+2])
                    x = int(line.strip().split(',')[(2*i)+3])

                    # save the resized coordinates
                    tmp.append([round(y*resize_value), round(x*resize_value)])

                # if args.annotation_text_name == "annotation_label6.txt":
                if args.output_channel == 6:
                    label_coordinate_list.append([
                        # f'{image_num}_pad.png',num_of_pixels,
                        image_name,num_of_pixels,
                        tmp[0][0], tmp[0][1], tmp[1][0], tmp[1][1], tmp[2][0], tmp[2][1],
                        tmp[3][0], tmp[3][1], tmp[4][0], tmp[4][1], tmp[5][0], tmp[5][1],
                    ])
                # elif args.annotation_text_name == "annotation_label8.txt":
                else:
                    label_coordinate_list.append([
                        # f'{image_num}_pad.png',num_of_pixels,
                        image_name,num_of_pixels,
                        tmp[0][0], tmp[0][1], tmp[1][0], tmp[1][1], tmp[2][0], tmp[2][1],
                        tmp[3][0], tmp[3][1], tmp[4][0], tmp[4][1], tmp[5][0], tmp[5][1],
                        tmp[6][0], tmp[6][1], tmp[7][0], tmp[7][1]
                    ])

    # if args.annotation_text_name == "annotation_label6.txt":
    if args.output_channel == 6:
        fields = ['image','data',
                'label_0_y', 'label_0_x', 'label_1_y', 'label_1_x', 'label_2_y', 'label_2_x',
                'label_3_y', 'label_3_x', 'label_4_y', 'label_4_x', 'label_5_y', 'label_5_x',
        ]
    # elif args.annotation_text_name == "annotation_label8.txt":
    else:
        fields = ['image','data',
                'label_0_y', 'label_0_x', 'label_1_y', 'label_1_x', 'label_2_y', 'label_2_x',
                'label_3_y', 'label_3_x', 'label_4_y', 'label_4_x', 'label_5_y', 'label_5_x',
                'label_6_y', 'label_6_x', 'label_7_y', 'label_7_x'
        ]
    
    if file_type == "train":
        with open(f'{args.dataset_csv_path}', 'w', newline='') as f:
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(label_coordinate_list)
    else:
        with open(f'{args.test_dataset_csv_path}', 'w', newline='') as f:
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(label_coordinate_list)


def create_dataset(args):
    """
    Annotation Dataset Approach 2
    After getting the annotation points from original image as text file, 
    create a csv file that resizes the values into resized image size
    """
    print("---------- Starting Creating Dataset ----------")

    annotation_file = f'{args.annotation_text_path}/{args.annotation_text_name}'
    annotation_file_test = f'{args.annotation_text_path}/{args.test_annotation_text_name}'

    text_to_csv(args, annotation_file, "train")
    text_to_csv(args, annotation_file_test, "test")
    
    # label_coordinate_list = []
    # with open(annotation_file, 'r') as f:
    #     for line in tqdm(f):
    #         if line.strip().split(',')[1] != '0':
    #             image_name = line.strip().split(',')[0]
    #             image_num = line.strip().split(',')[0].split('_')[0]
    #             num_of_pixels = int(line.strip().split(',')[1])

    #             image = cv2.imread(f'{args.padded_image}/{image_name}')
    #             resize_value = args.image_resize / image.shape[0]
    #             tmp = []

    #             for i in range(num_of_labels):
    #                 y = int(line.strip().split(',')[(2*i)+2])
    #                 x = int(line.strip().split(',')[(2*i)+3])

    #                 # save the resized coordinates
    #                 tmp.append([round(y*resize_value), round(x*resize_value)])

    #             if args.annotation_text_name == "annotation_label6.txt":
    #                 label_coordinate_list.append([
    #                     f'{image_num}_pad.png',num_of_pixels,
    #                     tmp[0][0], tmp[0][1], tmp[1][0], tmp[1][1], tmp[2][0], tmp[2][1],
    #                     tmp[3][0], tmp[3][1], tmp[4][0], tmp[4][1], tmp[5][0], tmp[5][1],
    #                 ])
    #             elif args.annotation_text_name == "annotation_label8.txt":
    #                 label_coordinate_list.append([
    #                     f'{image_num}_pad.png',num_of_pixels,
    #                     tmp[0][0], tmp[0][1], tmp[1][0], tmp[1][1], tmp[2][0], tmp[2][1],
    #                     tmp[3][0], tmp[3][1], tmp[4][0], tmp[4][1], tmp[5][0], tmp[5][1],
    #                     tmp[6][0], tmp[6][1], tmp[7][0], tmp[7][1]
    #                 ])

    # if args.annotation_text_name == "annotation_label6.txt":
    #     fields = ['image','data',
    #             'label_0_y', 'label_0_x', 'label_1_y', 'label_1_x', 'label_2_y', 'label_2_x',
    #             'label_3_y', 'label_3_x', 'label_4_y', 'label_4_x', 'label_5_y', 'label_5_x',
    #     ]
    # elif args.annotation_text_name == "annotation_label8.txt":
    #     fields = ['image','data',
    #             'label_0_y', 'label_0_x', 'label_1_y', 'label_1_x', 'label_2_y', 'label_2_x',
    #             'label_3_y', 'label_3_x', 'label_4_y', 'label_4_x', 'label_5_y', 'label_5_x',
    #             'label_6_y', 'label_6_x', 'label_7_y', 'label_7_x'
    #     ]
    

    # with open(f'{args.dataset_csv_path}', 'w', newline='') as f:
    #     write = csv.writer(f)
    #     write.writerow(fields)
    #     write.writerows(label_coordinate_list)

    print("---------- Creating Dataset Done ----------\n")