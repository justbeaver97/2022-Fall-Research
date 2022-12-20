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


class CustomDataset(Dataset):
    def __init__(self, df, args, transform=None):
        super().__init__()
        self.df = df.reset_index()
        self.image_dir = self.df["image"]
        self.label_0_y, self.label_0_x = self.df['label_0_y'], self.df['label_0_x']
        self.label_1_y, self.label_1_x = self.df['label_1_y'], self.df['label_1_x']
        self.label_2_y, self.label_2_x = self.df['label_2_y'], self.df['label_2_x']
        self.label_3_y, self.label_3_x = self.df['label_3_y'], self.df['label_3_x']
        self.label_4_y, self.label_4_x = self.df['label_4_y'], self.df['label_4_x']
        self.label_5_y, self.label_5_x = self.df['label_5_y'], self.df['label_5_x']
        self.dataset_path = args.overlaid_image
        self.image_resize = args.image_resize
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_dir = self.image_dir[idx]
        label_0_y, label_0_x = self.label_0_y[idx], self.label_0_x[idx]
        label_1_y, label_1_x = self.label_1_y[idx], self.label_1_x[idx]
        label_2_y, label_2_x = self.label_2_y[idx], self.label_2_x[idx]
        label_3_y, label_3_x = self.label_3_y[idx], self.label_3_x[idx]
        label_4_y, label_4_x = self.label_4_y[idx], self.label_4_x[idx]
        label_5_y, label_5_x = self.label_5_y[idx], self.label_5_x[idx]

        image_path = f'{self.dataset_path}/{image_dir}'
        image = np.array(Image.open(image_path).convert("RGB"))

        # reference
        # https://albumentations.ai/docs/getting_started/mask_augmentation/
        # https://medium.com/pytorch/multi-target-in-albumentations-16a777e9006e
        mask0 = np.zeros([self.image_resize, self.image_resize])
        mask1 = np.zeros([self.image_resize, self.image_resize])
        mask2 = np.zeros([self.image_resize, self.image_resize])
        mask3 = np.zeros([self.image_resize, self.image_resize])
        mask4 = np.zeros([self.image_resize, self.image_resize])
        mask5 = np.zeros([self.image_resize, self.image_resize])
        mask0[label_0_y, label_0_x] = 1.0
        mask1[label_1_y, label_1_x] = 1.0
        mask2[label_2_y, label_2_x] = 1.0
        mask3[label_3_y, label_3_x] = 1.0
        mask4[label_4_y, label_4_x] = 1.0
        mask5[label_5_y, label_5_x] = 1.0

        if self.transform:
            augmentations = self.transform(
                image=image, masks=[mask0, mask1, mask2, mask3, mask4, mask5])
            image = augmentations["image"]
            mask0 = augmentations["masks"][0]
            mask1 = augmentations["masks"][1]
            mask2 = augmentations["masks"][2]
            mask3 = augmentations["masks"][3]
            mask4 = augmentations["masks"][4]
            mask5 = augmentations["masks"][5]

        # reference: https://sanghyu.tistory.com/85
        masks = torch.stack([mask0, mask1, mask2, mask3, mask4, mask5], dim=0)

        return image, masks


def load_data(args):
    print("---------- Starting Loading Dataset ----------")
    IMAGE_RESIZE = args.image_resize
    BATCH_SIZE = args.batch_size

    dataset_df = pd.read_csv(args.dataset_csv_path)
    split_point = int((len(dataset_df)*args.dataset_split)/10)
    train_df = dataset_df[:split_point]
    val_df = dataset_df[split_point:]

    train_transform = A.Compose([
        A.Resize(height=IMAGE_RESIZE, width=IMAGE_RESIZE),
        A.Rotate(limit=15, p=1.0),
        A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.1),
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

    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=BATCH_SIZE
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=1
    )

    print("---------- Loading Dataset Done ----------")

    return train_loader, val_loader

# def create_dataset(args):
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
    print("---------- Starting Creating Dataset ----------")
    annotation_file = f'{args.annotation_text_path}/{args.annotation_text_name}'

    label_coordinate_list = []
    with open(annotation_file, 'r') as f:
        for line in tqdm(f):
            if line.strip().split(',')[1] != '0':
                image_name = line.strip().split(',')[0]
                image_num = line.strip().split(',')[0].split('_')[0]
                image = cv2.imread(f'{args.padded_image}/{image_name}')
                resize_value = args.image_resize / image.shape[0]
                tmp = []

                for i in range(6):
                    y = int(line.strip().split(',')[(2*i)+2])
                    x = int(line.strip().split(',')[(2*i)+3])

                    # save the resized coordinates
                    tmp.append([round(y*resize_value), round(x*resize_value)])

                label_coordinate_list.append([
                    f'{image_num}_original.png',
                    tmp[0][0], tmp[0][1], tmp[1][0], tmp[1][1], tmp[2][0], tmp[2][1],
                    tmp[3][0], tmp[3][1], tmp[4][0], tmp[4][1], tmp[5][0], tmp[5][1],
                ])

    fields = ['image',
              'label_0_y', 'label_0_x', 'label_1_y', 'label_1_x', 'label_2_y', 'label_2_x',
              'label_3_y', 'label_3_x', 'label_4_y', 'label_4_x', 'label_5_y', 'label_5_x'
              ]

    with open(f'{args.dataset_csv_path}', 'w', newline='') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(label_coordinate_list)

    print("---------- Creating Dataset Done ----------\n")
