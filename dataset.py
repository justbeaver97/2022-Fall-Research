import os 
import cv2
import csv
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
        self.label_dir = self.df['label']
        self.dataset_path = args.dataset_path
        self.transform = transform
        # self.image_transform = image_transform
        # self.label_transform = label_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_dir = self.image_dir[idx]
        label_dir = self.label_dir[idx]
        data_dir = os.path.join(
            self.dataset_path.split('/')[0],self.dataset_path.split('/')[1]
        )

        image_path = f'{data_dir}/{image_dir}'
        mask_path = f'{data_dir}/{label_dir}'
        # image = Image.open(image_path)
        # label = Image.open(label_path)
        # if self.image_transform:
        #     image = self.image_transform(image)
        # if self.label_transform:
        #     label = self.label_transform(label)

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # mask[mask == 255.0] = 1.0

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

def load_data(args):
    print("---------- Starting Loading Dataset ----------")
    IMAGE_RESIZE = args.image_resize
    BATCH_SIZE = args.batch_size

    dataset_df = pd.read_csv(args.dataset_csv_path, header=None, names=['image','label'])
    split_point = int((len(dataset_df)*args.dataset_split)/10)
    train_df = dataset_df[:split_point]
    val_df = dataset_df[split_point:]

    train_transform = A.Compose([
            A.Resize(height=IMAGE_RESIZE, width=IMAGE_RESIZE),
            A.Rotate(limit=15, p=1.0),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])
    val_transform = A.Compose([
            A.Resize(height=IMAGE_RESIZE, width=IMAGE_RESIZE),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
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
    with open(annotation_file,'r') as f:
        for line in tqdm(f):
            if line.strip().split(',')[1] != '0':
                image_name = line.strip().split(',')[0]
                image_num = line.strip().split(',')[0].split('_')[0]
                tmp = []
                image = cv2.imread(f'{args.padded_image}/{image_name}')
                resize_value = args.image_resize / image.shape[0]

                for i in range(6):
                    y = int(line.strip().split(',')[(2*i)+2])
                    x = int(line.strip().split(',')[(2*i)+3])

                    ## save the resized coordinates
                    tmp.append([round(y*resize_value),round(x*resize_value)])
                label_coordinate_list.append([
                    f'{image_num}_original.png', 
                    tmp[0][0],tmp[0][1],tmp[1][0],tmp[1][1],tmp[2][0],tmp[2][1],
                    tmp[3][0],tmp[3][1],tmp[4][0],tmp[4][1],tmp[5][0],tmp[5][1],
                ])
                # label_coordinate_list.append([
                #     f'{image_num}_original.png', 
                #     tmp[0],tmp[1],tmp[2],
                #     tmp[3],tmp[4],tmp[5],
                # ])

    # fields = ['image','label_0','label_1','label_2','label_3','label_4','label_5']
    fields = ['image',
        'label_0_y','label_0_x','label_1_y','label_1_x','label_2_y','label_2_x',
        'label_3_y','label_3_x','label_4_y','label_4_x','label_5_y','label_5_x'
    ]
    with open(f'{args.dataset_csv_path}', 'w',newline='') as f: 
        write = csv.writer(f) 
        write.writerow(fields) 
        write.writerows(label_coordinate_list)

    print("---------- Creating Dataset Done ----------\n")