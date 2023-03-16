import torch
import pandas as pd
import numpy as np
import albumentations as A

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, df, args, transform=None):
        super().__init__()
        self.args = args
        self.df = df.reset_index()
        self.dataset_path = args.padded_image
        self.image_resize = args.image_resize
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        self.df = self.df.fillna(0)

        label_0_y, label_0_x = self.df['label_0_y'][idx], self.df['label_0_x'][idx]
        label_1_y, label_1_x = self.df['label_1_y'][idx], self.df['label_1_x'][idx]
        label_2_y, label_2_x = self.df['label_2_y'][idx], self.df['label_2_x'][idx]
        label_3_y, label_3_x = self.df['label_3_y'][idx], self.df['label_3_x'][idx]
        label_4_y, label_4_x = self.df['label_4_y'][idx], self.df['label_4_x'][idx]
        label_5_y, label_5_x = self.df['label_5_y'][idx], self.df['label_5_x'][idx]
        label_list = torch.Tensor([
            label_0_y, label_0_x, label_1_y, label_1_x, label_2_y, label_2_x,
            label_3_y, label_3_x, label_4_y, label_4_x, label_5_y, label_5_x,
        ])

        image_dir = self.df['image'][idx]
        image_path = f'{self.dataset_path}/{image_dir}'
        image = np.array(Image.open(image_path).convert("L"))
        image = self.transform(image=image)

        return image['image'], label_list


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
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(height=IMAGE_RESIZE, width=IMAGE_RESIZE),
        ToTensorV2(),
    ])

    train_dataset = CustomDataset(train_df, args, train_transform)
    val_dataset = CustomDataset(val_df, args, val_transform)
    print('len of train dataset: ', len(train_dataset))
    print('len of val dataset: ', len(val_dataset))

    num_workers = 4 * torch.cuda.device_count()
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=num_workers)

    print("---------- Loading Dataset Done ----------")

    return train_loader, val_loader