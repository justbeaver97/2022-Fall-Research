import os 
import pandas as pd
import numpy as np
import albumentations as A

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df.reset_index()
        self.image_dir = self.df["image"]
        self.label_dir = self.df['label']
        self.transform = transform
        # self.image_transform = image_transform
        # self.label_transform = label_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_dir = self.image_dir[idx]
        label_dir = self.label_dir[idx]

        image_path = f'/content/gdrive/MyDrive/research/{image_dir}'
        mask_path = f'/content/gdrive/MyDrive/research/{label_dir}'
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
    IMAGE_HEIGHT = args.height
    IMAGE_WIDTH = args.width
    BATCH_SIZE = args.batch_size

    dataset_path = '/content/gdrive/MyDrive/research'
    csv_path = os.path.join(dataset_path,'dataset.csv') 
    dataset_df = pd.read_csv(csv_path, header=None, names=['image','label'])

    split_point = int((len(dataset_df)*9)/10)
    train_df = dataset_df[:split_point]
    val_df = dataset_df[split_point:]

    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=15, p=1.0),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    train_dataset = CustomDataset(
        train_df, train_transform
    )
    val_dataset = CustomDataset(
        val_df, val_transform
    )
    print('len of train dataset: ', len(train_dataset))
    print('len of val dataset: ', len(val_dataset))

    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=BATCH_SIZE
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=BATCH_SIZE
    )

    return train_loader, val_loader