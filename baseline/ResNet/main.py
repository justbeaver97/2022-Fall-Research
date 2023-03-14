import timm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
import wandb

from dataset import load_data
from train import train

def customize_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def initiate_wandb(args):
    wandb.init(
        project=f"{args.wandb_project}", 
        entity=f"{args.wandb_entity}",
        name=f"{args.wandb_name}"
    )


def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Torch is running on {DEVICE}')

    customize_seed(args.seed)
    initiate_wandb(args)

    train_loader, val_loader = load_data(args)
    model = timm.create_model("resnet101", pretrained=True, num_classes=args.output_class)
    model.to(DEVICE)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(args, DEVICE, model, loss_fn, optimizer, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## data preprocessing
    parser.add_argument('--overlaid_image', type=str, default="../../data/overlay_image_to_label", help='path to all the data from overlaying')
    parser.add_argument('--overlaid_image_only', type=str, default="../../data/overlay_only", help='path to save overlaid data')
    parser.add_argument('--overlaid_padded_image', type=str, default="../../data/overlay_padded_image", help='path to save padded data')
    parser.add_argument('--padded_image', type=str, default="../../data/padded_image", help='path to save padded data')

    ## hyperparameters - data
    parser.add_argument('--dataset_path', type=str, default="../../data/dataset", help='dataset path')
    parser.add_argument('--dataset_csv_path', type=str, default="../../xlsx/dataset.csv", help='dataset excel file path')
    parser.add_argument('--dataset_split', type=int, default=9, help='dataset split ratio')
    parser.add_argument('--image_resize', type=int, default=512, help='image resize value')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    
    ## hyperparameters - model
    parser.add_argument('--seed', type=int, default=2022, help='seed customization for result reproduction')
    parser.add_argument('--output_class', type=int, default=12, help='output channel size for UNet')
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=250, help='number of epochs')

    ## wandb
    parser.add_argument('--wandb_project', type=str, default="joint-replacement", help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default="yehyun-suh", help='wandb entity name')
    parser.add_argument('--wandb_name', type=str, default="baseline_ResNet", help='wandb name')

    args = parser.parse_args()
    main(args)