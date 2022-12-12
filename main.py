import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim

from preprocessing import get_dataset, get_path, dicom2png, dicom2png_overlay, customize_seed
from dataset import load_data
from model import get_model
from train import train

def main(args):
    ## initialize wandb
    wandb.init(project="joint-replacement", entity="yehyun-suh")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    customize_seed(args.seed)

    useful_dicom_list, original_annotation_list = get_dataset()
    useful_dicom_path_list = get_path(useful_dicom_list, args)
    
    # dicom2png(useful_dicom_path_list, args)
    dicom2png_overlay(original_annotation_list, args)

    ## load data into a form that can be fed into the model
    train_loader, val_loader = load_data(args)

    ## load model & set loss function, optimizer, ...
    model = get_model(args, DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    ## train model
    train(args, DEVICE, model, loss_fn, optimizer, scaler)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## dicom2png
    parser.add_argument('--data_path', type=str, default="./data", help='path to the dicom dataset')
    parser.add_argument('--save_path', type=str, default="./preprocess", help='path to save dicom to png preprocessed data')

    ## dicom2png_overlay
    parser.add_argument('--save_everything_path', type=str, default="./all", help='path to all the data from function')
    parser.add_argument('--save_overlay_path', type=str, default="./overlay_only", help='path to save overlaid data')

    ## hyperparameters - data
    parser.add_argument('--image_height', type=int, default=512, help='image resize height')
    parser.add_argument('--image_width', type=int, default=512, help='image resize width')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    
    ## hyperparameters - model
    parser.add_argument('--seed', type=int, default=2022, help='seed customization for result reproduction')
    parser.add_argument('--input_channel', type=int, default=3, help='input channel size for UNet')
    parser.add_argument('--output_channel', type=int, default=1, help='output channel size for UNet')
    parser.add_argument('--lr', '--learning_rate', type=int, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')

    ## hyperparameters - reuslt
    parser.add_argument('--pth', '--pth_path', type=str, default="./results", help='path to save pth file')

    args = parser.parse_args()
    main(args)