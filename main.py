import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import preprocess
from dataset import load_data, create_dataset
from model import get_model
from train import train
from log import initiate_wandb

def main(args):
    preprocess.customize_seed(args.seed)

    if args.wandb:
        initiate_wandb(args)

    ## data preprocessing
    if args.data_preprocessing:
        useful_dicom_list, original_annotation_list = preprocess.get_dataset(args)
        useful_dicom_path_list = preprocess.get_path(useful_dicom_list, args)
        
        preprocess.dicom2png(useful_dicom_path_list, args)
        preprocess.dicom2png_overlay(original_annotation_list, args)

    ## pad the original image & get annotation coordintaes
    if args.pad_image:
        preprocess.pad_original_image(args)
        ## annotation should be manually done by using ./create data/select_point.py

    ## create dataset from padded images & annotation text file
    if args.create_dataset: 
        create_dataset(args)

    ## load data into a form that can be fed into the model
    train_loader, val_loader = load_data(args)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Torch is running on {DEVICE}')

    ## load model & set loss function, optimizer, ...
    model = get_model(args, DEVICE)

    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    ## train model
    train(args, DEVICE, model, loss_fn, optimizer, scaler, train_loader, val_loader)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## get dataset
    parser.add_argument('--excel_path', type=str, default="./xlsx/dataset.xlsx", help='path to dataset excel file')

    ## data preprocessing
    parser.add_argument('--data_preprocessing', type=bool, default=False, help='whether to do data preprocessing or not')
    parser.add_argument('--dicom_data_path', type=str, default="./data/dicom_data", help='path to the dicom dataset')
    parser.add_argument('--dicom_to_png_path', type=str, default="./data/dicom_to_png", help='path to save dicom to png preprocessed data')
    parser.add_argument('--overlaid_image', type=str, default="./data/overlay_image_to_label", help='path to all the data from overlaying')
    parser.add_argument('--overlaid_image_only', type=str, default="./data/overlay_only", help='path to save overlaid data')
    parser.add_argument('--pad_image', type=bool, default=False, help='whether to pad the original image')
    parser.add_argument('--padded_image', type=str, default="./data/padded_image", help='path to save padded data')

    ## hyperparameters - data
    parser.add_argument('--create_dataset', type=bool, default=False, help='whether to create dataset or not')
    parser.add_argument('--dataset_path', type=str, default="./data/dataset", help='dataset path')
    parser.add_argument('--dataset_csv_path', type=str, default="./xlsx/dataset.csv", help='dataset excel file path')
    parser.add_argument('--annotation_text_path', type=str, default="./data/annotation_text_files", help='annotation text file path')
    parser.add_argument('--annotation_text_name', type=str, default="annotation.txt", help='annotation text file name')
    parser.add_argument('--dataset_split', type=int, default=9, help='dataset split ratio')

    parser.add_argument('--image_path', type=str, default="./overlay_only", help='path to save overlaid data')
    parser.add_argument('--image_resize', type=int, default=512, help='image resize value')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size')
    
    ## hyperparameters - model
    parser.add_argument('--seed', type=int, default=2022, help='seed customization for result reproduction')
    parser.add_argument('--input_channel', type=int, default=3, help='input channel size for UNet')
    parser.add_argument('--output_channel', type=int, default=1, help='output channel size for UNet')
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')

    ## hyperparameters - results

    ## wandb
    parser.add_argument('--wandb', type=bool, default=True, help='whether to use wandb or not')
    parser.add_argument('--wandb_project', type=str, default="joint-replacement", help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default="yehyun-suh", help='wandb entity name')
    parser.add_argument('--wandb_name', type=str, default="temporary", help='wandb name')

    args = parser.parse_args()
    main(args)