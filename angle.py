import argparse
import torch
import segmentation_models_pytorch as smp

from tqdm import tqdm

from dataset import load_data
# from model import get_pretrained_model
from utility import extract_highest_probability_pixel, calculate_mse_predicted_to_annotation, calculate_angle


def get_pretrained_model(DEVICE):
    print("---------- Loading Model Pretrained ----------")

    ENCODER = 'resnet101'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation

    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=6, 
        activation=ACTIVATION,
    )
    return model.to(DEVICE)


def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Torch is running on {DEVICE}')

    model = get_pretrained_model(DEVICE)
    checkpoint = torch.load('./results/MIDL_D60-10_W_P_progressive_weighted_erosion_epoch300.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    _, val_loader = load_data(args)

    for idx, (image, _, _, label_list) in enumerate(val_loader):
        print(f"===== validation {idx} =====")
        image = image.to(device=DEVICE)

        with torch.no_grad():
            preds = model(image)

        mse_list = [[0]*len(val_loader) for _ in range(args.output_channel)]
        index_list = extract_highest_probability_pixel(args, preds)
        
        LDFA   , MPTA   , mHKA    = calculate_angle(args, index_list, "preds")
        LDFA_GT, MPTA_GT, mHKA_GT = calculate_angle(args, label_list, "label")
        print(f"LDFA   , MPTA   , mHKA   : {LDFA:.2f}, {MPTA:.2f}, {mHKA:.2f}")
        print(f"LDFA_GT, MPTA_GT, mHKA_GT: {LDFA_GT:.2f}, {MPTA_GT:.2f}, {mHKA_GT:.2f}")
        print(f"Difference: {LDFA-LDFA_GT:.2f}, {MPTA-MPTA_GT:.2f}, {mHKA-mHKA_GT:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## boolean arguments
    parser.add_argument('--data_preprocessing', action='store_true', help='whether to do data preprocessing or not')
    parser.add_argument('--pad_image', action='store_true', help='whether to pad the original image')
    parser.add_argument('--create_dataset', action='store_true', help='whether to create dataset or not')
    parser.add_argument('--multi_gpu', action='store_true', help='whether to use multiple gpus or not')
    parser.add_argument('--only_pixel', action='store_true', help='whether to use only pixel loss')
    parser.add_argument('--only_geom', action='store_true', help='whether to use only geometry loss')
    parser.add_argument('--patience', action='store_true', help='whether to stop when loss does not decrease')
    parser.add_argument('--progressive_erosion', action='store_true', help='whether to use progressive erosion')
    parser.add_argument('--progressive_weight', action='store_true', help='whether to use progressive weight')
    parser.add_argument('--pretrained', action='store_true', help='whether to pretrained model')
    parser.add_argument('--wandb', action='store_true', help='whether to use wandb or not')
    parser.add_argument('--no_image_save', action='store_true', help='whether to save image or not')

    ## get dataset
    parser.add_argument('--excel_path', type=str, default="./xlsx/dataset.xlsx", help='path to dataset excel file')

    ## data preprocessing
    parser.add_argument('--dicom_data_path', type=str, default="./data/dicom_data", help='path to the dicom dataset')
    parser.add_argument('--dicom_to_png_path', type=str, default="./data/dicom_to_png", help='path to save dicom to png preprocessed data')
    parser.add_argument('--overlaid_image', type=str, default="./data/overlay_image_to_label", help='path to all the data from overlaying')
    parser.add_argument('--overlaid_image_only', type=str, default="./data/overlay_only", help='path to save overlaid data')
    parser.add_argument('--overlaid_padded_image', type=str, default="./data/overlay_padded_image", help='path to save padded data')
    parser.add_argument('--padded_image', type=str, default="./data/padded_image", help='path to save padded data')

    ## hyperparameters - data
    parser.add_argument('--dataset_path', type=str, default="./data/dataset", help='dataset path')
    parser.add_argument('--dataset_csv_path', type=str, default="./xlsx/dataset.csv", help='dataset excel file path')
    parser.add_argument('--annotation_text_path', type=str, default="./data/annotation_text_files", help='annotation text file path')
    parser.add_argument('--annotation_text_name', type=str, default="annotation_label8.txt", help='annotation text file name')
    parser.add_argument('--dataset_split', type=int, default=9, help='dataset split ratio')
    parser.add_argument('--dilate', type=int, default=2, help='dilate iteration')
    parser.add_argument('--dilation_decrease', type=int, default=5, help='dilation decrease in progressive erosion')
    parser.add_argument('--dilation_epoch', type=int, default=50, help='dilation per epoch')
    parser.add_argument('--image_path', type=str, default="./overlay_only", help='path to save overlaid data')
    parser.add_argument('--image_resize', type=int, default=512, help='image resize value')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size')
    parser.add_argument('--delete_method', type=str, default="", help='how to delete unnecessary data in the xray images ["", "letter", "box"]')
    
    ## hyperparameters - model
    parser.add_argument('--seed', type=int, default=2022, help='seed customization for result reproduction')
    parser.add_argument('--input_channel', type=int, default=3, help='input channel size for UNet')
    parser.add_argument('--output_channel', type=int, default=6, help='output channel size for UNet')
    parser.add_argument('--encoder_depth', type=int, default=5, help='model depth for UNet')
    # parser.add_argument("--decoder_channel", type=arg_as_list, default=[256,128,64,32,16], help='model decoder channels')
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--patience_threshold', type=int, default=10, help='early stopping threshold')
    parser.add_argument('--loss_weight', type=int, default=1, help='weight of the loss function')

    ## hyperparameters - results
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for binary prediction')

    ## wandb
    parser.add_argument('--wandb_project', type=str, default="joint-replacement", help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default="yehyun-suh", help='wandb entity name')
    parser.add_argument('--wandb_name', type=str, default="temporary", help='wandb name')

    args = parser.parse_args()
    main(args)