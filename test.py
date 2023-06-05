"""
reference:
    two sample t-test:
        https://mindscale.kr/course/basic-stat-python/16/
"""

import argparse
import torch
import segmentation_models_pytorch as smp
import scipy.stats

from tqdm import tqdm

from argument import arg_as_list
from dataset import load_data
from utility import extract_highest_probability_pixel, calculate_mse_predicted_to_annotation, calculate_angle
from visualization import angle_visualization, angle_graph


def get_pretrained_model(args, DEVICE):
    print("---------- Loading Model Pretrained ----------")

    model = smp.Unet(
        encoder_name    = 'resnet101', 
        encoder_weights = 'imagenet', 
        encoder_depth   = args.encoder_depth,
        classes         = args.output_channel, 
        activation      = 'sigmoid',
        decoder_channels= args.decoder_channel,
    )
    return model.to(DEVICE)


def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Torch is running on {DEVICE}')

    model = get_pretrained_model(args, DEVICE)

    experiment = 'Label6_D68-2_to2_AW-P1000+A_every10_chx2'
    # experiment = args.wandb_name
    print(experiment)
    path = f'./plot_results/{experiment}/results/{experiment}_best.pth'
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    _, val_loader = load_data(args)

    total_LDFA, total_MPTA, total_mHKA = 0, 0, 0
    LDFA_list, MPTA_list, mHKA_list = [], [], []
    predict_list = []
    angles = []
    for idx, (image, _, data_path, label_list) in enumerate(tqdm(val_loader)):
        # print(f"===== validation {idx} =====")
        image = image.to(device=DEVICE)

        with torch.no_grad():
            preds = model(image)
        index_list = extract_highest_probability_pixel(args, preds)
        
        LDFA   , MPTA   , mHKA    = calculate_angle(args, index_list, "preds")
        LDFA_GT, MPTA_GT, mHKA_GT = calculate_angle(args, label_list, "label")
        # print(f"LDFA   , MPTA   , mHKA   : {LDFA:.2f}, {MPTA:.2f}, {mHKA:.2f}")
        # print(f"LDFA_GT, MPTA_GT, mHKA_GT: {LDFA_GT:.2f}, {MPTA_GT:.2f}, {mHKA_GT:.2f}")
        # print(f"Difference: {LDFA-LDFA_GT:.2f}, {MPTA-MPTA_GT:.2f}, {mHKA-mHKA_GT:.2f}")
        
        total_LDFA += abs(LDFA-LDFA_GT)
        total_MPTA += abs(MPTA-MPTA_GT)
        total_mHKA += abs(mHKA-mHKA_GT)

        angles.append([LDFA, MPTA, mHKA, LDFA_GT, MPTA_GT, mHKA_GT])
        # angle_visualization(args, experiment, data_path, idx, 300, index_list, label_list, 0, angles[idx], "without label")
        # angle_visualization(args, experiment, data_path, idx, 300, index_list, label_list, 0, angles[idx], "with label")
    angle_graph(angles)

    # for idx, (image, _, data_path, label_list) in enumerate(tqdm(test_loader)):
    #     # print(f"===== test {idx} =====")
    #     image = image.to(device=DEVICE)

    #     with torch.no_grad():
    #         preds = model(image)
    #     index_list = extract_highest_probability_pixel(args, preds)
        
    #     LDFA   , MPTA   , mHKA    = calculate_angle(args, index_list, "preds")
    #     LDFA_GT, MPTA_GT, mHKA_GT = calculate_angle(args, label_list, "label")
    #     # print(f"LDFA   , MPTA   , mHKA   : {LDFA:.2f}, {MPTA:.2f}, {mHKA:.2f}")
    #     # print(f"LDFA_GT, MPTA_GT, mHKA_GT: {LDFA_GT:.2f}, {MPTA_GT:.2f}, {mHKA_GT:.2f}")
    #     # print(f"Difference: {LDFA-LDFA_GT:.2f}, {MPTA-MPTA_GT:.2f}, {mHKA-mHKA_GT:.2f}")
        
    #     total_LDFA += abs(LDFA-LDFA_GT)
    #     total_MPTA += abs(MPTA-MPTA_GT)
    #     total_mHKA += abs(mHKA-mHKA_GT)

    #     angles.append([LDFA, MPTA, mHKA, LDFA_GT, MPTA_GT, mHKA_GT])
    #     LDFA_list.append(LDFA)
    #     MPTA_list.append(MPTA)
    #     mHKA_list.append(mHKA)

    #     predict_list.append(LDFA)
    #     predict_list.append(MPTA)
    #     predict_list.append(mHKA)

    #     # angle_visualization(args, experiment, data_path, idx, 300, index_list, label_list, 0, angles[idx], "without label", "test")
    #     # angle_visualization(args, experiment, data_path, idx, 300, index_list, label_list, 0, angles[idx], "with label", "test")
    # # angle_graph(angles)
    # print(f"Average Difference: {total_LDFA/len(test_loader):.2f}, {total_MPTA/len(test_loader):.2f}, {total_mHKA/len(test_loader):.2f}")
    # print(f"Average Total: {(total_LDFA+total_MPTA+total_mHKA)/(len(test_loader)*3)}\n\n")

    # """
    # 1) t-test (two sample unpooled) between Single Pixel Segmentation errors(!) and your 
    # final model errors. This value will be absurdly low, so please express it scientific 
    # notation i.e. p<1e-15.
    # plot_results/J_D0/results/J_D0_best.pth
    # plot_results/j_Label6_D65-10_to5_AW_every50/results/j_Label6_D65-10_to5_AW_every50_best.pth
   
    # 2) Same t-test, between just the dilation version of your final model (Dilation 60) and 
    # your actual final model.
    # plot_results/J_D60/results/J_D60_best.pth
    
    # 3) Same t-test, between the Dilation + Adaptive weighting (Dilation 60 + Adapt W) and 
    # your actual model.
    # plot_results/J_D60_AW/results/J_D60_AW_best.pth
    # """

    # args.decoder_channel = [512,256,128,64,32]
    # model_paired = get_pretrained_model(args, DEVICE)

    # experiment = 'J_Label6_D68-2_to2_AW-P1000+A_every10_chx2'
    # # experiment = 'Label6_D68-2_to2_AW-P1000+A_every10_chx2'
    # # experiment = 'j_Label6_D65-10_to5_AW_every50'
    # print(experiment)
    # path = f'./plot_results/{experiment}/results/{experiment}_best.pth'
    # checkpoint = torch.load(path)

    # model_paired.load_state_dict(checkpoint['state_dict'])
    # model_paired.eval()

    # total_LDFA_paired, total_MPTA_paired, total_mHKA_paired = 0, 0, 0 
    # LDFA_list_paired, MPTA_list_paired, mHKA_list_paired = [], [], []
    # predict_list_paired = []
    # angles_paired = []
    # for idx, (image, _, data_path, label_list) in enumerate(tqdm(test_loader)):
    #     # print(f"===== test {idx} =====")
    #     image = image.to(device=DEVICE)

    #     with torch.no_grad():
    #         preds = model_paired(image)
    #     index_list = extract_highest_probability_pixel(args, preds)
        
    #     LDFA   , MPTA   , mHKA    = calculate_angle(args, index_list, "preds")
    #     LDFA_GT, MPTA_GT, mHKA_GT = calculate_angle(args, label_list, "label")
    #     # print(f"LDFA   , MPTA   , mHKA   : {LDFA:.2f}, {MPTA:.2f}, {mHKA:.2f}")
    #     # print(f"LDFA_GT, MPTA_GT, mHKA_GT: {LDFA_GT:.2f}, {MPTA_GT:.2f}, {mHKA_GT:.2f}")
    #     # print(f"Difference: {LDFA-LDFA_GT:.2f}, {MPTA-MPTA_GT:.2f}, {mHKA-mHKA_GT:.2f}")
        
    #     total_LDFA_paired += abs(LDFA-LDFA_GT)
    #     total_MPTA_paired += abs(MPTA-MPTA_GT)
    #     total_mHKA_paired += abs(mHKA-mHKA_GT)

    #     angles_paired.append([LDFA, MPTA, mHKA, LDFA_GT, MPTA_GT, mHKA_GT])
    #     LDFA_list_paired.append(LDFA)
    #     MPTA_list_paired.append(MPTA)
    #     mHKA_list_paired.append(mHKA)

    #     predict_list_paired.append(LDFA)
    #     predict_list_paired.append(MPTA)
    #     predict_list_paired.append(mHKA)

    #     # angle_visualization(args, experiment, data_path, idx, 300, index_list, label_list, 0, angles[idx], "without label", "test")
    #     # angle_visualization(args, experiment, data_path, idx, 300, index_list, label_list, 0, angles[idx], "with label", "test")
    # # angle_graph(angles)
    # print(f"Average Difference: {total_LDFA_paired/len(test_loader):.2f}, {total_MPTA_paired/len(test_loader):.2f}, {total_mHKA_paired/len(test_loader):.2f}")
    # print(f"Average Total: {(total_LDFA_paired+total_MPTA_paired+total_mHKA_paired)/(len(test_loader)*3)}\n\n")

    # LDFA_t_test = scipy.stats.ttest_ind(LDFA_list, LDFA_list_paired, equal_var=False)
    # MPTA_t_test = scipy.stats.ttest_ind(MPTA_list, MPTA_list_paired, equal_var=False)
    # mHKA_t_test = scipy.stats.ttest_ind(mHKA_list, mHKA_list_paired, equal_var=False)
    # pred_t_test = scipy.stats.ttest_ind(predict_list, predict_list_paired, equal_var=False)

    # print("LDFA :",LDFA_t_test)
    # print("MPTA :",MPTA_t_test)
    # print("mHKA :",mHKA_t_test)
    # print("Prediction :",pred_t_test)
    # print("\n\n==========================\n\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## test 
    parser.add_argument('--angle', action='store_true')
    parser.add_argument('--t_test', action='store_true')

    ## boolean arguments
    parser.add_argument('--data_preprocessing', action='store_true', help='whether to do data preprocessing or not')
    parser.add_argument('--pad_image', action='store_true', help='whether to pad the original image')
    parser.add_argument('--create_dataset', action='store_true', help='whether to create dataset or not')
    parser.add_argument('--multi_gpu', action='store_true', help='whether to use multiple gpus or not')
    parser.add_argument('--no_sigmoid', action='store_true', help='whether not to use sigmoid at the end of the model')
    parser.add_argument('--pixel_loss', action='store_true', help='whether to use only pixel loss')
    parser.add_argument('--geom_loss', action='store_true', help='whether to use only geometry loss')
    parser.add_argument('--angle_loss', action='store_true', help='whether to use only angular loss')
    parser.add_argument('--augmentation', action='store_true', help='whether to use augmentation')
    parser.add_argument('--patience', action='store_true', help='whether to stop when loss does not decrease')
    parser.add_argument('--progressive_erosion', action='store_true', help='whether to use progressive erosion')
    parser.add_argument('--progressive_weight', action='store_true', help='whether to use progressive weight')
    parser.add_argument('--pretrained', action='store_true', help='whether to pretrained model')
    parser.add_argument('--wandb', action='store_true', help='whether to use wandb or not')
    parser.add_argument('--wandb_sweep', action='store_true', help='whether to use wandb or not')
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
    parser.add_argument('--dataset_csv_path', type=str, default="./xlsx/train_dataset.csv", help='dataset excel file path')
    parser.add_argument('--test_dataset_csv_path', type=str, default="./xlsx/test_dataset.csv", help='dataset excel file path')
    parser.add_argument('--annotation_text_path', type=str, default="./data/annotation_text_files", help='annotation text file path')
    parser.add_argument('--annotation_text_name', type=str, default="annotation_label8.txt", help='annotation text file name')
    parser.add_argument('--test_annotation_text_name', type=str, default="annotation_label6_test.txt", help='annotation text file name')
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
    parser.add_argument("--decoder_channel", type=arg_as_list, default=[256,128,64,32,16], help='model decoder channels')
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--patience_threshold', type=int, default=10, help='early stopping threshold')
    parser.add_argument('--geom_loss_weight', type=int, default=1, help='weight of the loss function')
    parser.add_argument('--angle_loss_weight', type=int, default=1, help='weight of the loss function')

    ## hyperparameters - results
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for binary prediction')

    ## wandb
    parser.add_argument('--wandb_project', type=str, default="joint-replacement", help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default="yehyun-suh", help='wandb entity name')
    parser.add_argument('--wandb_name', type=str, default="temporary", help='wandb name')

    args = parser.parse_args()
    main(args)