import argparse
import pprint

from preprocessing import get_dataset, get_path, dicom2png, dicom2png_overlay

def main(args):
    useful_dicom_list, original_annotation_list = get_dataset()
    useful_dicom_path_list = get_path(useful_dicom_list, args)
    
    # dicom2png(useful_dicom_path_list, args)
    dicom2png_overlay(original_annotation_list, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## dicom2png
    parser.add_argument('--data_path', type=str, default="./data", help='path to the dicom dataset')
    parser.add_argument('--save_path', type=str, default="./preprocess", help='path to save dicom to png preprocessed data')

    ## dicom2png_overlay
    parser.add_argument('--save_everything_path', type=str, default="./all", help='path to all the data from function')
    parser.add_argument('--save_overlay_path', type=str, default="./overlay_only", help='path to save overlaid data')

    args = parser.parse_args()
    main(args)