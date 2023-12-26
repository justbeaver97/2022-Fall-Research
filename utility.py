"""
Reference:
    index of max value of tensor:
        https://stackoverflow.com/questions/71788996/how-can-i-find-multiple-maximum-indices-of-a-torch-tensor
    calculation of angle:
        https://velog.io/@gyuho/Week2-Day6-Numpy-%EB%B2%A1%ED%84%B0-%ED%96%89%EB%A0%AC
"""

import os
import math
import torch
import numpy as np

from sklearn.metrics import mean_squared_error as mse


def compare_labels(preds, label, num_labels, num_labels_correct, predict_as_label, prediction_correct):
    for i in range(len(preds[0][0])):
        for j in range(len(preds[0][0][i])):
            if float(label[0][0][i][j]) == 1.0:
                num_labels += 1
                if float(preds[0][0][i][j]) == 1.0:
                    num_labels_correct += 1

            if float(preds[0][0][i][j]) == 1.0:
                predict_as_label += 1
                if float(label[0][0][i][j]) == 1.0:
                    prediction_correct += 1
    
    return num_labels, num_labels_correct, predict_as_label, prediction_correct


def calculate_mse_predicted_to_annotation(args, highest_probability_pixels, label_list, idx, mse_list):
    highest_probability_pixels = torch.Tensor(np.array(highest_probability_pixels)).squeeze(0).reshape(args.output_channel*2,1)
    label_list = np.array(torch.Tensor(label_list), dtype=object).reshape(args.output_channel*2,1)
    label_list = np.ndarray.tolist(label_list)

    ## squared=False for RMSE values
    mse_value = mse(highest_probability_pixels, label_list, squared=False) 
    for i in range(args.output_channel):
        mse_list[i][idx] = mse(highest_probability_pixels[2*i:2*(i+1)]  ,label_list[2*i:2*(i+1)], squared=False)

    return mse_value, mse_list


def extract_highest_probability_pixel(args, prediction_tensor): 
    index_list = []
    for i in range(args.output_channel):
        index = (prediction_tensor[0][i] == torch.max(prediction_tensor[0][i])).nonzero()
        if len(index) > 1:
            index = torch.Tensor([[sum(index)[0]/len(index), sum(index)[1]/len(index)]])
        index_list.append(index.detach().cpu().numpy())

    return index_list


def create_directories(args, folder='./plot_results'):
    num_channels = args.output_channel

    # if not os.path.exists('./results'):
    #     os.mkdir(f'./results')
    if not os.path.exists('./plot_data'):
        os.mkdir(f'./plot_data')
    if not os.path.exists(f'./{folder}'):
        os.mkdir(f'./{folder}')
    if not os.path.exists(f'{folder}/{args.wandb_name}'):
        os.mkdir(f'{folder}/{args.wandb_name}')
    if not os.path.exists(f'./plot_results/{args.wandb_name}/annotation'):
        os.mkdir(f'./plot_results/{args.wandb_name}/annotation')
    if not os.path.exists(f'./plot_results/{args.wandb_name}/overlaid'):
        os.mkdir(f'./plot_results/{args.wandb_name}/overlaid')
    if not os.path.exists(f'./plot_results/{args.wandb_name}/overlaid/all'):
        os.mkdir(f'./plot_results/{args.wandb_name}/overlaid/all')
    for i in range(num_channels):
        if not os.path.exists(f'./plot_results/{args.wandb_name}/overlaid/label{i}'):
            os.mkdir(f'./plot_results/{args.wandb_name}/overlaid/label{i}')
    for i in range(num_channels):
        if not os.path.exists(f'./plot_results/{args.wandb_name}/label{i}'):
            os.mkdir(f'./plot_results/{args.wandb_name}/label{i}')
    if not os.path.exists(f'./plot_results/{args.wandb_name}/results'):
        os.mkdir(f'./plot_results/{args.wandb_name}/results')
    if not os.path.exists(f'./plot_results/{args.wandb_name}/angles'):
        os.mkdir(f'./plot_results/{args.wandb_name}/angles')


def calculate_number_of_dilated_pixel(k):
    sum = 0
    for i in range(k+1):
        if i == 0: sum += 1
        else:      sum += 4 * i
    return sum


def innerProduct(v1, v2):
    EPSILON = 1e-8

    # θ = inner_product(x, y) / (L2(x) * L2(y))
    inner_product = np.dot(v1, v2)
    v1_L2_norm = np.linalg.norm(v1)
    v2_L2_norm = np.linalg.norm(v2)
    theta = inner_product / (v1_L2_norm * v2_L2_norm) 
    if theta < -1:
        theta = -1 + EPSILON
    if theta > 1:
        theta = 1 - EPSILON

    # radian value
    x = math.acos(theta)

    # pi value
    return math.degrees(x)
    

def lateral_distal_femoral_angle(medial_femur, upper_implant_left, upper_implant_center):
    """
    calculate angles between
    Medial Femur - Upper Implant Center / Upper Implant Center - Upper Implant Left
    """
    vector1 = [
        medial_femur[0]-upper_implant_center[0],
        medial_femur[1]-upper_implant_center[1]
    ]
    vector2 = [
        upper_implant_left[0]-upper_implant_center[0],
        upper_implant_left[1]-upper_implant_center[1]
    ]
    
    if vector1 == [0,0]:
        vector1 = [0.1, 0.1]
    if vector2 == [0,0]:
        vector2 = [0.1, 0.1]

    return innerProduct(vector1, vector2)


def medial_proximal_tibial_angle(lower_implant_left, lower_implant_center, medial_tibia):
    """
    calculate angles between
    Lower Implant Left - Lower Implant Center / Lower Implant Center - Medial Tibia
    """
    vector1 = [
        lower_implant_left[0]-lower_implant_center[0],
        lower_implant_left[1]-lower_implant_center[1]
    ]
    vector2 = [
        medial_tibia[0]-lower_implant_center[0],
        medial_tibia[1]-lower_implant_center[1]
    ]

    if vector1 == [0,0]:
        vector1 = [0.1, 0.1]
    if vector2 == [0,0]:
        vector2 = [0.1, 0.1]
    
    return innerProduct(vector1, vector2)


def mechanical_hip_knee_ankle_angle(medial_femur, upper_implant_center, medial_tibia):
    """
    calculate angles between
    Medial Femur - Upper Implant Center / Upper Implant Center - Medial Tibia
    """
    vector1 = [
        medial_femur[0]-upper_implant_center[0],
        medial_femur[1]-upper_implant_center[1]
    ]
    vector2 = [
        medial_tibia[0]-upper_implant_center[0],
        medial_tibia[1]-upper_implant_center[1]
    ]
    
    if vector1 == [0,0]:
        vector1 = [0.1, 0.1]
    if vector2 == [0,0]:
        vector2 = [0.1, 0.1]
    
    return innerProduct(vector1, vector2)


def calculate_angle(args, coordinates, method):
    """
    calculate angles between
    Lower Implant Left - Lower Implant Center / Lower Implant Center - Medial Tibia
    """
    if method == "preds":
        if args.output_channel == 6:
            medial_femur         = [coordinates[0][0][0],coordinates[0][0][1]]
            upper_implant_left   = [coordinates[1][0][0],coordinates[1][0][1]]
            upper_implant_center = [coordinates[2][0][0],coordinates[2][0][1]]
            lower_implant_left   = [coordinates[3][0][0],coordinates[3][0][1]]
            lower_implant_center = [coordinates[4][0][0],coordinates[4][0][1]]
            medial_tibia         = [coordinates[5][0][0],coordinates[5][0][1]]
        else: 
            medial_femur         = [coordinates[0][0][0],coordinates[0][0][1]]
            upper_implant_left   = [coordinates[1][0][0],coordinates[1][0][1]]
            upper_implant_center = [coordinates[3][0][0],coordinates[3][0][1]]
            lower_implant_left   = [coordinates[4][0][0],coordinates[4][0][1]]
            lower_implant_center = [coordinates[6][0][0],coordinates[6][0][1]]
            medial_tibia         = [coordinates[7][0][0],coordinates[7][0][1]]
    elif method == "label":
        if args.output_channel == 6:
            medial_femur         = [coordinates[0].numpy()[0], coordinates[1].numpy()[0]]
            upper_implant_left   = [coordinates[2].numpy()[0], coordinates[3].numpy()[0]]
            upper_implant_center = [coordinates[4].numpy()[0], coordinates[5].numpy()[0]]
            lower_implant_left   = [coordinates[6].numpy()[0], coordinates[7].numpy()[0]]
            lower_implant_center = [coordinates[8].numpy()[0], coordinates[9].numpy()[0]]
            medial_tibia         = [coordinates[10].numpy()[0], coordinates[11].numpy()[0]]
        else: 
            medial_femur         = [coordinates[0].numpy()[0], coordinates[1].numpy()[0]]
            upper_implant_left   = [coordinates[2].numpy()[0], coordinates[3].numpy()[0]]
            upper_implant_center = [coordinates[6].numpy()[0], coordinates[7].numpy()[0]]
            lower_implant_left   = [coordinates[8].numpy()[0], coordinates[9].numpy()[0]]
            lower_implant_center = [coordinates[12].numpy()[0], coordinates[13].numpy()[0]]
            medial_tibia         = [coordinates[14].numpy()[0], coordinates[15].numpy()[0]]

    # print(medial_femur, upper_implant_left, upper_implant_center,lower_implant_left, lower_implant_center, medial_tibia)
    LDFA = lateral_distal_femoral_angle(medial_femur, upper_implant_left, upper_implant_center)
    MPTA = medial_proximal_tibial_angle(lower_implant_left, lower_implant_center, medial_tibia)
    mHKA = mechanical_hip_knee_ankle_angle(medial_femur, upper_implant_center, medial_tibia)
    return LDFA, MPTA, 180-mHKA