"""
Reference:
    index of max value of tensor:
        https://stackoverflow.com/questions/71788996/how-can-i-find-multiple-maximum-indices-of-a-torch-tensor
"""

import os
import torch
import numpy as np

from tqdm import tqdm
from sklearn.metrics import mean_squared_error as mse
from spatial_mean import SpatialMean_CHAN


def calculate_mse_predicted_to_annotation(highest_probability_pixels, label_list, idx, mse_list):
    highest_probability_pixels = torch.Tensor(highest_probability_pixels).squeeze(0).reshape(12,1)
    label_list = np.array(torch.Tensor(label_list), dtype=object).reshape(12,1)
    label_list = np.ndarray.tolist(label_list)
    ordered_label_list = [
        label_list[1], label_list[0],
        label_list[3], label_list[2],
        label_list[5], label_list[4],
        label_list[7], label_list[6],
        label_list[9], label_list[8],
        label_list[11], label_list[10],
    ]
    mse_value = mse(highest_probability_pixels, ordered_label_list) 
    for i in range(6):
        # mse_list[i].append(mse(highest_probability_pixels[2*i:2*(i+1)]  ,ordered_label_list[2*i:2*(i+1)]))
        mse_list[i][idx] = mse(highest_probability_pixels[2*i:2*(i+1)]  ,ordered_label_list[2*i:2*(i+1)])

    # return mse_value
    return mse_value, mse_list


def calculate_mse_predicted_to_annotation2(highest_probability_pixels, label_list, idx, mse_list):
    highest_probability_pixels = torch.Tensor(np.array(highest_probability_pixels)).squeeze(0).reshape(12,1)
    label_list = np.array(torch.Tensor(label_list), dtype=object).reshape(12,1)
    label_list = np.ndarray.tolist(label_list)
    mse_value = mse(highest_probability_pixels, label_list) 
    for i in range(6):
        # mse_list[i].append(mse(highest_probability_pixels[2*i:2*(i+1)]  ,ordered_label_list[2*i:2*(i+1)]))
        mse_list[i][idx] = mse(highest_probability_pixels[2*i:2*(i+1)]  ,label_list[2*i:2*(i+1)])

    # return mse_value
    return mse_value, mse_list


def extract_highest_probability_pixel(prediction_tensor): 
    # if args.delete_method == 'letter': num_channels = 7
    # else:                              num_channels = 6
    # index_list = []
    # for i in range(6):
    #     index = (prediction_tensor[0][i] == torch.max(prediction_tensor[0][i])).nonzero()
    #     index_list.append(index)

    # mse_value = calculate_mse_predicted_to_annotation(index_list, label_list)

    # return index_list, mse_value

    # if epoch == 49:
    #     for i in range(len(prediction_tensor[0])):
    #         for j in range(len(prediction_tensor[0][i])):
    #             for k in range(len(prediction_tensor[0][i][j])):
    #                 if int(prediction_tensor[0][i][j][k]) == 1:
    #                     print(i,j,k)
    
    index_list = []
    for i in range(6):
        index = (prediction_tensor[0][i] == torch.max(prediction_tensor[0][i])).nonzero()
        if len(index) > 1:
            index = torch.Tensor([[sum(index)[0]/len(index), sum(index)[1]/len(index)]])
        index_list.append(index.detach().cpu().numpy())

    return index_list


def create_directories(args, folder='./plot_results'):
    if not args.delete_method: num_channels = 6
    else:                      num_channels = 7

    if not os.path.exists('./results'):
        os.mkdir(f'./results')
    if not os.path.exists(f'{folder}/{args.wandb_name}'):
        os.mkdir(f'{folder}/{args.wandb_name}')
    if not os.path.exists(f'./plot_results/{args.wandb_name}/annotation'):
        os.mkdir(f'./plot_results/{args.wandb_name}/annotation')
    if not os.path.exists(f'./plot_results/{args.wandb_name}/overlaid'):
        os.mkdir(f'./plot_results/{args.wandb_name}/overlaid')
    for i in range(num_channels):
        if not os.path.exists(f'./plot_results/{args.wandb_name}/overlaid/label{i}'):
            os.mkdir(f'./plot_results/{args.wandb_name}/overlaid/label{i}')
    for i in range(num_channels):
        if not os.path.exists(f'./plot_results/{args.wandb_name}/label{i}'):
            os.mkdir(f'./plot_results/{args.wandb_name}/label{i}')
    if not os.path.exists(f'./plot_results/{args.wandb_name}/results'):
        os.mkdir(f'./plot_results/{args.wandb_name}/results')


def calculate_number_of_dilated_pixel(k):
    sum = 0
    for i in range(k+1):
        if i == 0: sum += 1
        else:      sum += 4 * i
    return sum