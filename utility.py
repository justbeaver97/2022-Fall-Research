"""
Reference:
    index of max value of tensor:
        https://stackoverflow.com/questions/71788996/how-can-i-find-multiple-maximum-indices-of-a-torch-tensor
"""

import os
import torch
import numpy as np

from sklearn.metrics import mean_squared_error as mse


def calculate_mse_predicted_to_annotation(highest_probability_pixels, label_list, idx, mse_list):
    highest_probability_pixels = torch.Tensor(np.array(highest_probability_pixels)).squeeze(0).reshape(12,1)
    label_list = np.array(torch.Tensor(label_list), dtype=object).reshape(12,1)
    label_list = np.ndarray.tolist(label_list)

    ## squared=False for RMSE values
    mse_value = mse(highest_probability_pixels, label_list, squared=False) 
    for i in range(8):
        mse_list[i][idx] = mse(highest_probability_pixels[2*i:2*(i+1)]  ,label_list[2*i:2*(i+1)], squared=False)

    return mse_value, mse_list


def extract_highest_probability_pixel(prediction_tensor): 
    index_list = []
    for i in range(8):
        index = (prediction_tensor[0][i] == torch.max(prediction_tensor[0][i])).nonzero()
        if len(index) > 1:
            index = torch.Tensor([[sum(index)[0]/len(index), sum(index)[1]/len(index)]])
        index_list.append(index.detach().cpu().numpy())

    return index_list


def create_directories(args, folder='./plot_results'):
    if not args.delete_method: num_channels = 8
    else:                      num_channels = 7

    if not os.path.exists('./results'):
        os.mkdir(f'./results')
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