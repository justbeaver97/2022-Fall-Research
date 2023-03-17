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
    highest_probability_pixels = highest_probability_pixels.squeeze(0).reshape(12,1).detach().cpu()
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


def extract_highest_probability_pixel(args, prediction_tensor, label_list, epoch): 
    # if args.delete_method == 'letter': num_channels = 7
    # else:                              num_channels = 6
    # index_list = []
    # for i in range(6):
    #     index = (prediction_tensor[0][i] == torch.max(prediction_tensor[0][i])).nonzero()
    #     index_list.append(index)

    # mse_value = calculate_mse_predicted_to_annotation(index_list, label_list)

    # return index_list, mse_value

    if epoch == 49:
        for i in range(len(prediction_tensor[0])):
            for j in range(len(prediction_tensor[0][i])):
                for k in range(len(prediction_tensor[0][i][j])):
                    if int(prediction_tensor[0][i][j][k]) == 1:
                        print(i,j,k)

    
    index_list = []
    for i in range(6):
        index = (prediction_tensor[0][i] == torch.max(prediction_tensor[0][i])).nonzero()
        index_list.append(index.detach().cpu().numpy())
    print(index_list)
    index_list = torch.Tensor(index_list)

    return index_list

def check_accuracy(loader, model, args, epoch, device):
    print("=====Starting Validation=====")

    num_correct, num_pixels = 0, 0
    num_labels, num_labels_correct = 0, 0
    predict_as_label, prediction_correct  = 0, 0
    dice_score = 0
    highest_probability_pixels_list = []
    highest_probability_mse_total = 0

    # if args.delete_method == 'letter': num_channels = 7
    # else:                              num_channels = 6
    mse_list = [[0]*len(loader) for _ in range(6)]

    model.eval()

    with torch.no_grad():
        label_list_total = []
        for idx, (image, label, _, label_list) in enumerate(tqdm(loader)):
            image = image.to(device)
            label = label.to(device)
            label_list_total.append(label.detach().cpu().numpy())
            
            if args.pretrained: preds = model(image)
            else:               preds = torch.sigmoid(model(image))

            # ## extract the pixel with highest probability value
            # highest_probability_pixels, highest_probability_mse = extract_highest_probability_pixel(args, preds, label_list)
            if epoch == 49:
                exit()

            if epoch % 10 == 5 or epoch == 49:
                index_list = extract_highest_probability_pixel(args, preds, label_list, epoch)
                print("index: ",index_list)
                print("label: ",label_list)
                a, b = calculate_mse_predicted_to_annotation(
                    index_list, label_list, idx, mse_list
                )
                print("highest_probability_mse: ", a)
                print("mse_list: ",b)

            ## extract pixel using spatial mean & calculating distance
            predict_spatial_mean_function = SpatialMean_CHAN(list(preds.shape[1:]))
            highest_probability_pixels    = predict_spatial_mean_function(preds)
            highest_probability_pixels_list.append(highest_probability_pixels.detach().cpu().numpy())
            # highest_probability_mse       = calculate_mse_predicted_to_annotation(
            #     highest_probability_pixels, label_list, _
            # )
            highest_probability_mse, mse_list       = calculate_mse_predicted_to_annotation(
                highest_probability_pixels, label_list, idx, mse_list
            )
            highest_probability_mse_total += highest_probability_mse

            ## make predictions to be 0. or 1.
            preds = (preds > 0.5).float()

            ## compare only labels
            if (epoch % 10 == 0 or epoch % 50 == 49) and args.wandb:
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

            # compare whole picture
            num_correct += (preds == label).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * label).sum()) / ((preds + label).sum() + 1e-8)

    label_accuracy, label_accuracy2 = 0, 0
    whole_image_accuracy = num_correct/num_pixels*100
    dice = dice_score/len(loader)

    if epoch % 10 == 0 or epoch % 50 == 49:
        label_accuracy = (num_labels_correct/(num_labels+(1e-8))) * 100        ## from GT, how many of them were predicted
        label_accuracy2 = (prediction_correct/(predict_as_label+(1e-8))) * 100 ## from prediction, how many of them were GT
        print(f"Number of pixels predicted as label: {predict_as_label}")
        print(f"From Prediction: Got {prediction_correct}/{predict_as_label} with acc {label_accuracy2:.2f}")
        print(f"From Ground Truth: Got {num_labels_correct}/{num_labels} with acc {label_accuracy:.2f}")
        
    print(f"Got {num_correct}/{num_pixels} with acc {whole_image_accuracy:.2f}")
    print(f"Dice score: {dice}")
    print(f"Pixel to Pixel Distance: {highest_probability_mse_total/len(loader)}")
    model.train()

    evaluation_list = [label_accuracy, label_accuracy2, whole_image_accuracy, predict_as_label, dice]
    return model, evaluation_list, highest_probability_pixels_list, highest_probability_mse_total, mse_list, label_list_total


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