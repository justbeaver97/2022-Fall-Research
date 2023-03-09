"""
Reference:
    heatmap: 
        https://stackoverflow.com/questions/53467215/convert-pytorch-cuda-tensor-to-numpy-array
        https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap 
    tensor to image: 
        https://hello-bryan.tistory.com/429
    two tensor overlay:
        https://stackoverflow.com/questions/10640114/overlay-two-same-sized-images-in-python
        https://discuss.pytorch.org/t/create-heatmap-over-image/41408
    ValueError: pic should be 2/3 dimensional. Got 4 dimensions:
        https://stackoverflow.com/questions/64364239/pytorch-error-valueerror-pic-should-be-2-3-dimensional-got-4-dimensions
    1 channel to 3 channel:
        https://stackoverflow.com/questions/71957324/is-there-a-pytorch-transform-to-go-from-1-channel-data-to-3-channels    
    TypeError: Cannot handle this data type: (1, 1, 512), <f4:
        https://stackoverflow.com/questions/60138697/typeerror-cannot-handle-this-data-type-1-1-3-f4
    KeyError: ((1, 1, 512), '|u1'):
        https://stackoverflow.com/questions/57621092/keyerror-1-1-1280-u1-while-using-pils-image-fromarray-pil
    permute tensor:
        https://stackoverflow.com/questions/71880540/how-to-change-an-image-which-has-dimensions-512-512-3-to-a-tensor-of-size
    index of max value of tensor:
        https://stackoverflow.com/questions/71788996/how-can-i-find-multiple-maximum-indices-of-a-torch-tensor
    draw circle in image:
        https://dsbook.tistory.com/102
    >  - Expected Ptr<cv::UMat> for argument 'img':
        https://github.com/opencv/opencv/issues/18120
        cv2 functions expects numpy array
"""

import os
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import numpy as np
import cv2

from tqdm import tqdm
from PIL import Image
from sklearn.metrics import mean_squared_error as mse
from spatial_mean import SpatialMean_CHAN


def save_label_image(label_tensor, args):
    if args.delete_method == 'letter': num_channels = 7
    else:                              num_channels = 6

    for i in range(num_channels):
        plt.imshow(label_tensor[0][i].detach().cpu().numpy(), cmap='hot', interpolation='nearest')
        plt.savefig(f'./plot_results/{args.wandb_name}/annotation/label{i}.png')


def save_heatmap(preds, preds_binary, args, epoch):
    if args.only_pixel and (epoch % 10 == 0 or epoch % 50 == 49):
        for i in range(len(preds[0])):
            plt.imshow(preds[0][i].detach().cpu().numpy(), cmap='hot', interpolation='nearest')
            plt.savefig(f'./plot_results/{args.wandb_name}/label{i}/epoch_{epoch}_heatmap.png')
            torchvision.utils.save_image(preds_binary[0][i], f'./plot_results/{args.wandb_name}/label{i}/epoch_{epoch}.png')
    elif not args.only_pixel:
        for i in range(len(preds[0])):
            plt.imshow(preds[0][i].detach().cpu().numpy(), cmap='hot', interpolation='nearest')
            plt.savefig(f'./plot_results/{args.wandb_name}/label{i}/epoch_{epoch}_heatmap.png')
            torchvision.utils.save_image(preds_binary[0][i], f'./plot_results/{args.wandb_name}/epoch_{epoch}.png')


def save_overlaid_image(args, idx, predicted_label, data_path, highest_probability_pixels_list, epoch):
    image_path = f'{args.padded_image}/{data_path}'
    if args.delete_method == 'letter': num_channels = 7
    else:                              num_channels = 6

    for i in range(num_channels):
        original = Image.open(image_path).resize((512,512)).convert("RGB")
        background = predicted_label[0][i].unsqueeze(0)
        background = TF.to_pil_image(torch.cat((background, background, background), dim=0))
        overlaid_image = Image.blend(original, background , 0.3)
        overlaid_image.save(f'./plot_results/{args.wandb_name}/overlaid/label{i}/val{idx}_overlaid.png')
        overlaid_image.save(f'./plot_results/{args.wandb_name}/label{i}/epoch_{epoch}_overlaid.png')

        if i != 6:
            # x, y = int(highest_probability_pixels_list[i][0][0].detach().cpu()), int(highest_probability_pixels_list[i][0][1].detach().cpu())
            x, y = int(highest_probability_pixels_list[idx][0][i][0]), int(highest_probability_pixels_list[idx][0][i][1])
            pixel_overlaid_image = Image.fromarray(cv2.circle(np.array(original), (x,y), 15, (255, 0, 0),-1))
            pixel_overlaid_image.save(f'./plot_results/{args.wandb_name}/overlaid/label{i}/val{idx}_pixel_overlaid.png')


def save_predictions_as_images(args, loader, model, epoch, highest_probability_pixels_list, device="cuda"):
    model.eval()

    for idx, (image, label, data_path, _) in enumerate(tqdm(loader)):
        image = image.to(device=device)
        label = label.to(device=device)
        data_path = data_path[0]

        with torch.no_grad():
            if args.pretrained: preds = model(image)
            else:               preds = torch.sigmoid(model(image))
            preds_binary = (preds > args.threshold).float()
            # printsave(preds[0][0][0])
            # printsave(preds_binary[0][0][0])

        ## todo: record heatmaps even if the loss hasn't decreased
        if epoch == 0: 
            save_label_image(label, args)
        if idx == 0:
            save_heatmap(preds, preds_binary, args, epoch)
        if epoch % 10 == 0 or epoch % 50 == 49:
            save_overlaid_image(args, idx, preds_binary, data_path, highest_probability_pixels_list, epoch)

    model.train()


def calculate_mse_predicted_to_annotation(highest_probability_pixels, label_list):
    mse_value = 0
    # for i in range(len(index_list)):
    #     ## todo: if more than 1 for index_list[i]
    #     true_x, true_y = int(label_list[2*i+1]), int(label_list[2*i+0])
    #     # pred_x, pred_y = int(index_list[i][0][0].detach().cpu()), int(index_list[i][0][1].detach().cpu())
    #     pred_x, pred_y = int(index_list[0][i][0].detach().cpu()), int(index_list[0][i][1].detach().cpu())
    #     mse_value += mse([true_x, true_y],[pred_x, pred_y])

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
    return mse_value


def extract_highest_probability_pixel(args, prediction_tensor, label_list): 
    # if args.delete_method == 'letter': num_channels = 7
    # else:                              num_channels = 6
    index_list = []
    for i in range(6):
        index = (prediction_tensor[0][i] == torch.max(prediction_tensor[0][i])).nonzero()
        index_list.append(index)

    mse_value = calculate_mse_predicted_to_annotation(index_list, label_list)

    return index_list, mse_value


def check_accuracy(loader, model, args, epoch, device):
    print("=====Starting Validation=====")

    num_correct, num_pixels = 0, 0
    num_labels, num_labels_correct = 0, 0
    predict_as_label, prediction_correct  = 0, 0
    dice_score = 0
    highest_probability_pixels_list = []
    highest_probability_mse_total = 0
    model.eval()

    with torch.no_grad():
        for image, label, _, label_list in tqdm(loader):
            image = image.to(device)
            label = label.to(device)
            
            if args.pretrained: preds = model(image)
            else:               preds = torch.sigmoid(model(image))

            # ## extract the pixel with highest probability value
            # highest_probability_pixels, highest_probability_mse = extract_highest_probability_pixel(args, preds, label_list)

            ## extract pixel using spatial mean & calculating distance
            predict_spatial_mean_function = SpatialMean_CHAN(list(preds.shape[1:]))
            highest_probability_pixels    = predict_spatial_mean_function(preds)
            highest_probability_pixels_list.append(highest_probability_pixels.detach().cpu().numpy())
            highest_probability_mse       = calculate_mse_predicted_to_annotation(highest_probability_pixels, label_list)
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

    return model, label_accuracy, label_accuracy2, whole_image_accuracy, predict_as_label, dice, highest_probability_pixels_list, highest_probability_mse_total


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


def printsave(*a):
    file = open('tmp/error_log.txt','a')
    print(*a,file=file)

def calculate_number_of_dilated_pixel(k):
    sum = 0
    for i in range(k+1):
        if i == 0: sum += 1
        else:      sum += 4 * i
    return sum