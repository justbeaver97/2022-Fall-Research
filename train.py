"""
reference:
    train process: https://github.com/aladdinpersson/Machine-Learning-Collection
"""

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from spatial_mean import SpatialMean_CHAN
from log import log_results, log_results_no_label
from utility import create_directories, calculate_number_of_dilated_pixel, extract_highest_probability_pixel, calculate_mse_predicted_to_annotation2
from visualization import save_predictions_as_images, box_plot
from dataset import load_data


def train_function(args, DEVICE, model, loss_fn_pixel, loss_fn_geometry, optimizer, loader):
    loop = tqdm(loader)

    for data, targets, _, _ in loop:
        data    = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        if args.pretrained: predictions = model(data)
        else:               predictions = torch.sigmoid(model(data))
        
        # calculate log loss with pixel value
        loss_pixel = loss_fn_pixel(predictions, targets)

        # calculate mse loss with spatial mean value
        predict_spatial_mean_function = SpatialMean_CHAN(list(predictions.shape[1:]))
        predict_spatial_mean          = predict_spatial_mean_function(predictions)
        targets_spatial_mean_function = SpatialMean_CHAN(list(targets.shape[1:]))
        targets_spatial_mean          = targets_spatial_mean_function(targets)
        loss_geometry                 = loss_fn_geometry(predict_spatial_mean, targets_spatial_mean)

        if args.only_pixel:  loss = loss_pixel
        elif args.only_geom: loss = loss_geometry
        else:                loss = args.loss_weight*loss_pixel + loss_geometry 

        ## backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    return loss.item(), loss_pixel.item(), loss_geometry.item()


def validate_function(loader, model, args, epoch, device):
    print("=====Starting Validation=====")
    model.eval()

    num_correct, num_pixels = 0, 0
    num_labels, num_labels_correct = 0, 0
    predict_as_label, prediction_correct  = 0, 0
    dice_score = 0
    highest_probability_pixels_list = []
    highest_probability_mse_total = 0
    mse_list = [[0]*len(loader) for _ in range(6)]

    with torch.no_grad():
        label_list_total = []
        for idx, (image, label, _, label_list) in enumerate(tqdm(loader)):
            image = image.to(device)
            label = label.to(device)
            label_list_total.append(label.detach().cpu().numpy())
            
            if args.pretrained: preds = model(image)
            else:               preds = torch.sigmoid(model(image))

            ## extract the pixel with highest probability value
            index_list = extract_highest_probability_pixel(preds)
            highest_probability_mse, mse_list = calculate_mse_predicted_to_annotation2(
                index_list, label_list, idx, mse_list
            )

            """
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
            """
            
            highest_probability_pixels_list.append(index_list)
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


def train(args, DEVICE, model, loss_fn_pixel, loss_fn_geometry, optimizer, train_loader, val_loader):
    count, pth_save_point, best_loss = 0, 0, np.inf
    create_directories(args, folder='./plot_results')
    
    for epoch in range(args.epochs):
        print(f"\nRunning Epoch # {epoch}")

        if epoch % args.dilation_epoch == 0:
            if args.progressive_erosion:
                ## TODO: Dilation cannot be 0
                train_loader, val_loader = load_data(args)
                if epoch != 0:
                    args.dilate = args.dilate - args.dilation_decrease

            if args.progressive_weight:
                image_size = args.image_resize * args.image_resize
                num_of_dil_pixels = calculate_number_of_dilated_pixel(args.dilate)
                w0 = (image_size * 100)/(image_size - num_of_dil_pixels)
                w1 = (image_size * 100)/(num_of_dil_pixels)
                weight = w1/w0
                print(f"Current weight for positive values is {weight}")
                loss_fn_pixel = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=DEVICE))

        loss, loss_pixel, loss_geometry = train_function(
            args, DEVICE, model, loss_fn_pixel, loss_fn_geometry, optimizer, train_loader
        )
        model, evaluation_list, highest_probability_pixels_list, highest_probability_mse_total, mse_list, label_list_total = validate_function(
            val_loader, model, args, epoch, device=DEVICE
        )

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":  optimizer.state_dict(),
        }

        # if pth_save_point % 5 == 0: torch.save(checkpoint, f"./results/UNet_Epoch_{epoch}.pth")
        # pth_save_point += 1

        print("Current loss ", loss)
        if best_loss > loss:
            print("=====New best model=====")
            # torch.save(checkpoint, f"./results/best.pth")
            # save_predictions_as_images(args, val_loader, model, epoch, highest_probability_pixels_list, device=DEVICE)
            best_loss, count = loss, 0
        else:
            count += 1

        ## For time efficiency, do not save visualized images
        if not args.no_image_save:
            ## For MIDL, always save visualization rather when having best loss
            save_predictions_as_images(args, val_loader, model, epoch, highest_probability_pixels_list, label_list_total, device=DEVICE)

        ## On the last epoch, save the model & create a box plot
        if epoch == args.epochs - 1:
            torch.save(checkpoint, f"./results/{args.wandb_name}.pth")
        if epoch % args.dilation_epoch == (args.dilation_epoch-1):
            box_plot(args, mse_list)

        if args.wandb:
            if epoch % 10 == 0 or epoch % 50 == 49: 
                log_results(
                    args, loss, loss_pixel, loss_geometry, evaluation_list, highest_probability_mse_total/18
                )
            else:               
                log_results_no_label(args, loss, loss_pixel, loss_geometry, evaluation_list, highest_probability_mse_total/18)

        # if count == args.patience:
        #     print("Early Stopping")
        #     break