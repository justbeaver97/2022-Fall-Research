"""
reference:
    train process: 
        https://github.com/aladdinpersson/Machine-Learning-Collection
"""

import torch
import torch.nn as nn
import numpy as np
import wandb

from tqdm import tqdm

from spatial_mean import SpatialMean_CHAN
from log import log_results, log_results_no_label
from utility import create_directories, calculate_number_of_dilated_pixel, extract_highest_probability_pixel, calculate_mse_predicted_to_annotation, calculate_angle, compare_labels
from visualization import save_predictions_as_images, box_plot, angle_visualization
from dataset import load_data


def train_function(args, DEVICE, model, loss_fn_pixel, loss_fn_geometry, loss_fn_angle, optimizer, loader):
    loop = tqdm(loader)
    total_loss, total_pixel_loss, total_geom_loss, total_angle_loss = 0, 0, 0, 0
    loss = None
    model.train()

    for data, targets, _, label_list in loop:
        data    = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        predictions =  model(data)

        # predictions =  torch.sigmoid(model(data))
        # if args.no_sigmoid:
        #     predictions_for_prob_pixel = model(data)
        # else:
        #     predictions_for_prob_pixel = torch.sigmoid(model(data))
        
        # calculate log loss with pixel value
        loss_pixel = loss_fn_pixel(predictions, targets)

        # calculate mse loss with spatial mean value
        predict_spatial_mean_function = SpatialMean_CHAN(list(predictions.shape[1:]))
        predict_spatial_mean          = predict_spatial_mean_function(predictions)
        targets_spatial_mean_function = SpatialMean_CHAN(list(targets.shape[1:]))
        targets_spatial_mean          = targets_spatial_mean_function(targets)
        loss_geometry                 = loss_fn_geometry(predict_spatial_mean, targets_spatial_mean)

        # # calculate the difference between GT angle and predicted angle
        # angle_pred, angle_gt = [], []
        # for i in range(len(predictions_for_prob_pixel)):
        #     index_list = extract_highest_probability_pixel(args, predictions_for_prob_pixel[i].unsqueeze(0))
        #     angle_pred.append([calculate_angle(args, index_list, "preds")])
        #     angle_gt.append([calculate_angle(args, label_list, "label")])
        # loss_angle = loss_fn_angle(torch.Tensor(angle_pred), torch.Tensor(angle_gt))

        # calculate the difference between GT angle and predicted angle
        angle_pred, angle_gt = [], []
        for i in range(len(predictions)):
            index_list = extract_highest_probability_pixel(args, predictions[i].unsqueeze(0))
            angle_pred.append([calculate_angle(args, index_list, "preds")])
            angle_gt.append([calculate_angle(args, label_list, "label")])
        loss_angle = loss_fn_angle(torch.Tensor(angle_pred), torch.Tensor(angle_gt))

        if args.pixel_loss:
            if args.angle_loss: 
                loss = args.angle_loss_weight*loss_pixel + loss_angle 
            else:
                loss = loss_pixel
        # if args.geom_loss:  
        #     loss = args.geom_loss_weight*loss_pixel + loss_geometry 

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        total_loss       += loss.item()
        total_pixel_loss += loss_pixel.item() 
        total_geom_loss  += loss_geometry.item()
        total_angle_loss += loss_angle.item()

    return loss.item(), loss_pixel.item(), loss_geometry.item(), loss_angle.item()


def validate_function(loader, model, args, epoch, device):
    print("=====Starting Validation=====")
    model.eval()

    num_correct, num_pixels = 0, 0
    num_labels, num_labels_correct = 0, 0
    predict_as_label, prediction_correct  = 0, 0
    dice_score = 0
    highest_probability_pixels_list = []
    highest_probability_mse_total = 0
    total_diff_LDFA, total_diff_MPTA, total_diff_mHKA = 0, 0, 0
    mse_list = [[0]*len(loader) for _ in range(args.output_channel)]

    with torch.no_grad():
        label_list_total, angles_total = [], []
        for idx, (image, label, data_path, label_list) in enumerate(tqdm(loader)):
            image = image.to(device)
            label = label.to(device)
            label_list_total.append(label.detach().cpu().numpy())
            
            if args.pretrained: preds = model(image)
            else:               preds = torch.sigmoid(model(image))

            ## extract the pixel with highest probability value
            index_list = extract_highest_probability_pixel(args, preds)
            highest_probability_mse, mse_list = calculate_mse_predicted_to_annotation(
                args, index_list, label_list, idx, mse_list
            )
            
            highest_probability_pixels_list.append(index_list)
            highest_probability_mse_total += highest_probability_mse

            ## calculate angles for evaluation
            LDFA   , MPTA   , mHKA    = calculate_angle(args, index_list, "preds")
            LDFA_GT, MPTA_GT, mHKA_GT = calculate_angle(args, label_list, "label")
            total_diff_LDFA += abs(LDFA_GT-LDFA)
            total_diff_MPTA += abs(MPTA_GT-MPTA)
            total_diff_mHKA += abs(mHKA_GT-mHKA)

            angles_total.append([LDFA, MPTA, mHKA, LDFA_GT, MPTA_GT, mHKA_GT])
            if idx == 0:
                angle_overlaid_image_w_label = angle_visualization(
                    args, args.wandb_name, data_path, idx, 300, index_list, label_list, 0, angles_total[idx], "without label"
                )
                angle_overlaid_image_wo_label = angle_visualization(
                    args, args.wandb_name, data_path, idx, 300, index_list, label_list, 0, angles_total[idx], "with label"
                )

            ## make predictions to be 0. or 1.
            preds = (preds > 0.5).float()

            # ## compare only labels
            # if (epoch % 10 == 0 or epoch % args.dilation_epoch == (args.dilation_epoch-1)) and args.wandb:
            #     if args.dilation_epoch >= 10:
            #         num_labels, num_labels_correct, predict_as_label, prediction_correct = compare_labels(
            #             preds, label, num_labels, num_labels_correct, predict_as_label, prediction_correct
            #         )
            #     else:
            #         if epoch % 10 > 3 and epoch % 10 < 7:
            #             num_labels, num_labels_correct, predict_as_label, prediction_correct = compare_labels(
            #                 preds, label, num_labels, num_labels_correct, predict_as_label, prediction_correct
            #             )

            # compare whole picture
            # num_correct += (preds == label).sum()
            # num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * label).sum()) / ((preds + label).sum() + 1e-8)

    label_accuracy, label_accuracy2 = 0, 0
    whole_image_accuracy = 0
    # whole_image_accuracy = num_correct/num_pixels*100
    dice = dice_score/len(loader)

    # if epoch % 10 == 0 or epoch % args.dilation_epoch == (args.dilation_epoch-1):
        # label_accuracy = (num_labels_correct/(num_labels+(1e-8))) * 100        ## from GT, how many of them were predicted
        # label_accuracy2 = (prediction_correct/(predict_as_label+(1e-8))) * 100 ## from prediction, how many of them were GT
        # print(f"Dice score: {dice}")
        # print(f"Number of pixels predicted as label: {predict_as_label}")
        # print(f"From Prediction: Got {prediction_correct}/{predict_as_label} with acc {label_accuracy2:.2f}")
        # print(f"From Ground Truth: Got {num_labels_correct}/{num_labels} with acc {label_accuracy:.2f}")
        
    # print(f"Got {num_correct}/{num_pixels} with acc {whole_image_accuracy:.2f}")
    print(f"Dice score: {dice}")
    print(f"Pixel to Pixel Distance: {highest_probability_mse_total/len(loader)}")
    print(f"Angular Difference: {[total_diff_LDFA/len(loader), total_diff_MPTA/len(loader), total_diff_mHKA/len(loader)]}")
    model.train()

    evaluation_list = [label_accuracy, label_accuracy2, whole_image_accuracy, predict_as_label, dice]
    angle_list = [total_diff_LDFA, total_diff_MPTA, total_diff_mHKA]
    return model, evaluation_list, highest_probability_pixels_list, highest_probability_mse_total, mse_list, label_list_total, angle_list, angle_overlaid_image_w_label


def train(
        args, DEVICE, model, loss_fn_pixel, loss_fn_geometry, loss_fn_angle,
        optimizer, train_loader, val_loader
    ):
    count, pth_save_point = 0, 0
    best_loss, best_angle_mean, best_rmse_mean = np.inf, 89.99, np.inf
    create_directories(args, folder='./plot_results')
    
    for epoch in range(args.epochs):
        print(f"\nRunning Epoch # {epoch}")

        if epoch % args.dilation_epoch == 0:
            if args.progressive_erosion:
                train_loader, val_loader, _ = load_data(args)
                if epoch != 0:
                    args.dilate = args.dilate - args.dilation_decrease
                if args.dilate < 1:
                    args.dilate = 0

            if args.progressive_weight:
                image_size = args.image_resize * args.image_resize
                num_of_dil_pixels = calculate_number_of_dilated_pixel(args.dilate)
                w0 = (image_size * 100)/(image_size - num_of_dil_pixels)
                w1 = (image_size * 100)/(num_of_dil_pixels)
                weight = w1/w0
                print(f"Current weight for positive values is {weight}")
                loss_fn_pixel = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=DEVICE))

        loss, loss_pixel, loss_geometry, loss_angle = train_function(
            args, DEVICE, model, loss_fn_pixel, loss_fn_geometry, loss_fn_angle, optimizer, train_loader
        )
        model, evaluation_list, highest_probability_pixels_list, highest_probability_mse_total, mse_list, label_list_total, angle_list, angle_overlaid_image = validate_function(
            val_loader, model, args, epoch, device=DEVICE
        )

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":  optimizer.state_dict(),
        }

        print("Current loss ", loss)
        if best_loss > loss:
            print("=====New best model=====")
            best_loss, count = loss, 0
        else:
            count += 1

        if not args.no_image_save:
            save_predictions_as_images(args, val_loader, model, epoch, highest_probability_pixels_list, label_list_total, device=DEVICE)
        
        # if pth_save_point % 5 == 0: 
        #     torch.save(checkpoint, f"./results/UNet_Epoch_{epoch}.pth")
        # pth_save_point += 1

        if sum(angle_list)/(len(val_loader)*3) < best_angle_mean:
            best_angle_mean = sum(angle_list)/(len(val_loader)*3)
            torch.save(checkpoint, f'./plot_results/{args.wandb_name}/results/{args.wandb_name}_best.pth')
        if epoch == args.epochs - 1:
            torch.save(checkpoint, f'./plot_results/{args.wandb_name}/results/{args.wandb_name}.pth')
            box_plot(args, mse_list)
            if args.no_image_save:
                save_predictions_as_images(args, val_loader, model, epoch, highest_probability_pixels_list, label_list_total, device=DEVICE)

        if highest_probability_mse_total/len(val_loader) < best_rmse_mean:
            best_rmse_mean = highest_probability_mse_total/len(val_loader)

        print(f'pixel loss: {loss_pixel}, geometry loss: {loss_geometry}, angle loss: {loss_angle}')
        print(f'best average rmse diff: {best_rmse_mean}, best average angle diff: {best_angle_mean}')

        if args.wandb:
            if epoch % 10 == 0 or epoch % args.dilation_epoch == (args.dilation_epoch-1): 
                log_results(
                    args, loss, loss_pixel, loss_geometry, loss_angle, 
                    evaluation_list, angle_list, best_angle_mean, 
                    highest_probability_mse_total, mse_list, best_rmse_mean, len(val_loader),
                    wandb.Image(angle_overlaid_image),
                )
            else:               
                log_results_no_label(
                    args, loss, loss_pixel, loss_geometry, loss_angle, 
                    evaluation_list, angle_list, best_angle_mean, 
                    highest_probability_mse_total, mse_list, best_rmse_mean, len(val_loader),
                )

        if args.patience and count == args.patience_threshold:
            print("Early Stopping")
            break