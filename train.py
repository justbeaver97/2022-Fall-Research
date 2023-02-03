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
from utility import save_predictions_as_images, check_accuracy, create_directories
from dataset import load_data


def train_function(args, DEVICE, model, loss_fn_pixel, loss_fn_geometry, optimizer, loader):
    loop = tqdm(loader)

    for batch_idx, (data, targets, image_dir) in enumerate(loop):
        data    = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)
        # save_image(data, f'./tmp/tmp_image/{image_dir[0].split(".")[0]}.png')

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


def train(args, DEVICE, model, loss_fn_pixel, loss_fn_geometry, optimizer, train_loader, val_loader):
    count, pth_save_point, best_loss = 0, 0, np.inf
    create_directories(args, folder='./plot_results')
    
    for epoch in range(args.epochs):
        print(f"\nRunning Epoch # {epoch}")

        ## todo: recall dataloader
        if epoch % 50 == 0:
            if args.progressive_erosion:
                args.dilate = args.dilate - args.dilation_decrease
                train_loader, val_loader = load_data(args)

            if args.progressive_weight:
                ## todo: after reduce in the dilation iteration,
                ## change the weight in the loss function 80 - 5, 60 - 10, 40 - 10, 20 - 10, 10 - 30
                if args.dilate <= 15:   args.loss_class_weight = 30
                elif args.dilate <= 60: args.loss_class_weight = 10
                else:                     args.loss_class_weight = 5
                loss_fn_pixel = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.loss_class_weight], device=DEVICE))

        loss, loss_pixel, loss_geometry = train_function(
            args, DEVICE, model, loss_fn_pixel, loss_fn_geometry, optimizer, train_loader
        )
        label_accuracy, label_accuracy2, segmentation_accuracy, predict_as_label, dice_score = check_accuracy(
            val_loader, model, args, epoch, device=DEVICE
        )

        if args.wandb:
            if epoch % 10 == 0: 
                log_results(
                    args, loss, loss_pixel, loss_geometry, label_accuracy, label_accuracy2, segmentation_accuracy, predict_as_label, dice_score
                )
            else:               
                log_results_no_label(args, loss, loss_pixel, loss_geometry, segmentation_accuracy, dice_score)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":  optimizer.state_dict(),
        }

        # if pth_save_point % 5 == 0: torch.save(checkpoint, f"./results/UNet_Epoch_{epoch}.pth")
        # pth_save_point += 1

        print("Current loss ", loss)
        if best_loss > loss:
            print("=====New best model=====")
            torch.save(checkpoint, f"./results/best.pth")
            save_predictions_as_images(args, val_loader, model, epoch, device=DEVICE)
            best_loss, count = loss, 0
        else:
            count += 1

        # if count == args.patience:
        #     print("Early Stopping")
        #     break