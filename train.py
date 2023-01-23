"""
reference:
    train process: https://github.com/aladdinpersson/Machine-Learning-Collection
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from spatial_mean import SpatialMean_CHAN
from log import log_results, log_results_no_label
from utility import save_predictions_as_images, check_accuracy, create_directories


def train_function(args, DEVICE, model, loss_fn_pixel, loss_fn_geometry, optimizer, loader):
    loop = tqdm(loader)

    for batch_idx, (data, targets, _) in enumerate(loop):
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


def train(args, DEVICE, model, loss_fn_pixel, loss_fn_geometry, optimizer, train_loader, val_loader):
    count, pth_save_point, best_loss = 0, 0, np.inf
    create_directories(args, folder='./plot_results')
    
    for epoch in range(args.epochs):
        print(f"\nRunning Epoch # {epoch}")

        loss, loss_pixel, loss_geometry = train_function(
            args, DEVICE, model, loss_fn_pixel, loss_fn_geometry, optimizer, train_loader
        )
        label_accuracy, label_accuracy2, segmentation_accuracy, predict_as_label, dice_score = check_accuracy(
            val_loader, model, args, epoch, device=DEVICE
        )

        if args.wandb:
            if epoch % 25 == 0: 
                log_results(
                    args, loss, loss_pixel, loss_geometry, label_accuracy, label_accuracy2, segmentation_accuracy, predict_as_label, dice_score
                )
            else:               
                log_results_no_label(args, loss, loss_pixel, loss_geometry, segmentation_accuracy, dice_score)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":  optimizer.state_dict(),
        }

        if pth_save_point % 5 == 0: torch.save(checkpoint, f"./results/UNet_Epoch_{epoch}.pth")
        pth_save_point += 1

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