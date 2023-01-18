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
from log import log_results 
from utility import save_predictions_as_images, check_accuracy


def train_function(args, DEVICE, model, loss_fn_pixel, loss_fn_geometry, optimizer, scaler, loader):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)
        # targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # # forward - with using cuda amp
        # with torch.cuda.amp.autocast():
        #     # predictions = torch.sigmoid(model(data))
        #     predictions = model(data)

        #     # calculate log loss with pixel value
        #     loss_pixel = loss_fn_pixel(predictions, targets)

        #     # calculate mse loss with spatial mean value
        #     # print(predictions.shape)
        #     predict_spatial_mean_function = SpatialMean_CHAN(list(predictions.shape[1:]))
        #     predict_spatial_mean = predict_spatial_mean_function(predictions)
        #     # print(predict_spatial_mean)
        #     targets_spatial_mean_function = SpatialMean_CHAN(list(targets.shape[1:]))
        #     targets_spatial_mean = targets_spatial_mean_function(targets)
        #     # print(targets_spatial_mean)
        #     loss_geometry = loss_fn_geometry(predict_spatial_mean, targets_spatial_mean)

        if args.pretrained:
            predictions = model(data)
        else:
            predictions = torch.sigmoid(model(data))
        
        # calculate log loss with pixel value
        loss_pixel = loss_fn_pixel(predictions, targets)

        # calculate mse loss with spatial mean value
        # print(predictions.shape)
        predict_spatial_mean_function = SpatialMean_CHAN(list(predictions.shape[1:]))
        predict_spatial_mean = predict_spatial_mean_function(predictions)
        # print(predict_spatial_mean)
        targets_spatial_mean_function = SpatialMean_CHAN(list(targets.shape[1:]))
        targets_spatial_mean = targets_spatial_mean_function(targets)
        # print(targets_spatial_mean)
        loss_geometry = loss_fn_geometry(predict_spatial_mean, targets_spatial_mean)

        if args.only_pixel:
            loss = loss_pixel
        elif args.only_geom:
            loss = loss_geometry
        else:
            loss = args.loss_weight*loss_pixel + loss_geometry 

        # print("prediction spatial mean\n",predict_spatial_mean)
        # print("targets spatial mean \n",targets_spatial_mean)

        optimizer.zero_grad()

        # backward - using scaler
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        ## backward - not using scaler
        loss.backward()
        optimizer.step()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    return loss.item(), loss_pixel.item(), loss_geometry.item()


def train(args, DEVICE, model, loss_fn_pixel, loss_fn_geometry, optimizer, scaler, train_loader, val_loader):
    count, pth_save_point, best_loss = 0, 0, np.inf

    if not os.path.exists('./results'):
        os.mkdir(f'./results')

    for epoch in range(args.epochs):
        print(f"\nRunning Epoch # {epoch}")

        loss, loss_pixel, loss_geometry = train_function(args, DEVICE, model, loss_fn_pixel, loss_fn_geometry, optimizer, scaler, train_loader)
        label_accuracy, segmentation_accuracy, predict_as_label = check_accuracy(val_loader, model, args, device=DEVICE)

        if args.wandb:
            # log_results(args, loss, label_accuracy, segmentation_accuracy, predict_as_label)
            log_results(args, loss, loss_pixel, loss_geometry, label_accuracy, segmentation_accuracy, predict_as_label)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":  optimizer.state_dict(),
        }

        if pth_save_point % 5 == 0: 
            torch.save(checkpoint, f"./results/UNet_Epoch_{epoch}.pth")
        pth_save_point += 1

        print("Current loss ", loss)
        if best_loss > loss:
            print("=====New best model=====")
            torch.save(checkpoint, f"./results/best.pth")
            save_predictions_as_images(args, val_loader, model, epoch, folder='./plot_results', device=DEVICE)
            best_loss, count = loss, 0
        else:
            count += 1

        # if count == args.patience:
        #     print("Early Stopping")
        #     break