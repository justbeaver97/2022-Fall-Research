"""
reference:
    heatmap: https://stackoverflow.com/questions/53467215/convert-pytorch-cuda-tensor-to-numpy-array
             https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap 
    train process: https://github.com/aladdinpersson/Machine-Learning-Collection
"""

import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from spatial_mean import SpatialMean_CHAN
from log import log_results


def check_accuracy(loader, model, device):
    print("=====Starting Validation=====")

    num_correct, num_pixels = 0, 0
    num_labels, num_labels_correct = 0, 0
    predict_as_label = 0
    dice_score, tmp = 0, 0
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader):
            tmp += 1
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            # compare only labels
            # for i in range(len(preds[0][0])):
            #     for j in range(len(preds[0][0][i])):
            #         if int(y[0][0][i][j]) != 0:
            #             # print(int(y[0][0][i][j]), i, j, end=' ')
            #             num_labels += 1
            #             # print(float(preds[0][0][i][j]), i, j, end=' ')
            #             if int(preds[0][0][i][j]) == 1:
            #                 num_labels_correct += 1
            #                 # print(int(y[0][0][i][j]), i, j, end=' ')
            #             # print(preds[0][0][i][j], i, j, end=' ')
            #             # pass

            #         if int(preds[0][0][i][j]) != 0:
            #             predict_as_label += 1
            #             # print(float(preds[0][0][i][j]), i, j)

            # compare whole picture
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    label_accuracy = num_labels_correct/(num_labels+1e-8)
    whole_image_accuracy = num_correct/num_pixels*100

    # print(f"Number of pixels predicted as label: {predict_as_label}")
    # print(f"Got {num_labels_correct}/{num_labels} with acc {label_accuracy:.2f}")
    print(f"Got {num_correct}/{num_pixels} with acc {whole_image_accuracy:.2f}")

    # print(f"Dice score: {dice_score/len(loader)}")
    model.train()

    return label_accuracy, whole_image_accuracy, predict_as_label


def save_predictions_as_imgs(args, loader, model, epoch, folder="plot_results", device="cuda"):
    model.eval()
    print("save")

    # if not os.path.exists(f'{folder}'):
    #     os.mkdir(f'{folder}')
    if not os.path.exists(f'{folder}/{args.wandb_name}'):
        os.mkdir(f'{folder}/{args.wandb_name}')

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            # printsave(preds[0][0][0])
            # preds_binary = (preds > 0.5).float()

        channel_0 = preds[0][0].detach().cpu().numpy()
        channel_1 = preds[0][1].detach().cpu().numpy()
        channel_2 = preds[0][2].detach().cpu().numpy()
        channel_3 = preds[0][3].detach().cpu().numpy()
        channel_4 = preds[0][4].detach().cpu().numpy()
        channel_5 = preds[0][5].detach().cpu().numpy()

        if not os.path.exists(f'./plot_results/{args.wandb_name}/label0'):
            os.mkdir(f'./plot_results/{args.wandb_name}/label0')
        plt.imshow(channel_0, cmap='hot', interpolation='nearest')
        plt.savefig(f'./plot_results/{args.wandb_name}/label0/epoch_{epoch}.png')

        if not os.path.exists(f'./plot_results/{args.wandb_name}/label1'):
            os.mkdir(f'./plot_results/{args.wandb_name}/label1')
        plt.imshow(channel_1, cmap='hot', interpolation='nearest')
        plt.savefig(f'./plot_results/{args.wandb_name}/label1/epoch_{epoch}.png')

        if not os.path.exists(f'./plot_results/{args.wandb_name}/label2'):
            os.mkdir(f'./plot_results/{args.wandb_name}/label2')
        plt.imshow(channel_2, cmap='hot', interpolation='nearest')
        plt.savefig(f'./plot_results/{args.wandb_name}/label2/epoch_{epoch}.png')

        if not os.path.exists(f'./plot_results/{args.wandb_name}/label3'):
            os.mkdir(f'./plot_results/{args.wandb_name}/label3')
        plt.imshow(channel_3, cmap='hot', interpolation='nearest')
        plt.savefig(f'./plot_results/{args.wandb_name}/label3/epoch_{epoch}.png')

        if not os.path.exists(f'./plot_results/{args.wandb_name}/label4'):
            os.mkdir(f'./plot_results/{args.wandb_name}/label4')
        plt.imshow(channel_4, cmap='hot', interpolation='nearest')
        plt.savefig(f'./plot_results/{args.wandb_name}/label4/epoch_{epoch}.png')

        if not os.path.exists(f'./plot_results/{args.wandb_name}/label5'):
            os.mkdir(f'./plot_results/{args.wandb_name}/label5')
        plt.imshow(channel_5, cmap='hot', interpolation='nearest')
        plt.savefig(f'./plot_results/{args.wandb_name}/label5/epoch_{epoch}.png')

        # torchvision.utils.save_image(preds_binary, f"{folder}/pred_{epoch}_{idx}.png")
        # torchvision.utils.save_image(preds_binary[0][0], f"{folder}/{Loss}_tmp_epoch{epoch}_0.png")
        # torchvision.utils.save_image(preds_binary[0][1], f"{folder}/{Loss}_tmp_epoch{epoch}_1.png")
        # torchvision.utils.save_image(preds_binary[0][2], f"{folder}/{Loss}_tmp_epoch{epoch}_2.png")
        # torchvision.utils.save_image(preds_binary[0][3], f"{folder}/{Loss}_tmp_epoch{epoch}_3.png")
        # torchvision.utils.save_image(preds_binary[0][4], f"{folder}/{Loss}_tmp_epoch{epoch}_4.png")
        # torchvision.utils.save_image(preds_binary[0][5], f"{folder}/{Loss}_tmp_epoch{epoch}_5.png")

        # torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx}.png")
        break

    model.train()

def printsave(*a):
    file = open('error_log.txt','a')
    print(*a,file=file)


def train_function(args, DEVICE, model, loss_fn_pixel, loss_fn_geometry, optimizer, scaler, loader):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)
        # targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)

            # calculate log loss with pixel value
            loss_pixel = loss_fn_pixel(predictions, targets)

            # calculate mse loss with spatial mean value
            predict_spatial_mean_function = SpatialMean_CHAN(list(predictions.shape[1:]))
            predict_spatial_mean = predict_spatial_mean_function(predictions)
            targets_spatial_mean_function = SpatialMean_CHAN(list(targets.shape[1:]))
            targets_spatial_mean = targets_spatial_mean_function(targets)
            loss_geometry = loss_fn_geometry(predict_spatial_mean, targets_spatial_mean)

            # add two losses
            loss = loss_pixel + loss_geometry 
            # print(loss_pixel)
            # print(loss_geometry)
            # print(loss)
            # exit()

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
        label_accuracy, segmentation_accuracy, predict_as_label = check_accuracy(val_loader, model, device=DEVICE)

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
            save_predictions_as_imgs(args, val_loader, model, epoch, folder='./plot_results', device=DEVICE)
            best_loss, count = loss, 0
        else:
            count += 1

        # if count == args.patience:
        #     print("Early Stopping")
        #     break
