"""
reference:
    https://github.com/aladdinpersson/Machine-Learning-Collection
"""

import os
import torch
import torchvision
import numpy as np

from tqdm import tqdm
from log import log_results


def check_accuracy(loader, model, device):
    print("Starting Validation")

    num_correct, num_pixels = 0, 0
    num_labels, num_labels_correct = 0, 0
    predict_as_label = 0
    dice_score, tmp = 0, 0
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader):
            # print("\n",tmp)
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

    print(f"Number of pixels predicted as label: {predict_as_label}")
    print(f"Got {num_labels_correct}/{num_labels} with acc {label_accuracy:.2f}")
    print(f"Got {num_correct}/{num_pixels} with acc {whole_image_accuracy:.2f}")

    # print(f"Dice score: {dice_score/len(loader)}")
    model.train()

    return label_accuracy, whole_image_accuracy, predict_as_label


def save_predictions_as_imgs(loader, model, folder="saved_images", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()


def train_function(args, DEVICE, model, loss_fn, optimizer, scaler, loader):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)
        # targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    return loss.item()


def train(args, DEVICE, model, loss_fn, optimizer, scaler, train_loader, val_loader):
    count, pth_save_point, best_loss = 0, 0, np.inf

    if not os.path.exists('./results'):
        os.mkdir(f'./results')

    for epoch in range(args.epochs):
        print(f"\nRunning Epoch # {epoch}")
        loss = train_function(args, DEVICE, model, loss_fn,
                              optimizer, scaler, train_loader)
        label_accuracy, segmentation_accuracy, predict_as_label = check_accuracy(
            val_loader, model, device=DEVICE)

        if args.wandb:
            log_results(args, loss, label_accuracy,
                        segmentation_accuracy, predict_as_label)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":  optimizer.state_dict(),
        }
        if pth_save_point % 5 == 0:
            torch.save(checkpoint, f"./results/UNet_Epoch_{epoch}.pth")
        pth_save_point += 1

        if best_loss > loss:
            print("New best model with loss ", loss)
            torch.save(checkpoint, f"./results/best.pth")

            # print some examples to a folder
            # save_predictions_as_imgs(
            #     val_loader, model, folder='./results', device=DEVICE
            # )

            best_loss = loss
            count = 0
        else:
            count += 1

        if count == args.patience:
            print("Early Stopping")
            break
