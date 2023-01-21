"""
Reference:
    heatmap: https://stackoverflow.com/questions/53467215/convert-pytorch-cuda-tensor-to-numpy-array
             https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap 
    tensor to image: https://hello-bryan.tistory.com/429
"""

import os
import torch
import torchvision
import matplotlib.pyplot as plt

from tqdm import tqdm


def save_heatmap(preds, preds_binary, args, epoch):
    channel_0, channel_1 = preds[0][0].detach().cpu().numpy(), preds[0][1].detach().cpu().numpy()
    channel_2, channel_3 = preds[0][2].detach().cpu().numpy(), preds[0][3].detach().cpu().numpy()
    channel_4, channel_5 = preds[0][4].detach().cpu().numpy(), preds[0][5].detach().cpu().numpy()

    if args.only_pixel and epoch % 50 == 0:
        if not os.path.exists(f'./plot_results/{args.wandb_name}/label0'):
            os.mkdir(f'./plot_results/{args.wandb_name}/label0')
        plt.imshow(channel_0, cmap='hot', interpolation='nearest')
        plt.savefig(f'./plot_results/{args.wandb_name}/label0/epoch_{epoch}_heatmap.png')
        torchvision.utils.save_image(preds_binary[0][0], f'./plot_results/{args.wandb_name}/label0/epoch_{epoch}.png')

        if not os.path.exists(f'./plot_results/{args.wandb_name}/label1'):
            os.mkdir(f'./plot_results/{args.wandb_name}/label1')
        plt.imshow(channel_1, cmap='hot', interpolation='nearest')
        plt.savefig(f'./plot_results/{args.wandb_name}/label1/epoch_{epoch}_heatmap.png')
        torchvision.utils.save_image(preds_binary[0][1], f'./plot_results/{args.wandb_name}/label1/epoch_{epoch}.png')

        if not os.path.exists(f'./plot_results/{args.wandb_name}/label2'):
            os.mkdir(f'./plot_results/{args.wandb_name}/label2')
        plt.imshow(channel_2, cmap='hot', interpolation='nearest')
        plt.savefig(f'./plot_results/{args.wandb_name}/label2/epoch_{epoch}_heatmap.png')
        torchvision.utils.save_image(preds_binary[0][2], f'./plot_results/{args.wandb_name}/label2/epoch_{epoch}.png')

        if not os.path.exists(f'./plot_results/{args.wandb_name}/label3'):
            os.mkdir(f'./plot_results/{args.wandb_name}/label3')
        plt.imshow(channel_3, cmap='hot', interpolation='nearest')
        plt.savefig(f'./plot_results/{args.wandb_name}/label3/epoch_{epoch}_heatmap.png')
        torchvision.utils.save_image(preds_binary[0][3], f'./plot_results/{args.wandb_name}/label3/epoch_{epoch}.png')

        if not os.path.exists(f'./plot_results/{args.wandb_name}/label4'):
            os.mkdir(f'./plot_results/{args.wandb_name}/label4')
        plt.imshow(channel_4, cmap='hot', interpolation='nearest')
        plt.savefig(f'./plot_results/{args.wandb_name}/label4/epoch_{epoch}_heatmap.png')
        torchvision.utils.save_image(preds_binary[0][4], f'./plot_results/{args.wandb_name}/label4/epoch_{epoch}.png')

        if not os.path.exists(f'./plot_results/{args.wandb_name}/label5'):
            os.mkdir(f'./plot_results/{args.wandb_name}/label5')
        plt.imshow(channel_5, cmap='hot', interpolation='nearest')
        plt.savefig(f'./plot_results/{args.wandb_name}/label5/epoch_{epoch}_heatmap.png')
        torchvision.utils.save_image(preds_binary[0][5], f'./plot_results/{args.wandb_name}/label5/epoch_{epoch}.png')

    elif not args.only_pixel:
        if not os.path.exists(f'./plot_results/{args.wandb_name}/label0'):
            os.mkdir(f'./plot_results/{args.wandb_name}/label0')
        plt.imshow(channel_0, cmap='hot', interpolation='nearest')
        plt.savefig(f'./plot_results/{args.wandb_name}/label0/epoch_{epoch}_heatmap.png')
        torchvision.utils.save_image(preds_binary[0][0], f'./plot_results/{args.wandb_name}/epoch_{epoch}.png')

        if not os.path.exists(f'./plot_results/{args.wandb_name}/label1'):
            os.mkdir(f'./plot_results/{args.wandb_name}/label1')
        plt.imshow(channel_1, cmap='hot', interpolation='nearest')
        plt.savefig(f'./plot_results/{args.wandb_name}/label1/epoch_{epoch}_heatmap.png')
        torchvision.utils.save_image(preds_binary[0][1], f'./plot_results/{args.wandb_name}/epoch_{epoch}.png')

        if not os.path.exists(f'./plot_results/{args.wandb_name}/label2'):
            os.mkdir(f'./plot_results/{args.wandb_name}/label2')
        plt.imshow(channel_2, cmap='hot', interpolation='nearest')
        plt.savefig(f'./plot_results/{args.wandb_name}/label2/epoch_{epoch}_heatmap.png')
        torchvision.utils.save_image(preds_binary[0][2], f'./plot_results/{args.wandb_name}/epoch_{epoch}.png')

        if not os.path.exists(f'./plot_results/{args.wandb_name}/label3'):
            os.mkdir(f'./plot_results/{args.wandb_name}/label3')
        plt.imshow(channel_3, cmap='hot', interpolation='nearest')
        plt.savefig(f'./plot_results/{args.wandb_name}/label3/epoch_{epoch}_heatmap.png')
        torchvision.utils.save_image(preds_binary[0][3], f'./plot_results/{args.wandb_name}/epoch_{epoch}.png')

        if not os.path.exists(f'./plot_results/{args.wandb_name}/label4'):
            os.mkdir(f'./plot_results/{args.wandb_name}/label4')
        plt.imshow(channel_4, cmap='hot', interpolation='nearest')
        plt.savefig(f'./plot_results/{args.wandb_name}/label4/epoch_{epoch}_heatmap.png')
        torchvision.utils.save_image(preds_binary[0][4], f'./plot_results/{args.wandb_name}/epoch_{epoch}.png')

        if not os.path.exists(f'./plot_results/{args.wandb_name}/label5'):
            os.mkdir(f'./plot_results/{args.wandb_name}/label5')
        plt.imshow(channel_5, cmap='hot', interpolation='nearest')
        plt.savefig(f'./plot_results/{args.wandb_name}/label5/epoch_{epoch}_heatmap.png')
        torchvision.utils.save_image(preds_binary[0][5], f'./plot_results/{args.wandb_name}/epoch_{epoch}.png')


def save_label_image(label_tensor, args):
    if not os.path.exists(f'./plot_results/{args.wandb_name}/annotation'):
        os.mkdir(f'./plot_results/{args.wandb_name}/annotation')
    
    for i in range(6):
        plt.imshow(label_tensor[0][i].detach().cpu().numpy(), cmap='hot', interpolation='nearest')
        plt.savefig(f'./plot_results/{args.wandb_name}/annotation/label{i}.png')


def save_overlaid_image(args, epoch, original_data, predicted_label):
    if not os.path.exists(f'./plot_results/{args.wandb_name}/overlaid'):
        os.mkdir(f'./plot_results/{args.wandb_name}/overlaid')

    for i in range(6):
        if not os.path.exists(f'./plot_results/{args.wandb_name}/overlaid/label{i}'):
            os.mkdir(f'./plot_results/{args.wandb_name}/overlaid/label{i}')
        torchvision.utils.save_image(
            predicted_label[0][i], f'./plot_results/{args.wandb_name}/overlaid/label{i}/epoch_{epoch}.png')
    
    torchvision.utils.save_image(original_data, f'./plot_results/{args.wandb_name}/overlaid/original.png')

    ## todo - overlay image
    
    exit()


def save_predictions_as_images(args, loader, model, epoch, folder="plot_results", device="cuda"):
    model.eval()

    if not os.path.exists(f'{folder}/{args.wandb_name}'):
        os.mkdir(f'{folder}/{args.wandb_name}')

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device)

        with torch.no_grad():
            if args.pretrained:
                preds = model(x)
            else:
                preds = torch.sigmoid(model(x))
            # printsave(preds[0][0][0])
            preds_binary = (preds > args.threshold).float()
            # printsave(preds_binary[0][0][0])

        if epoch == 0: 
            save_label_image(y, args)
        save_heatmap(preds, preds_binary, args, epoch)
        save_overlaid_image(args, epoch, x, preds_binary)
        break

    model.train()


def check_accuracy(loader, model, args, epoch, device):
    print("=====Starting Validation=====")

    num_correct, num_pixels = 0, 0
    num_labels, num_labels_correct = 0, 0
    predict_as_label, prediction_correct  = 0, 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device)
            
            if args.pretrained: preds = model(x)
            else:               preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            # ## compare only labels
            # if epoch % 25 == 0:
            #     for i in range(len(preds[0][0])):
            #         for j in range(len(preds[0][0][i])):
            #             if float(y[0][0][i][j]) == 1.0:
            #                 num_labels += 1
            #                 if float(preds[0][0][i][j]) == 1.0:
            #                     num_labels_correct += 1

            #             if float(preds[0][0][i][j]) == 1.0:
            #                 predict_as_label += 1
            #                 if float(y[0][0][i][j]) == 1.0:
            #                     prediction_correct += 1

            # compare whole picture
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    label_accuracy, label_accuracy2 = 0, 0
    whole_image_accuracy = num_correct/num_pixels*100
    dice = dice_score/len(loader)

    if epoch % 25 == 0:
        label_accuracy = (num_labels_correct/(num_labels+(1e-8))) * 100        ## from GT, how many of them were predicted
        label_accuracy2 = (prediction_correct/(predict_as_label+(1e-8))) * 100 ## from prediction, how many of them were GT
        print(f"Number of pixels predicted as label: {predict_as_label}")
        print(f"From Prediction: Got {prediction_correct}/{predict_as_label} with acc {label_accuracy2:.2f}")
        print(f"From Ground Truth: Got {num_labels_correct}/{num_labels} with acc {label_accuracy:.2f}")
        
    print(f"Got {num_correct}/{num_pixels} with acc {whole_image_accuracy:.2f}")
    print(f"Dice score: {dice}")

    model.train()

    return label_accuracy, label_accuracy2, whole_image_accuracy, predict_as_label, dice


def printsave(*a):
    file = open('tmp/error_log.txt','a')
    print(*a,file=file)
