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

    if args.only_pixel and epoch % 10 == 0:
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
    channel_0 = label_tensor[0][0].detach().cpu().numpy()
    channel_1 = label_tensor[0][1].detach().cpu().numpy()
    channel_2 = label_tensor[0][2].detach().cpu().numpy()
    channel_3 = label_tensor[0][3].detach().cpu().numpy()
    channel_4 = label_tensor[0][4].detach().cpu().numpy()
    channel_5 = label_tensor[0][5].detach().cpu().numpy()

    if not os.path.exists(f'./plot_results/{args.wandb_name}/annotation'):
        os.mkdir(f'./plot_results/{args.wandb_name}/annotation')

    plt.imshow(channel_0, cmap='hot', interpolation='nearest')
    plt.savefig(f'./plot_results/{args.wandb_name}/annotation/label0.png')
    plt.imshow(channel_1, cmap='hot', interpolation='nearest')
    plt.savefig(f'./plot_results/{args.wandb_name}/annotation/label1.png')
    plt.imshow(channel_2, cmap='hot', interpolation='nearest')
    plt.savefig(f'./plot_results/{args.wandb_name}/annotation/label2.png')
    plt.imshow(channel_3, cmap='hot', interpolation='nearest')
    plt.savefig(f'./plot_results/{args.wandb_name}/annotation/label3.png')
    plt.imshow(channel_4, cmap='hot', interpolation='nearest')
    plt.savefig(f'./plot_results/{args.wandb_name}/annotation/label4.png')
    plt.imshow(channel_5, cmap='hot', interpolation='nearest')
    plt.savefig(f'./plot_results/{args.wandb_name}/annotation/label5.png')


def printsave(*a):
    file = open('tmp/error_log.txt','a')
    print(*a,file=file)


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
            printsave(preds[0][0][0])
            preds_binary = (preds > 0.70).float()
            printsave(preds_binary[0][0][0])

        if epoch == 0: 
            save_label_image(y, args)
        save_heatmap(preds, preds_binary, args, epoch)
        break

    model.train()


def check_accuracy(loader, model, args, device):
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
            if args.pretrained:
                preds = model(x)
            else:
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
