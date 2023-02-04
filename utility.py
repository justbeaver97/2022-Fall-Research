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
    TypeError: Cannot handle this data type: (1, 1, 512), <f4
        https://stackoverflow.com/questions/60138697/typeerror-cannot-handle-this-data-type-1-1-3-f4
    KeyError: ((1, 1, 512), '|u1')
        https://stackoverflow.com/questions/57621092/keyerror-1-1-1280-u1-while-using-pils-image-fromarray-pil
    permute tensor
        https://stackoverflow.com/questions/71880540/how-to-change-an-image-which-has-dimensions-512-512-3-to-a-tensor-of-size
"""

import os
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from tqdm import tqdm
from PIL import Image


def save_label_image(label_tensor, args):
    if args.delete_method == 'letter': num_channels = 7
    else:                              num_channels = 6

    for i in range(num_channels):
        plt.imshow(label_tensor[0][i].detach().cpu().numpy(), cmap='hot', interpolation='nearest')
        plt.savefig(f'./plot_results/{args.wandb_name}/annotation/label{i}.png')


def save_heatmap(preds, preds_binary, args, epoch):
    if args.only_pixel and epoch % 10 == 0:
        for i in range(len(preds[0])):
            plt.imshow(preds[0][i].detach().cpu().numpy(), cmap='hot', interpolation='nearest')
            plt.savefig(f'./plot_results/{args.wandb_name}/label{i}/epoch_{epoch}_heatmap.png')
            torchvision.utils.save_image(preds_binary[0][i], f'./plot_results/{args.wandb_name}/label{i}/epoch_{epoch}.png')
    elif not args.only_pixel:
        for i in range(len(preds[0])):
            plt.imshow(preds[0][i].detach().cpu().numpy(), cmap='hot', interpolation='nearest')
            plt.savefig(f'./plot_results/{args.wandb_name}/label{i}/epoch_{epoch}_heatmap.png')
            torchvision.utils.save_image(preds_binary[0][i], f'./plot_results/{args.wandb_name}/epoch_{epoch}.png')


def save_overlaid_image(args, idx, predicted_label, data_path):
    image_path = f'{args.padded_image}/{data_path}'
    if args.delete_method == 'letter': num_channels = 7
    else:                              num_channels = 6

    for i in range(num_channels):
        original = Image.open(image_path).resize((512,512)).convert("RGB")
        background = predicted_label[0][i].unsqueeze(0)
        background = TF.to_pil_image(torch.cat((background, background, background), dim=0))
        overlaid_image = Image.blend(original, background , 0.3)
        overlaid_image.save(f'./plot_results/{args.wandb_name}/overlaid/label{i}/val{idx}_overlaid.png')


def save_predictions_as_images(args, loader, model, epoch, device="cuda"):
    model.eval()

    for idx, (image, label, data_path) in enumerate(tqdm(loader)):
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
        if epoch % 10 == 0:
            save_overlaid_image(args, idx, preds_binary, data_path)

    model.train()


def check_accuracy(loader, model, args, epoch, device):
    print("=====Starting Validation=====")

    num_correct, num_pixels = 0, 0
    num_labels, num_labels_correct = 0, 0
    predict_as_label, prediction_correct  = 0, 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for image, label, _ in tqdm(loader):
            image = image.to(device)
            label = label.to(device)
            
            if args.pretrained: preds = model(image)
            else:               preds = torch.sigmoid(model(image))
            preds = (preds > 0.5).float()

            ## compare only labels
            if epoch % 10 == 0 and args.wandb:
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

    if epoch % 10 == 0:
        label_accuracy = (num_labels_correct/(num_labels+(1e-8))) * 100        ## from GT, how many of them were predicted
        label_accuracy2 = (prediction_correct/(predict_as_label+(1e-8))) * 100 ## from prediction, how many of them were GT
        print(f"Number of pixels predicted as label: {predict_as_label}")
        print(f"From Prediction: Got {prediction_correct}/{predict_as_label} with acc {label_accuracy2:.2f}")
        print(f"From Ground Truth: Got {num_labels_correct}/{num_labels} with acc {label_accuracy:.2f}")
        
    print(f"Got {num_correct}/{num_pixels} with acc {whole_image_accuracy:.2f}")
    print(f"Dice score: {dice}")

    model.train()

    return label_accuracy, label_accuracy2, whole_image_accuracy, predict_as_label, dice


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
