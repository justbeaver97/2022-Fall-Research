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
    TypeError: Cannot handle this data type: (1, 1, 512), <f4:
        https://stackoverflow.com/questions/60138697/typeerror-cannot-handle-this-data-type-1-1-3-f4
    KeyError: ((1, 1, 512), '|u1'):
        https://stackoverflow.com/questions/57621092/keyerror-1-1-1280-u1-while-using-pils-image-fromarray-pil
    permute tensor:
        https://stackoverflow.com/questions/71880540/how-to-change-an-image-which-has-dimensions-512-512-3-to-a-tensor-of-size
    draw circle in image:
        https://dsbook.tistory.com/102
    >  - Expected Ptr<cv::UMat> for argument 'img':
        https://github.com/opencv/opencv/issues/18120
        cv2 functions expects numpy array
"""

import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import seaborn as sns

from tqdm import tqdm
from PIL import Image


def save_label_image(label_tensor, args):
    if args.delete_method == 'letter': num_channels = 7
    else:                              num_channels = 6

    for i in range(num_channels):
        plt.imshow(label_tensor[0][i].detach().cpu().numpy(), cmap='hot', interpolation='nearest')
        plt.savefig(f'./plot_results/{args.wandb_name}/annotation/label{i}.png')


def save_heatmap(preds, preds_binary, args, epoch):
    if args.only_pixel and (epoch % 10 == 0 or epoch % 50 == 49):
        for i in range(len(preds[0])):
            plt.imshow(preds[0][i].detach().cpu().numpy(), cmap='hot', interpolation='nearest')
            plt.savefig(f'./plot_results/{args.wandb_name}/label{i}/epoch_{epoch}_heatmap.png')
            torchvision.utils.save_image(preds_binary[0][i], f'./plot_results/{args.wandb_name}/label{i}/epoch_{epoch}.png')
    elif not args.only_pixel:
        for i in range(len(preds[0])):
            plt.imshow(preds[0][i].detach().cpu().numpy(), cmap='hot', interpolation='nearest')
            plt.savefig(f'./plot_results/{args.wandb_name}/label{i}/epoch_{epoch}_heatmap.png')
            torchvision.utils.save_image(preds_binary[0][i], f'./plot_results/{args.wandb_name}/epoch_{epoch}.png')


def prediction_plot(args, idx, highest_probability_pixels_list, i, original):
    # print(highest_probability_pixels_list)
    # print(np.array(highest_probability_pixels_list).shape)
    # x, y = int(highest_probability_pixels_list[idx][0][i][0]), int(highest_probability_pixels_list[idx][0][i][1])
    x, y = int(highest_probability_pixels_list[idx][i][0][1]), int(highest_probability_pixels_list[idx][i][0][0])
    pixel_overlaid_image = Image.fromarray(cv2.circle(np.array(original), (x,y), 15, (255, 0, 0),-1))
    pixel_overlaid_image.save(f'./plot_results/{args.wandb_name}/overlaid/label{i}/val{idx}_pixel_overlaid.png')


def ground_truth_prediction_plot(highest_probability_pixels_list, label_list_total):
    printsave(highest_probability_pixels_list, label_list_total)
    # x, y = int(highest_probability_pixels_list[idx][0][i][0]), int(highest_probability_pixels_list[idx][0][i][1])
    # pixel_overlaid_image = Image.fromarray(cv2.circle(np.array(original), (x,y), 15, (255, 0, 0),-1))
    # pixel_overlaid_image.save(f'./plot_results/{args.wandb_name}/overlaid/label{i}/val{idx}_pixel_overlaid.png')
    

def save_overlaid_image(args, idx, predicted_label, data_path, highest_probability_pixels_list, label_list_total, epoch):
    image_path = f'{args.padded_image}/{data_path}'
    if args.delete_method == 'letter': num_channels = 7
    else:                              num_channels = 6

    for i in range(num_channels):
        original = Image.open(image_path).resize((512,512)).convert("RGB")
        background = predicted_label[0][i].unsqueeze(0)
        background = TF.to_pil_image(torch.cat((background, background, background), dim=0))
        overlaid_image = Image.blend(original, background , 0.3)
        overlaid_image.save(f'./plot_results/{args.wandb_name}/overlaid/label{i}/val{idx}_overlaid.png')
        overlaid_image.save(f'./plot_results/{args.wandb_name}/label{i}/epoch_{epoch}_overlaid.png')

        if i != 6:
            prediction_plot(args, idx, highest_probability_pixels_list, i, original)
            ground_truth_prediction_plot(highest_probability_pixels_list, label_list_total)


def save_predictions_as_images(args, loader, model, epoch, highest_probability_pixels_list, label_list_total, device="cuda"):
    model.eval()

    for idx, (image, label, data_path, _) in enumerate(tqdm(loader)):
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
        if epoch % 10 == 0 or epoch % args.dilation_epoch == (args.dilation_epoch-1):
            save_overlaid_image(args, idx, preds_binary, data_path, highest_probability_pixels_list, label_list_total, epoch)
            
    model.train()


def printsave(*a):
    file = open('tmp/error_log.txt','a')
    print(*a,file=file)


def box_plot(mse_list):
    print("box plot")