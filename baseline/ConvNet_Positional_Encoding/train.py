"""
reference:
    train process: 
        https://github.com/aladdinpersson/Machine-Learning-Collection
"""

import torch
import numpy as np
import wandb

from tqdm import tqdm
from sklearn.metrics import mean_squared_error as mse
from positional_encoding import positional_encoding


def box_plot():
    pass


def calculate_mse_predicted_to_annotation(highest_probability_pixels, label_list, idx, mse_list):
    highest_probability_pixels = highest_probability_pixels.squeeze(0).reshape(12,1).detach().cpu()
    label_list = np.array(torch.Tensor(label_list).detach().cpu(), dtype=object).reshape(12,1)
    label_list = np.ndarray.tolist(label_list)
    ordered_label_list = [
        label_list[1], label_list[0],
        label_list[3], label_list[2],
        label_list[5], label_list[4],
        label_list[7], label_list[6],
        label_list[9], label_list[8],
        label_list[11], label_list[10],
    ]
    mse_value = mse(highest_probability_pixels, ordered_label_list) 
    for i in range(6):
        mse_list[i][idx] = mse(highest_probability_pixels[2*i:2*(i+1)] , ordered_label_list[2*i:2*(i+1)])

    return mse_value, mse_list


def train(args, DEVICE, model, loss_fn, optimizer, train_loader, val_loader):    
    for epoch in range(args.epochs):
        model.train()
        print(f"\nRunning Epoch # {epoch}")

        for idx, (data, targets) in enumerate(tqdm(train_loader)):
            data    = positional_encoding(args, data).to(DEVICE)
            targets = targets.to(DEVICE)

            predictions = model(data)
            loss = loss_fn(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            highest_probability_mse_total = 0
            mse_list = [[0]*len(val_loader) for _ in range(6)]
            for idx, (data, targets) in enumerate(tqdm(val_loader)):
                data    = positional_encoding(args, data).to(DEVICE)
                targets = targets.to(DEVICE)
                preds = model(data)

                highest_probability_mse, mse_list = calculate_mse_predicted_to_annotation(
                    preds, targets, idx, mse_list
                )
                highest_probability_mse_total += highest_probability_mse

        if epoch == args.epochs - 1:
            box_plot(mse_list)

        print("Current loss ", loss)
        print("Current MSE ", highest_probability_mse_total/len(val_loader))
        wandb.log({
            'train loss': loss,
            'pred to gt distance': highest_probability_mse_total/len(val_loader),
        })
        