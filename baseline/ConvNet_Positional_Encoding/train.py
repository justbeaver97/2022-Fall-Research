import torch
import numpy as np
import wandb

from tqdm import tqdm
from sklearn.metrics import mean_squared_error as mse
from positional_encoding import positional_encoding


def printsave(name, *a):
    file = open(f'../../plot_data/box_plot/txt_files/{name}','a')
    print(*a,file=file)


def box_plot(args, mse_list):
    ## I can't make box plot of 3 different methods 
    ## I have to just save it as a file, and then create it from saved text files
    printsave(f'{args.wandb_name}_MSE_LIST', mse_list)
    

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
    mse_value = mse(highest_probability_pixels, ordered_label_list, squared=False) 
    for i in range(6):
        mse_list[i][idx] = mse(highest_probability_pixels[2*i:2*(i+1)] , ordered_label_list[2*i:2*(i+1)], squared=False) 

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

        print("Current loss ", loss)
        print("Current MSE ", highest_probability_mse_total/len(val_loader))
        wandb.log({
            'Train Loss': loss,
            'Average RMSE': highest_probability_mse_total/len(val_loader),
            'Label0 RMSE': sum(mse_list[0])/len(val_loader),
            'Label1 RMSE': sum(mse_list[1])/len(val_loader),
            'Label2 RMSE': sum(mse_list[2])/len(val_loader),
            'Label3 RMSE': sum(mse_list[3])/len(val_loader),
            'Label4 RMSE': sum(mse_list[4])/len(val_loader),
            'Label5 RMSE': sum(mse_list[5])/len(val_loader),
        })

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":  optimizer.state_dict(),
        }

        if epoch == args.epochs - 1:
            box_plot(args, mse_list)
            torch.save(checkpoint, f"../../results/{args.wandb_name}.pth")