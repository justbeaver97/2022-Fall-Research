
import torch
import torchvision
import numpy as np

from tqdm import tqdm

def check_accuracy(loader, model, device):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    # print(f"Dice score: {dice_score/len(loader)}")
    model.train()

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
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

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
    count, best_loss = 0, np.inf

    for epoch in range(args.epochs):
        loss = train_function(train_loader, model, optimizer, loss_fn, scaler)
        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        if best_loss < loss:
            best_loss = loss

        if best_loss > loss:
            # save model
            print("New best model with loss ", loss)
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            torch.save(checkpoint, f"{args.pth_path}/UNet_epoch_{epoch}.pth")

            # print some examples to a folder
            # save_predictions_as_imgs(
            #     val_loader, model, folder=f"{args.pth_path}", device=DEVICE
            # )

            best_loss = loss
        else:
            count += 1

        if count == 3:
            print("Early Stopping")
            break