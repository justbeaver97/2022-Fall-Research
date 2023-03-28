import wandb

def initiate_wandb(args):
    if args.wandb:
        wandb.init(
            project=f"{args.wandb_project}", 
            entity=f"{args.wandb_entity}",
            name=f"{args.wandb_name}"
        )


def log_results_no_label(
        args, train_loss, loss_pixel, loss_geometry, evaluation_list, highest_probability_mse, mse_list, val_loader_len
    ):
    wandb.log({
        'Train Loss': train_loss,
        'Pixel Loss': loss_pixel,
        'Geometry Loss': loss_geometry,
        'Whole Image Accuracy': evaluation_list[2],
        'DICE Score': evaluation_list[4],
        'Average RMSE': highest_probability_mse/val_loader_len,
        'Label0 RMSE': sum(mse_list[0])/val_loader_len,
        'Label1 RMSE': sum(mse_list[1])/val_loader_len,
        'Label2 RMSE': sum(mse_list[2])/val_loader_len,
        'Label3 RMSE': sum(mse_list[3])/val_loader_len,
        'Label4 RMSE': sum(mse_list[4])/val_loader_len,
        'Label5 RMSE': sum(mse_list[5])/val_loader_len,
    })
    

def log_results(
        args, train_loss, loss_pixel, loss_geometry, evaluation_list, highest_probability_mse, mse_list, val_loader_len
    ):
    wandb.log({
        'Train Loss': train_loss,
        'Pixel Loss': loss_pixel,
        'Geometry Loss': loss_geometry,
        'Whole Image Accuracy': evaluation_list[2],
        'Label Accuracy(1) - Preds/GT': evaluation_list[0],
        'Label Accuracy(2) - GT/Preds': evaluation_list[1],
        'Predicted as Label': evaluation_list[3],
        'DICE Score': evaluation_list[4],
        'Average RMSE': highest_probability_mse/val_loader_len,
        'Label0 RMSE': sum(mse_list[0])/val_loader_len,
        'Label1 RMSE': sum(mse_list[1])/val_loader_len,
        'Label2 RMSE': sum(mse_list[2])/val_loader_len,
        'Label3 RMSE': sum(mse_list[3])/val_loader_len,
        'Label4 RMSE': sum(mse_list[4])/val_loader_len,
        'Label5 RMSE': sum(mse_list[5])/val_loader_len,
    })