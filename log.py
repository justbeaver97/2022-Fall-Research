import wandb

def initiate_wandb(args):
    if args.wandb:
        wandb.init(
            project=f"{args.wandb_project}", 
            entity=f"{args.wandb_entity}",
            name=f"{args.wandb_name}"
        )


def log_results_no_label(
        args, loss, loss_pixel, loss_geometry, loss_angle, 
        evaluation_list, angle_list, highest_probability_mse, mse_list, val_loader_len
    ):
    if args.output_channel == 6:
        wandb.log({
            'Train Loss': loss,
            'Pixel Loss': loss_pixel,
            'Geometry Loss': loss_geometry,
            'Angular Loss': loss_angle,
            'Whole Image Accuracy': evaluation_list[2],
            'DICE Score': evaluation_list[4],
            'Average RMSE': highest_probability_mse/val_loader_len,
            'Medial Femur': sum(mse_list[0])/val_loader_len,
            'Implant Upper Left': sum(mse_list[1])/val_loader_len,
            'Implant Upper Center': sum(mse_list[2])/val_loader_len,
            'Implant Lower Left': sum(mse_list[3])/val_loader_len,
            'Implant Lower Center': sum(mse_list[4])/val_loader_len,
            'Medial Tibia': sum(mse_list[5])/val_loader_len,
            'LDFA Difference': angle_list[0]/val_loader_len,
            'MPTA Difference': angle_list[1]/val_loader_len,
            'mHKA Difference': angle_list[2]/val_loader_len,
        })
    elif args.output_channel == 8:
        wandb.log({
            'Train Loss': loss,
            'Pixel Loss': loss_pixel,
            'Geometry Loss': loss_geometry,
            'Angular Loss': loss_angle,
            'Whole Image Accuracy': evaluation_list[2],
            'DICE Score': evaluation_list[4],
            'Average RMSE': highest_probability_mse/val_loader_len,
            'Medial Femur': sum(mse_list[0])/val_loader_len,
            'Implant Upper Left': sum(mse_list[1])/val_loader_len,
            'Implant Upper Right': sum(mse_list[2])/val_loader_len,
            'Implant Upper Center': sum(mse_list[3])/val_loader_len,
            'Implant Lower Left': sum(mse_list[4])/val_loader_len,
            'Implant Lower Right': sum(mse_list[5])/val_loader_len,
            'Implant Lower Center': sum(mse_list[6])/val_loader_len,
            'Medial Tibia': sum(mse_list[7])/val_loader_len,
            'LDFA Difference': angle_list[0]/val_loader_len,
            'MPTA Difference': angle_list[1]/val_loader_len,
            'mHKA Difference': angle_list[2]/val_loader_len,
        })
    

def log_results(
        args, loss, loss_pixel, loss_geometry, loss_angle, 
        evaluation_list, angle_list, highest_probability_mse, mse_list, val_loader_len
    ):
    if args.output_channel == 6:
        wandb.log({
            'Train Loss': loss,
            'Pixel Loss': loss_pixel,
            'Geometry Loss': loss_geometry,
            'Angular Loss': loss_angle,
            'Whole Image Accuracy': evaluation_list[2],
            'Label Accuracy(1) - Preds/GT': evaluation_list[0],
            'Label Accuracy(2) - GT/Preds': evaluation_list[1],
            'Predicted as Label': evaluation_list[3],
            'DICE Score': evaluation_list[4],
            'Average RMSE': highest_probability_mse/val_loader_len,
            'Medial Femur': sum(mse_list[0])/val_loader_len,
            'Implant Upper Left': sum(mse_list[1])/val_loader_len,
            'Implant Upper Center': sum(mse_list[2])/val_loader_len,
            'Implant Lower Left': sum(mse_list[3])/val_loader_len,
            'Implant Lower Center': sum(mse_list[4])/val_loader_len,
            'Medial Tibia': sum(mse_list[5])/val_loader_len,
            'LDFA Difference': angle_list[0]/val_loader_len,
            'MPTA Difference': angle_list[1]/val_loader_len,
            'mHKA Difference': angle_list[2]/val_loader_len,
        })
    elif args.output_channel == 8:
        wandb.log({
            'Train Loss': loss,
            'Pixel Loss': loss_pixel,
            'Geometry Loss': loss_geometry,
            'Angular Loss': loss_angle,
            'Whole Image Accuracy': evaluation_list[2],
            'Label Accuracy(1) - Preds/GT': evaluation_list[0],
            'Label Accuracy(2) - GT/Preds': evaluation_list[1],
            'Predicted as Label': evaluation_list[3],
            'DICE Score': evaluation_list[4],
            'Average RMSE': highest_probability_mse/val_loader_len,
            'Medial Femur': sum(mse_list[0])/val_loader_len,
            'Implant Upper Left': sum(mse_list[1])/val_loader_len,
            'Implant Upper Right': sum(mse_list[2])/val_loader_len,
            'Implant Upper Center': sum(mse_list[3])/val_loader_len,
            'Implant Lower Left': sum(mse_list[4])/val_loader_len,
            'Implant Lower Right': sum(mse_list[5])/val_loader_len,
            'Implant Lower Center': sum(mse_list[6])/val_loader_len,
            'Medial Tibia': sum(mse_list[7])/val_loader_len,
            'LDFA Difference': angle_list[0]/val_loader_len,
            'MPTA Difference': angle_list[1]/val_loader_len,
            'mHKA Difference': angle_list[2]/val_loader_len,
        })