import wandb

def initiate_wandb(args):
    if args.wandb:
        wandb.init(
            project=f"{args.wandb_project}", 
            entity=f"{args.wandb_entity}",
            name=f"{args.wandb_name}"
        )

# def log_results(
#     args, train_loss, label_accuracy, segmentation_accuracy, predict_as_label
#     ):
#     wandb.log({
#         'train loss': train_loss,
#         'whole image accuracy': segmentation_accuracy,
#         # 'label accuracy': label_accuracy,
#         # 'predicted as label': predict_as_label,
#     })

def log_results_no_label(args, train_loss, loss_pixel, loss_geometry, evaluation_list, highest_probability_mse):
    wandb.log({
        'train loss': train_loss,
        'pixel loss': loss_pixel,
        'geometry loss': loss_geometry,
        'whole image accuracy': evaluation_list[2],
        'dice score': evaluation_list[4],
        'pred to gt distance': highest_probability_mse,
    })

def log_results(args, train_loss, loss_pixel, loss_geometry, evaluation_list, highest_probability_mse):
    wandb.log({
        'train loss': train_loss,
        'pixel loss': loss_pixel,
        'geometry loss': loss_geometry,
        'whole image accuracy': evaluation_list[2],
        'label accuracy(1) - Preds/GT': evaluation_list[0],
        'label accuracy(2) - GT/Preds': evaluation_list[1],
        'predicted as label': evaluation_list[3],
        'dice score': evaluation_list[4],
        'pred to gt distance': highest_probability_mse,
    })