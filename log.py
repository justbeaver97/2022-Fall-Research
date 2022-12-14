import wandb

def initiate_wandb(args):
    wandb.init(
        project=f"{args.wandb_project}", 
        entity=f"{args.wandb_entity}",
        name=f"{args.wandb_name}"
    )

def log_results(
    args, train_loss, label_accuracy, segmentation_accuracy, predict_as_label
    ):
    wandb.log({
        'train loss': train_loss,
        'label accuracy': label_accuracy,
        'whole image accuracy': segmentation_accuracy,
        'predicted as label': predict_as_label,
    })