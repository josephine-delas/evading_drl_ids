import wandb

wandb.init(project="wandb-pytorch-test", settings=wandb.Settings(start_method="fork"))

for my_metric in range(10):
    wandb.log({'my_metric': my_metric})