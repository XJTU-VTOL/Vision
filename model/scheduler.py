import torch.optim.lr_scheduler as lr_sched
import math

def config_cosine_scheduler(config, optimizer):
    last_epoch = config["last_epoch"] if "last_epoch" in config.keys() else -1
    lf = lambda x: (((1 + math.cos(x * math.pi / config['total_epoch'])) / 2) ** 1.0) * 0.95 + 0.05  # cosine
    return lr_sched.LambdaLR(optimizer, lr_lambda= lf, last_epoch = last_epoch)

scheduler_all = {
    "cosine": config_cosine_scheduler
}

def config_scheduler(config, optimizer):
    return scheduler_all[config["name"]](config, optimizer)

