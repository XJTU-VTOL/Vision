import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_sched

def SGD_init(config, parameters):
    """
    configure the SGD optimizer for neural network
    please specify the dictionary as the `default` on below.
    API doc https://pytorch.org/docs/stable/optim.html?highlight=sgd#torch.optim.SGD
    """
    default = {"lr": 0.001, 
                "momentum": 0.0, 
                "weight_decay": 0.0, 
                "dampening": 0.0, 
                "nesterov": False}

    for key in default.keys():
        if key not in config.keys():
            print("WARN: {} not in input config file for SGD optimizer, setting to default {}".format(key, default[key]))
        else:
            default[key] = config[key]

    return torch.optim.SGD(
        parameters,
        lr = default["lr"],
        momentum = default["momentum"],
        dampening = default['dampening'],
        weight_decay = default["weight_decay"],
        nesterov = default["nesterov"]
    )

optimizers_all = {
    "SGD": SGD_init
}

def configure_optimizer(config, parameter):
    return optimizers_all[config["name"]](config, parameter)