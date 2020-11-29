import pytorch_lightning as pl

from .optimizer import configure_optimizer
from .scheduler import config_scheduler

class ModelBase(pl.LightningModule):
    """
    Base Class for lightning model
    Configure Training Optimizer and other traing process.
    Input:
        Configuration Dictionary for Optimizer and Scheduler Setting.
    
    """
    def __init__(self, hyper):
        super(ModelBase, self).__init__()
        self.hparams = hyper
        # set the leraning rate attribute to find auto learning rate
        self.lr = self.hparams["lr"] if "lr" in self.hparams.keys() else 0.0001
        # get the configuration for optmizer
        self.optimizer_config = self.hparams['training']['optimizer']
        self.scheduler_config = self.hparams['training']['scheduler']

        # updated layer index for training
        self.updated_layer_index = None

    def configure_optimizers(self):
        updated_params = []
        if self.updated_layer_index:
            for i in self.updated_layer_index:
                updated_params += list(self.model.module_list[i].parameters())
        else:
            updated_params = self.parameters()

        self.optimizer = configure_optimizer(self.optimizer_config, updated_params)
        self.lr_scheduler = config_scheduler(self.scheduler_config, self.optimizer)

        return  [self.optimizer], [
                {
                 'scheduler': self.lr_scheduler,
                 'interval': self.scheduler_config['interval'],
                 'frequency': self.scheduler_config['frequency'],
                }]