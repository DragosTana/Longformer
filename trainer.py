import torch
import torch.nn as nn
import wandb
import time
from tqdm import tqdm

class Trainer():
    """
    A class to handle the training of PyTorch models.
    
    ### Arguments:
    - device: A string. The device to use for training. Choose between 'cpu' and 'cuda'.
    - default_root_dir: A string. The default root directory to save the model weights.
    - optimizer: A PyTorch optimizer.
    - scheduler: A PyTorch scheduler.
    - compute_metrics: A function to compute the metrics.
    - continue_training: A boolean. Whether to continue training from the last checkpoint.
    - logger: A string. The logger to use for logging the training. Choose between 'wandb' and 'tensorboard'.
    - log: A boolean. Whether to log the training.
    - max_epochs: An integer. The maximum number of epochs to train the model.
    - gradient_accumulation_steps: An integer. The number of gradient accumulation steps.
    - project_name: A string. The name of the project.
    """
    def __init__(self, 
                 device,
                 default_root_dir,
                 optimizer = None,
                 scheduler = None,
                 compute_metrics = None,
                 continue_training=False,
                 logger = None,
                 log = False,
                 max_epochs=500,
                 gradient_accumulation_steps=1,
                 project_name="my-awesome-project",
                ):
        self.device = device
        self.logger = logger
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.compute_metrics = compute_metrics
        self.default_root_dir = default_root_dir
        self.continue_training = continue_training
        self.log = log
        self.max_epochs = max_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.project_name = project_name
        
        self.metrics = {}
        self.best_val_metrics = None
        self._start_epoch = 0
        self._has_validation = False
        
        assert self.logger in ["wandb", "tensorboard", None], "Invalid logger. Choose between 'wandb' and 'tensorboard'."
        assert self.device in ["cpu", "cuda"], "Invalid device. Choose between 'cpu' and 'cuda'."
        assert self.log == False if self.logger == None else True, "You need to specify a logger if you want to log the training."
        assert self.optimizer != None, "You need to specify an optimizer."
        
        if logger == "wandb":
            try:
                self.wandb = wandb
                self.wandb.init(
                    project=self.project_name,
                    config={
                        "learning_rate": 5e-5,
                        "architecture": "Longformer",
                        "dataset": "Wikipedia",
                        "epochs": 500,
                    }
                )
            except:
                self.logger = None
                self.log = False
                raise Warning("You need to install wandb to use it as a logger.")
        elif logger == "tensorboard":
            raise NotImplementedError("Tensorboard logging is not implemented yet.")

    
    def _continue_training(self, model, optimizer, scheduler=None):
        """
        Loads the trained model, optimizer and scheduler from the checkpoint.
        """
        model_name = model.__class__.__name__
        checkpoint = torch.load(self.default_root_dir + 'weights/{}_*.model'.format(model_name))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        self._start_epoch = checkpoint['epoch']
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return model, optimizer, loss, scheduler

    
    def _save_checkpoint(self, model, optimizer, epoch, scheduler=None):
        """
        Saves the model, optimizer, and scheduler to the checkpoint if metrics are better.
        """
        if not self._has_validation:
            # if no validation set is provided save only the last model
            return

        if self.best_val_metrics is None or self.metrics["val_loss"] < self.best_val_metrics["val_loss"]:
            self.best_val_metrics = self.metrics
            model_name = model.__class__.__name__
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': self.metrics["val_loss"],
                "train_loss": self.metrics["train_loss"],
            }
            if scheduler:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()

            torch.save(checkpoint, f"{self.default_root_dir}weights/{model_name}_{epoch}.model")

    
    def _train_one_epoch(self, model, dataloader):
        model.train()
        training_loss = 0.0  

        with tqdm(enumerate(dataloader), total=len(dataloader)) as pbar:
            for i, data in pbar:
                self.optimizer.zero_grad()
                loss = model.train_step(data)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                training_loss += loss.item()
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    pbar.set_postfix({'Training Loss': training_loss / (i + 1)})  # Update the progress bar with the current loss

        return training_loss / len(dataloader)


    def _validate_one_epoch(self, model, dataloader):
        model.eval()
        val_loss = 0.0
        aggregated_metrics = {} 

        with torch.no_grad():
            with tqdm(dataloader, total=len(dataloader)) as pbar:
                for data in pbar:
                    val_loss += model.validation_step(data)
                    batch_metrics = self.compute_metrics(data)

                    for metric_name, metric_value in batch_metrics.items():
                        if metric_name not in aggregated_metrics:
                            aggregated_metrics[metric_name] = metric_value
                        else:
                            aggregated_metrics[metric_name] += metric_value

        num_batches = len(dataloader)
        for metric_name in aggregated_metrics:
            aggregated_metrics[metric_name] /= num_batches

        self.metrics["val_loss"] = val_loss / num_batches
        self.metrics.update(aggregated_metrics)

        return self.metrics["val_loss"]


    def fit(self, model, train_dataloaders, val_dataloaders=None):
        """
        Runs the full optimization routine.
        ### Arguments:
        - model: A PyTorch model.
        - train_dataloaders: A PyTorch dataloader.
        - val_dataloaders: A PyTorch dataloader.
        """
        if val_dataloaders:
            self._has_validation = True
        if self.continue_training:
            model, self.optimizer, _, self.scheduler = self._continue_training(model, self.optimizer, self.scheduler)
        
        for epoch in range(self._start_epoch, self.max_epochs):
            start_time = time.time()
            
            # Training
            for dataloader in train_dataloaders:
                train_loss = self._train_one_epoch(model, dataloader)
            self.metrics["train_loss"] = train_loss 
            
            # Validation
            if val_dataloaders:
                for dataloader in val_dataloaders:
                    val_loss = self._validate_one_epoch(model, dataloader)
                self.metrics["val_loss"] = val_loss
            if self.log == True:
                if self.logger == "wandb":
                    self.wandb.log(self.metrics)
                elif self.logger == "tensorboard":
                    raise NotImplementedError("Tensorboard logging is not implemented yet.")
            
            if self.scheduler:
                self.scheduler.step()
                
            self._save_checkpoint(model, self.optimizer, epoch, self.scheduler)

        model_name = model.__class__.__name__
        torch.save(model.state_dict(), self.default_root_dir + "weights/{}_final.model".format(model_name))
        
        if self.logger == "wandb":
            self.wandb.finish()
