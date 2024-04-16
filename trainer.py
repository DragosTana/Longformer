import torch
import torch.nn as nn
import wandb
import time

class Trainer():
    """
    A class to handle the training of PyTorch models.
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
        
        assert self.logger in ["wandb", "tensorboard"], "Invalid logger. Choose between 'wandb' and 'tensorboard'."
        assert self.device in ["cpu", "cuda"], "Invalid device. Choose between 'cpu' and 'cuda'."
        assert self.log == False if self.logger == None else True, "You need to specify a logger if you want to log the training."
        assert self.optimizer != None, "You need to specify an optimizer."
        
        if logger == "wandb":
            self.wandb = wandb
            try:
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
                raise ValueError("Wandb is not installed. Please install it using 'pip install wandb'.")
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

    
    def _save_model(self, model, optimizer, epoch, scheduler=None ):
        """
        Saves the model, optimizer and scheduler to the checkpoint if metrics are better.
        """
        if self.best_val_metrics is None or self.metrics["val_loss"] < self.best_val_metrics["val_loss"]:
            self.best_val_metrics = self.metrics
            model_name = model.__class__.__name__
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': self.metrics["val_loss"],
                'train_loss': self.metrics["train_loss"],
                'scheduler_state_dict': scheduler.state_dict(),
            }, self.default_root_dir + 'weights/{}_{}.model'.format(model_name, epoch))
    
    
    def _train_one_epoch(self, model, dataloader):
        model.train()
        training_loss = 0.0  
        for i, data in enumerate(dataloader):
            self.optimizer.zero_grad()
            loss = model.train_step(data)
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            training_loss += loss.item()
            if (i + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        return training_loss / len(dataloader)
    
    
    def _validate_one_epoch(self, model, dataloader):
        model.eval()
        val_loss = 0.0
        aggregated_metrics = {} 

        with torch.no_grad():
            for data in dataloader:
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
            - val_dataloaders: A dataloader.
            """
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
                    
                self._save_model(model, self.optimizer, epoch, self.metrics["val_loss"], self.scheduler)
                    
                print(f"Epoch {epoch+1}/{self.max_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Time: {time.time() - start_time:.2f}s")

                
