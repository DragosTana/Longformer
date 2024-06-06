import os
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import wandb
from tqdm import tqdm
from typing import Optional, Callable, List, Dict, Any, Union
from torch.cuda.amp import GradScaler, autocast

class Trainer:
    """
    A class to handle the training of PyTorch models.
    """

    def __init__(self,
                 device: str,
                 default_root_dir: str,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 compute_metrics: Optional[Callable[[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, float]]] = None,
                 continue_training: bool = False,
                 logger: Optional[str] = None,
                 log: bool = False,
                 max_epochs: int = 500,
                 gradient_accumulation_steps: int = 1,
                 max_grad_norm : float = None,
                 use_mixed_precision: bool = False,
                 project_name: str = "my-awesome-project"):
        """
        Initialize the Trainer class.

        Args:
            device (str): The device to use for training ('cpu' or 'cuda').
            default_root_dir (str): The default root directory to save model weights.
            optimizer (torch.optim.Optimizer): The optimizer for the model.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler.
            compute_metrics (Callable[[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, float]], optional): A function to compute metrics.
            continue_training (bool, optional): Whether to continue training from the last checkpoint.
            logger (str, optional): The logger to use for logging ('wandb' or 'tensorboard').
            log (bool, optional): Whether to log the training.
            max_epochs (int, optional): The maximum number of epochs to train the model.
            gradient_accumulation_steps (int, optional): The number of gradient accumulation steps.
            max_grad_norm (float, optional): The maximum gradient norm for gradient clipping.
            use_mixed_precision (bool, optional): Whether to use mixed precision training.
            project_name (str, optional): The name of the project for logging.
        """
        self.device = device
        self.default_root_dir = default_root_dir
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.compute_metrics = compute_metrics
        self.continue_training = continue_training
        self.log = log
        self.max_epochs = max_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_mixed_precision = use_mixed_precision
        self.scaler = GradScaler() if self.use_mixed_precision else None
        self.project_name = project_name

        self.metrics = {}
        self.best_val_metrics = None
        self._start_epoch = 0
        self._has_validation = False

        assert logger in ["wandb", "tensorboard", None], "Invalid logger. Choose between 'wandb' and 'tensorboard'."
        assert self.device in ["cpu", "cuda"], "Invalid device. Choose between 'cpu' and 'cuda'."
        assert not (self.log and logger is None), "You need to specify a logger if you want to log the training."

        if self.log:
            self.logger = self._initialize_logger(logger)
        else:    
            self.logger = None


    def _initialize_logger(self, logger: str):
        if logger == "wandb":
            try:
                wandb.init(
                    project=self.project_name,
                    config={
                        "learning_rate": self.optimizer.defaults['lr'],
                        "architecture": "CustomModel",
                        "dataset": "YourDataset",
                        "epochs": self.max_epochs,
                    }
                )
                return wandb
            except ImportError:
                self.log = False
                raise ImportError("You need to install wandb to use it as a logger.")
        elif logger == "tensorboard":
            raise NotImplementedError("Tensorboard logging is not implemented yet.")
        return None


    def _continue_training(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                           scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Union[nn.Module, torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
        """
        Loads the trained model, optimizer, and scheduler from the checkpoint.
        """
        model_name = model.__class__.__name__
        checkpoint_path = os.path.join(self.default_root_dir, 'weights', f'{model_name}_*.model')

        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self._start_epoch = checkpoint['epoch']
            if scheduler:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            return model, optimizer, checkpoint['loss'], scheduler
        except FileNotFoundError:
            print(f"No checkpoint found at {checkpoint_path}")
            return model, optimizer, None, scheduler


    def _save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
                         scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        """
        Saves the model, optimizer, and scheduler to the checkpoint if metrics are better.
        """
        if not self._has_validation:
            # If no validation set is provided save only the last model
            return
        if self.best_val_metrics:
            print(f"Best validation loss: {self.best_val_metrics['val_loss']}")
            print(f"Current validation loss: {self.metrics['val_loss']}")
        if ((self.best_val_metrics is None) or (self.metrics["val_loss"] < self.best_val_metrics["val_loss"])):
            print(f"Saving the model at epoch {epoch}")
            self.best_val_metrics = self.metrics.copy()
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

            os.makedirs(os.path.join(self.default_root_dir, 'weights'), exist_ok=True)
            torch.save(checkpoint, os.path.join(self.default_root_dir, 'weights', f'{model_name}_{epoch}.model'))


    def _train_one_epoch(self, model, dataloader):
        model.train()
        training_loss = 0.0
        with tqdm(enumerate(dataloader), total=len(dataloader)) as pbar:
            for i, data in pbar:
                data = {key: value.to(self.device) for key, value in data.items()}
                self.optimizer.zero_grad()
                with autocast(enabled=self.use_mixed_precision, dtype=torch.float16):
                    loss, _ = model.train_step(data)
                    loss = loss / self.gradient_accumulation_steps
                if self.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                training_loss += loss.item()
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    if self.use_mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                        if self.max_grad_norm is not None:
                            nn_utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        if self.max_grad_norm is not None:
                            nn_utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                pbar.set_postfix({'Training Loss': training_loss / (i + 1)})
                if self.log:
                    self._log_metrics({"Batch Loss": training_loss / (i + 1)})
        return training_loss / len(dataloader)


    def _validate_one_epoch(self, model, dataloader):
        model.eval()
        val_loss = 0.0
        aggregated_metrics = {}

        with torch.no_grad():
            with tqdm(dataloader, total=len(dataloader)) as pbar:
                for data in pbar:
                    data = {key: value.to(self.device) for key, value in data.items()}
                    with autocast(enabled=self.use_mixed_precision):
                        loss, outputs = model.validation_step(data)
                    val_loss += loss.item()
                    batch_metrics = self.compute_metrics(data, outputs) if self.compute_metrics else {}

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


    def _log_metrics(self, metrics: Dict[str, float]):
        if self.logger:
            self.logger.log(metrics)


    def fit(self, model: nn.Module, train_dataloaders: List[torch.utils.data.DataLoader],
            val_dataloaders: Optional[List[torch.utils.data.DataLoader]] = None):
        """
        Runs the full optimization routine.
        ### Arguments:
        - model: A PyTorch model.
        - train_dataloaders: A list of PyTorch dataloaders for training.
        - val_dataloaders: A list of PyTorch dataloaders for validation.
        """
        if val_dataloaders:
            self._has_validation = True
        if self.continue_training:
            model, self.optimizer, _, self.scheduler = self._continue_training(model, self.optimizer, self.scheduler)
        model.to(self.device)

        for epoch in range(self._start_epoch, self.max_epochs):
            # Training
            for dataloader in train_dataloaders:
                train_loss = self._train_one_epoch(model, dataloader)
            self.metrics["train_loss"] = train_loss

            # Validation
            if val_dataloaders:
                for dataloader in val_dataloaders:
                    val_loss = self._validate_one_epoch(model, dataloader)
                self.metrics["val_loss"] = val_loss
            if self.log:
                self._log_metrics(self.metrics)

            if self.scheduler:
                self.scheduler.step()

            self._save_checkpoint(model, self.optimizer, epoch, self.scheduler)

        model_name = model.__class__.__name__
        torch.save(model.state_dict(), os.path.join(self.default_root_dir, 'weights', f'{model_name}_final.model'))

        if self.logger == "wandb" and self.log:
            self.logger.finish()
