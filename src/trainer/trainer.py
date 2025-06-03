import os
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import wandb
from tqdm import tqdm
from typing import Optional, Callable, List, Dict, Any, Union
from torch.cuda.amp import GradScaler, autocast
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class WarmupScheduler(_LRScheduler):
    """Custom learning rate scheduler with linear warmup."""
    
    def __init__(self, optimizer: Optimizer, warmup_steps: int, after_scheduler: _LRScheduler, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.after_scheduler = after_scheduler
        self.finished_warmup = False
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_factor = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            if not self.finished_warmup:
                self.after_scheduler.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
                self.finished_warmup = True
            return self.after_scheduler.get_last_lr()

    def step(self, epoch=None):
        if self.finished_warmup:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.warmup_steps)
        else:
            return super(WarmupScheduler, self).step(epoch)
            
class Trainer:

    def __init__(self,
                 device: str,
                 default_root_dir: str,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 compute_metrics: Optional[Callable[[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, float]]] = None,
                 logger: Optional[str] = None,
                 log: bool = False,
                 max_epochs: int = 500,
                 gradient_accumulation_steps: int = 1,
                 warmup_steps: Optional[int] = None,
                 save_every_n_steps: Optional[int] = None,
                 val_check_interval: Optional[int] = None,
                 evaluate_first: Optional[bool] = False,
                 max_grad_norm: Optional[float] = None,
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
            evaluate_first (bool, optional): Whether to evaluate the model before training.
            gradient_accumulation_steps (int, optional): The number of gradient accumulation steps.
            max_grad_norm (float, optional): The maximum gradient norm for gradient clipping.
            use_mixed_precision (bool, optional): Whether to use mixed precision training.
            project_name (str, optional): The name of the project for logging.
        """
        self.device = device
        self.default_root_dir = default_root_dir
        self.optimizer = optimizer
        self.scheduler = WarmupScheduler(optimizer, warmup_steps=warmup_steps, after_scheduler=scheduler) if warmup_steps!=None else scheduler
        self.compute_metrics = compute_metrics
        self.log = log
        self.max_epochs = max_epochs
        self.save_every_n_steps = save_every_n_steps
        self.val_check_interval = val_check_interval
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_mixed_precision = use_mixed_precision
        self.evaluate_first = evaluate_first
        self.warmup_steps = warmup_steps
        self.scaler = GradScaler() if self.use_mixed_precision else None
        self.project_name = project_name
        self.epoch = 0

        self.metrics = {}
        self.best_val_metrics = None
        self._start_epoch = 0
        self._has_test = False
        self._model_name = None

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


    def _save_checkpoint(self, model, optimizer, epoch, scheduler=None):
        """Save the model checkpoint."""
        model_name = model.__class__.__name__
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        os.makedirs(os.path.join(self.default_root_dir, 'weights'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.default_root_dir, 'weights', f'{model_name}_{epoch}.model'))


    def _train_one_epoch(self, model, trainloader, testloader):
        """Train the model for one epoch."""
        model.train()
        training_loss = 0.0
        accumulated_loss = 0.0
        total_steps = len(trainloader) // self.gradient_accumulation_steps
        with tqdm(total=total_steps) as pbar:
            for i, data in enumerate(trainloader):
                
                # Move data to the device
                if isinstance(data, list):
                    data = [item.to(self.device) for item in data]
                elif isinstance(data, dict):
                    data = {key: value.to(self.device) for key, value in data.items()}
                else:
                    data = data.to(self.device)
                
                # Forward pass
                with autocast(enabled=self.use_mixed_precision, dtype=torch.float16):
                    output = model.train_step(data)
                    loss = output[0] if isinstance(output, tuple) else output
                    loss /= self.gradient_accumulation_steps
                    
                # Backward pass
                if self.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
    
                real_loss = loss.item() * self.gradient_accumulation_steps
                accumulated_loss += real_loss
                
                # Update the weights
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
                    pbar.update(1)
                      
                    if self.scheduler:
                        self.scheduler.step()

                    if self.log:
                        self._log_metrics({"Training Loss": real_loss, 
                                            "Learning Rate": self.optimizer.param_groups[0]['lr']})
                pbar.set_postfix({'Training Loss': real_loss})
                
                #if (self.epoch + i+1) % self.val_check_interval == 0:
                #    if self._has_test:
                #        test_loss, test_metrics = self.test(model, testloader)
                #        self._log_metrics({"Test Loss": test_loss})
                #        self._log_metrics(test_metrics)
                
            training_loss = accumulated_loss / len(trainloader)
        self.optimizer.zero_grad()
        return training_loss
    
    def test(self, model, dataloader):
        """Test the model for one epoch."""
        model.eval()
        total_loss = 0.0
        total_metrics = {}

        with torch.no_grad():
            with tqdm(total=len(dataloader)) as pbar:
                for i, data in enumerate(dataloader):
                    # Move data to the device
                    if isinstance(data, list):
                        data = [item.to(self.device) for item in data]
                    elif isinstance(data, dict):
                        data = {key: value.to(self.device) for key, value in data.items()}
                    else:
                        data = data.to(self.device)

                    # Forward pass
                    with autocast(enabled=self.use_mixed_precision, dtype=torch.float16):
                        output = model.test_step(data)
                        loss = output[0] if isinstance(output, tuple) else output  # Loss is the first element in the output tuple
                        total_loss += loss.item()  # Accumulate the loss

                    # Compute metrics
                    metrics = self.compute_metrics(data, output) if self.compute_metrics else {}
                    for key, value in metrics.items():
                        if key not in total_metrics:
                            total_metrics[key] = 0.0
                        total_metrics[key] += value

                    pbar.update(1)

                    # Update progress bar with current average loss
                    avg_loss = total_loss / (i + 1)
                    avg_metrics = {key: total_metrics[key] / (i + 1) for key in total_metrics}
                    pbar.set_postfix({'Test Loss': avg_loss})

        # Calculate average loss and metrics for the epoch
        avg_loss = total_loss / len(dataloader)
        avg_metrics = {key: total_metrics[key] / len(dataloader) for key in total_metrics}
        
        # add prefix to metrics
        avg_metrics = {f"Test {key}": value for key, value in avg_metrics.items()}

        return avg_loss, avg_metrics
    

    def _log_metrics(self, metrics: Dict[str, float]):
        if self.logger:
            self.logger.log(metrics)

    def train(self, model, train_dataloader, test_dataloader=None):
        
        batch_size = train_dataloader.batch_size
        steps_per_epoch = len(train_dataloader) // self.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.max_epochs
        
        if test_dataloader:
            self._has_test = True
        model.to(self.device)
        self._model_name = model.__class__.__name__

        if self.evaluate_first:
            first_train_loss, _ = self.test(model, train_dataloader)
            self._log_metrics({"Training Loss per epoch": first_train_loss})
            first_test_loss, first_test_metrics = self.test(model, test_dataloader)
            self._log_metrics({"Test Loss": first_test_loss})
            self._log_metrics(first_test_metrics)
        
        for epoch in range(self._start_epoch, self.max_epochs):
            print("Epoch:", epoch)
            
            train_loss = self._train_one_epoch(model, train_dataloader, test_dataloader)
            self._log_metrics({"Training Loss per epoch": train_loss})
            self.epoch += 1
            
            if self._has_test:
                test_loss, test_metrics = self.test(model, test_dataloader)
                self._log_metrics({"Test Loss": test_loss})
                self._log_metrics(test_metrics)

            self._save_checkpoint(model, self.optimizer, epoch, self.scheduler)

        model_name = model.__class__.__name__
        torch.save(model.state_dict(), os.path.join(self.default_root_dir, 'weights', f'{model_name}_final.model'))

        if self.logger == "wandb" and self.log:
            self.logger.finish()
