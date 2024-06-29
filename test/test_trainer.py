        
if __name__ == "__main__":
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from trainer import Trainer  # Assuming the Trainer class is in a file named trainer.py

    # Define a simple neural network model
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = x.view(-1, 784)  # Flatten the input
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return self.softmax(x)

        def train_step(self, batch):
            data, target = batch
            output = self(data)
            loss = nn.CrossEntropyLoss()(output, target)
            return loss
        
        def validation_step(self, batch):
            data, target = batch
            output = self(data)
            loss = nn.CrossEntropyLoss()(output, target)
            return loss

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    dataset_len = len(dataset)  
    train_len = int(0.8 * dataset_len)
    val_len = dataset_len - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    

    # Initialize the model, optimizer, and scheduler
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=4000, power=1.0)

    def compute_metrics(batch):
        data, target = batch
        output = model(data)
        _, preds = output.max(dim=1)
        correct = (preds == target).sum().item()
        return {"accuracy": correct / len(target)}
    
    # Create an instance of the Trainer class and start training
    trainer = Trainer(
        device="cuda" if torch.cuda.is_available() else "cpu",
        default_root_dir="./model/",
        optimizer=optimizer,
        scheduler=None,
        compute_metrics=compute_metrics,
        continue_training=False, 
        logger="wandb",
        log=True, 
        max_epochs=40,  
        gradient_accumulation_steps=1,
        warmup_steps=300,
        project_name="prova_MNIST",
    )

    # Start the training process
    trainer.fit(model, [train_loader], [val_loader])