import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import DataCollatorWithPadding
from tqdm import tqdm

from data import IMDB

# Define the custom model
class NLPClassifier(nn.Module):
    def __init__(self, transformer_model_name, num_classes):
        super(NLPClassifier, self).__init__()
        
        self.transformer = AutoModel.from_pretrained(transformer_model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = transformer_outputs.last_hidden_state
        pooled_output = hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        return logits


# Initialize the dataset
dataset = IMDB(tokenizer_name="distilbert-base-uncased", max_seq_len=512, num_workers=16, cache_dir="./data", shuffle=True)
train_dataset, test_dataset = dataset.split()

# Define dataloaders
datacollator = DataCollatorWithPadding(tokenizer=dataset.tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=datacollator)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=datacollator)

# Initialize the model, loss function, and optimizer
model = NLPClassifier(transformer_model_name="distilbert-base-uncased", num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Training loop
num_epochs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions.double() / len(train_loader.dataset)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

# Evaluation loop
model.eval()
eval_loss = 0.0
correct_predictions = 0

with torch.no_grad():
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)

        eval_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels)

eval_loss = eval_loss / len(test_loader)
eval_accuracy = correct_predictions.double() / len(test_loader.dataset)

print(f"Test Loss: {eval_loss:.4f}, Test Accuracy: {eval_accuracy:.4f}")
