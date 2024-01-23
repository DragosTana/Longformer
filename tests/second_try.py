import pandas as pd
from sklearn.model_selection import train_test_split
import re
import os

from tqdm.auto import tqdm

from transformers import BertTokenizer, BertModel, get_scheduler

import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/IMDB.csv')
BERT_CHECKPOINT = 'bert-base-cased'
MAX_LEN = 128
BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCHS = 1
LEARNING_RATE = 2e-5


print('Loading data...')
data = pd.read_csv(DATA_PATH)
print('Done.')
print(" ")

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r' +', ' ', text)
    
    
    return text

print('Cleaning text...')
data['review'] = data.review.apply(clean_text)
data['sentiment'] = data.sentiment.apply(lambda x: 1 if x == 'positive' else 0)
print('Done.')
print(" ")

class IMDB(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        assert len(self.reviews) == len(self.targets)
        return len(self.reviews)
    
    def __getitem__(self, item):
        Y = torch.tensor(self.targets[item], dtype=torch.long)
        X = str(self.reviews[item])
        
        encoding = self.tokenizer(
            X,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': Y
        }
        
print('Splitting data...')
data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)
data_val, data_test = train_test_split(data_test, test_size=0.5, random_state=42)   
print('Done.')
print(" ")

data_train.reset_index(drop=True, inplace=True)
data_val.reset_index(drop=True, inplace=True)
data_test.reset_index(drop=True, inplace=True)

print('Loading tokenizer...')
tokenizer = BertTokenizer.from_pretrained(BERT_CHECKPOINT)
print('Done.')
print(" ")


print('Creating data loaders...')
train_dataset = IMDB(
    reviews=data_train.review,
    targets=data_train.sentiment,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

valid_dataset = IMDB(
    reviews=data_val.review,
    targets=data_val.sentiment,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

test_dataset = IMDB(
    reviews=data_test.review,
    targets=data_test.sentiment,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

data_loader_train = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
data_loader_valid = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
data_loader_test = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

print('Done.')
print(" ")


class SentimentClassifier(torch.nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(BERT_CHECKPOINT)
        self.drop = torch.nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(output['pooler_output'])
        return self.out(output)
    
def train_epoch(model, 
                data_loader,
                optimizer,
                scheduler,
                loss_fn,):
    losses = [] 
    accuracies = []
    
    model.train()
    
    for batch in tqdm(data_loader, total=len(data_loader)):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        _, preds = torch.max(outputs, dim=1)
        
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        accuracy = torch.sum(preds == targets) / len(targets)
        
        losses.append(loss.item())
        accuracies.append(accuracy.item())
        
    return sum(losses) / len(losses), sum(accuracies) / len(accuracies)

def eval_model(model, data_loader, loss_fn):
    losses = []
    accuracies = []
    
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            _, preds = torch.max(outputs, dim=1)
            
            loss = loss_fn(outputs, targets)
            
            accuracy = torch.sum(preds == targets) / len(targets)
            
            losses.append(loss.item())
            accuracies.append(accuracy.item())
            
    return sum(losses) / len(losses), sum(accuracies) / len(accuracies)\
        
model = SentimentClassifier(NUM_CLASSES)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
num_training_steps = len(data_loader_train) * EPOCHS
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

loss_fn = torch.nn.CrossEntropyLoss().to(device)

best_accuracy = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    
    train_loss, train_acc = train_epoch(
        model,
        data_loader_train,
        optimizer,
        scheduler,
        loss_fn
    )
    
    print(f'Train loss {train_loss} accuracy {train_acc}')
    
    val_loss, val_acc = eval_model(
        model,
        data_loader_valid,
        loss_fn
    )
    
    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()
    
    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc
