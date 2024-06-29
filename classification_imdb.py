from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch
from model.roberta import RobertaForSequenceClassification
from transformers import AutoModelForMaskedLM, AutoTokenizer
from model.longformer import LongformerForSequenceClassification
from model.config import RobertaConfig, LongformerConfig
from trainer import Trainer
import torch.optim as optim
from torch.utils.data import Dataset
from safetensors.torch import load_file
import re
from datasets import load_dataset
import numpy as np
import random
import wandb
import torch.nn as nn
import tqdm
from torch.cuda.amp import GradScaler, autocast

#seed = 420
#random.seed(seed)
#np.random.seed(seed)
#torch.manual_seed(seed)
#if torch.cuda.is_available():
#    torch.cuda.manual_seed(seed)
#    torch.cuda.manual_seed_all(seed)

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base", model_max_length=2048)

class IMDB(Dataset):
    def __init__(self,
             tokenizer_name: str = "distilbert-base-uncased",
             max_seq_len: int = 512,
             num_workers: int = 16,
             cache_dir: str = "./data", 
             shuffle: bool = False,
             longformer: bool = False,
             ): 
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        self.longformer = longformer
        
        if type(self.tokenizer_name) == str:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True, return_tensors="pt")
        else:
            self.tokenizer = self.tokenizer_name
        self.data = self._raw_text_to_tokens()
        print("IMDB dataset loaded and tokenized!")
        
    @staticmethod
    def clean_text(x):
        x = re.sub('<.*?>', ' ', x)
        x = re.sub('http\S+', ' ', x)
        x = re.sub('\s+', ' ', x)
        return x.lower().strip()

    def _preprocess_data(self, examples):
        examples["text"] = [self.clean_text(text) for text in examples["text"]]
        tokenized_data = self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=self.max_seq_len)
        
        if self.longformer:
            attention_mask = torch.tensor(tokenized_data["attention_mask"])
            attention_mask[:, 0] = 2
            tokenized_data["attention_mask"] = attention_mask.tolist()
            
        return tokenized_data
    
    def _raw_text_to_tokens(self):
        print("Loading IMDB dataset...")
        raw_data = load_dataset("imdb", cache_dir=self.cache_dir, trust_remote_code=True)
        tokenized_imdb = raw_data.map(self._preprocess_data, batched=True, num_proc=self.num_workers, remove_columns=["text"])
        
        return tokenized_imdb
    
    def split(self):
        return self.data["train"], self.data["test"]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

dataset = IMDB(tokenizer_name=tokenizer, 
                        max_seq_len=2048, 
                        num_workers=16, 
                        cache_dir="./data", 
                        shuffle=True,
                        longformer=False,
                        )

train, test = dataset.split()
datacollator = DataCollatorWithPadding(tokenizer=dataset.tokenizer)
train_loader = DataLoader(train, batch_size=2, shuffle=True, collate_fn=datacollator)
test_loader = DataLoader(test, batch_size=2, shuffle=True, collate_fn=datacollator)

config = LongformerConfig(vocab_size=50265,
                            num_hidden_layers=6,
                            hidden_size=768,
                            num_attention_heads=12,
                            max_position_embeddings=2050,
                            attention_window=[256]*6,
                            attention_dilation=[1]*6, 
                            num_labels=2, 
                            type_vocab_size=1,
                            )
    
    
model = LongformerForSequenceClassification(config)
state_dict = load_file("./checkpoint-3000/model.safetensors")
model.load_state_dict(state_dict, strict=False)

#config = RobertaConfig(vocab_size=50265,
#                        num_hidden_layers=6,
#                        hidden_size=768,
#                        num_attention_heads=12,
#                        max_position_embeddings=514,
#                        num_labels=2, 
#                        type_vocab_size=1,
#                        )
#model = RobertaForSequenceClassification(config)
#for p in model.parameters():
#    if p.dim() > 1:
#        torch.nn.init.xavier_uniform_(p)
#distil_roberta = AutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")
#model.load_state_dict(distil_roberta.state_dict(), strict=False)

epochs = 10
training_steps = epochs * len(train_loader) // 8
print(f"Total training steps: {training_steps}")
warmup_steps = 0.1 * training_steps    

optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
scheduler = optim.lr_scheduler.PolynomialLR(optimizer, training_steps, 1.0)

def compute_metrics(data, output):
    loss, logits = output
    preds = torch.argmax(logits, dim=1)
    return {"accuracy": (preds == data["labels"]).float().mean().item()}

trainer = Trainer(
    device="cuda" if torch.cuda.is_available() else "cpu",
    default_root_dir="./model/",
    optimizer=optimizer,
    scheduler=scheduler,
    compute_metrics=compute_metrics,
    logger="wandb",
    log=True,
    max_epochs=epochs,
    use_mixed_precision=True,
    gradient_accumulation_steps=8, 
    warmup_steps=warmup_steps,
    val_check_interval=500,
    project_name="Classification_IMDB",
)

trainer.train(model, train_loader, test_loader)