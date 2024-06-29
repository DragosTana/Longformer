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
import copy
from sklearn.metrics import f1_score

#seed = 420
#random.seed(seed)
#np.random.seed(seed)
#torch.manual_seed(seed)
#if torch.cuda.is_available():
#    torch.cuda.manual_seed(seed)
#    torch.cuda.manual_seed_all(seed)

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base", model_max_length=2048)

class Hyperpartisan(Dataset):
    def __init__(self,
                 tokenizer_name: str = "distilbert-base-uncased",
                 max_seq_len: int = 512,
                 num_workers: int = 16,
                 cache_dir: str = "./data", 
                 shuffle: bool = False,
                 longformer: bool = False,
                ): 
        """
        Initializes the Hyperpartisan dataset.

        Args:
            tokenizer_name (str): Name of the tokenizer.
            max_seq_len (int): Maximum sequence length for tokenization.
            num_workers (int): Number of workers for data processing.
            cache_dir (str): Directory to cache the dataset.
            shuffle (bool): Whether to shuffle the dataset.
        """
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
        print("Hyperpartisan dataset loaded and tokenized!")
    
    @staticmethod
    def clean_text(text):
        cleaned_text = re.sub(r"[a-zA-Z]+\/[a-zA-Z]+", " ", text)
        cleaned_text = re.sub(r"\n", " ", cleaned_text)
        cleaned_text = re.sub(r"&#160;", "", cleaned_text)
        cleaned_text = re.sub(r'<.*?>', '', cleaned_text)
        
        #remove urls
        cleaned_text = re.sub(r'http\S+', '', cleaned_text)
        cleaned_text = re.sub(r'www\S+', '', cleaned_text)
        cleaned_text = re.sub(r'href\S+', '', cleaned_text)
        
        #remove multiple spaces
        cleaned_text = re.sub(r"[ \s\t\n]+", " ", cleaned_text)
        
        #remove repetitions
        cleaned_text = re.sub(r"([!?.]){2,}", r"\1", cleaned_text)
        cleaned_text = re.sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2", cleaned_text)
        
        return cleaned_text
        
    def _preprocess_data(self, examples):
        examples["text"] = [self.clean_text(text) for text in examples["text"]]
        tokenized_examples = self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=self.max_seq_len)
        tokenized_examples["labels"] = examples["hyperpartisan"]
        tokenized_examples["labels"] = [int(label) for label in tokenized_examples["labels"]]
        if self.longformer:
            attention_mask = torch.tensor(tokenized_examples["attention_mask"])
            attention_mask[:, 0:3] = 2
            tokenized_examples["attention_mask"] = attention_mask.tolist()
            
        return tokenized_examples
    
    def _raw_text_to_tokens(self):
        print("Loading Hyperpartisan News Detection dataset...")
        raw_data = load_dataset("SemEvalWorkshop/hyperpartisan_news_detection", "byarticle", cache_dir=self.cache_dir, trust_remote_code=True)
        raw_data = raw_data.remove_columns(['title', 'url', 'published_at'])
        tokenized_data = raw_data.map(self._preprocess_data, batched=True, num_proc=self.num_workers, remove_columns=["text", "hyperpartisan"])
        return tokenized_data["train"]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def split(self, split_ratio: float = 0.8):
        train_size = int(split_ratio * len(self))
        test_size = len(self) - train_size
        return torch.utils.data.random_split(self, [train_size, test_size])

dataset = Hyperpartisan(tokenizer_name=tokenizer,
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
for p in model.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)
state_dict = load_file("./checkpoint-3000/model.safetensors")
model.load_state_dict(state_dict)

for i, layer in enumerate(model.roberta.encoder.layer):
    layer.attention.self.query_global = copy.deepcopy(layer.attention.self.query)
    layer.attention.self.key_global = copy.deepcopy(layer.attention.self.key)
    layer.attention.self.value_global = copy.deepcopy(layer.attention.self.value)

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
    labels = data["labels"]
    accuracy = (preds == labels).float().mean()
    f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="weighted")
    return {"accuracy": accuracy.item(), "f1": f1}

trainer = Trainer(
    device="cuda" if torch.cuda.is_available() else "cpu",
    default_root_dir="./model/",
    optimizer=optimizer,
    scheduler=scheduler,
    compute_metrics=compute_metrics,
    logger="wandb",
    log=False,
    max_epochs=epochs,
    use_mixed_precision=True,
    gradient_accumulation_steps=8, 
    warmup_steps=warmup_steps,
    val_check_interval=500,
    project_name="Classification_Hyperpartisan",
)

trainer.train(model, train_loader, test_loader)