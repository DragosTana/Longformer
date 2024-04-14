from datasets import load_from_disk, load_dataset
from transformers import LongformerTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn import CrossEntropyLoss

from transformers import AutoTokenizer
from transformers import RobertaForMaskedLM, RobertaConfig

from model.longformer import LongformerForMaskedLM
from model.config import TransformerConfig
import wandb    
from tqdm import tqdm
from copy import copy

def process_data(data, tokenizer):
    return tokenizer(["".join(x) for x in data["text"]])

def group_texts(examples, max_seq_len=128, pad_token_id=1):
    """Group texts together in a dataset so that they match the block size.
    Add padding to the end of the texts if there are not enough."""
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # Pad the sequences to ensure uniform length.
    remainder = total_length % max_seq_len
    if remainder != 0:
        padding_length = max_seq_len - remainder
        for k, t in concatenated_examples.items():
            concatenated_examples[k] = t + [pad_token_id] * padding_length
        total_length += padding_length
    concatenated_examples["attention_mask"][-padding_length:] = [0] * padding_length
    # Split by chunks of max_seq_len.
    result = {
        k: [t[i : i + max_seq_len] for i in range(0, total_length, max_seq_len)]
        for k, t in concatenated_examples.items()
    }
    return result

def raw_text_to_tokens(dataset: str = "wikipedia",
                       tokenizer: str = "roberta-base",
                       max_seq_len: int = 128,
                       num_workers: int = 16,
                       cache_dir: str = "./data", 
                       shuffle: bool = True
                    ): 
    print(f"Loading {dataset} dataset...")
    
    if dataset == "wikipedia":
        raw_data = load_dataset("wikipedia", "20220301.en", split="train", cache_dir=cache_dir, trust_remote_code=True)
        text_data = raw_data.remove_columns(['id', 'url', 'title',])
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True, return_tensors="pt")
    text_data.shuffle() if shuffle else None
    text_data = text_data.select(range(10))
    
    text_data = text_data.map(process_data, batched=True, num_proc=num_workers, remove_columns=text_data.column_names, fn_kwargs={"tokenizer": tokenizer})
    dataset = text_data.map(group_texts, batched=True, num_proc=num_workers, remove_columns=text_data.column_names, fn_kwargs={"max_seq_len": max_seq_len, "pad_token_id": tokenizer.pad_token_id})

    return dataset

class WikiDataset(Dataset):
    def __init__(self,
                 tokenizer: str = "roberta-base",
                 max_seq_len: int = 128,
                 num_workers: int = 16,
                 cache_dir: str = "./data", 
                 shuffle: bool = True
                ): 
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        
        self.data = raw_text_to_tokens(dataset="wikipedia", tokenizer=self.tokenizer, max_seq_len=self.max_seq_len, num_workers=self.num_workers, cache_dir=self.cache_dir, shuffle=self.shuffle)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        labels = copy(self.data[idx]["input_ids"])
        return {"input_ids": self.data[idx]["input_ids"], "attention_mask": self.data[idx]["attention_mask"], "labels": labels}
        
dataset = WikiDataset(tokenizer="roberta-base", max_seq_len=64, num_workers=16, cache_dir="./data")
tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True, return_tensors="pt")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=4, collate_fn=data_collator, shuffle=True)

print(dataset)


batch = next(iter(dataloader))

MODEL_PATH = "./runs/"

#wandb.init(
#    project="my-awesome-project",
#    config={
#        "learning_rate": 5e-5,
#        "architecture": "Longformer",
#        "dataset": "Wikipedia",
#        "epochs": 500,
#    }
#)

#config = TransformerConfig(vocab_size=tokenizer.vocab_size, 
#                           num_hidden_layers=6,
#                           num_attention_heads=8,
#                            model_dim=512,)

config = TransformerConfig(vocab_size=tokenizer.vocab_size,)

#config = RobertaConfig(vocab_size=tokenizer.vocab_size,
#                       num_hidden_layers=6,
#                       num_attention_heads=8,
#                       hidden_size=512,)

#model = RobertaForMaskedLM(config)
model = LongformerForMaskedLM(config)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
model.to(device)
epochs = 50
criterion = torch.nn.CrossEntropyLoss()
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    outputs = model(input_ids)
    loss = criterion(outputs.view(-1, tokenizer.vocab_size), labels.view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item()}")
   
#wandb.finish()

print("Saving model...")
torch.save(model.state_dict(), MODEL_PATH + "model.pth")

print("Loading model...")
new_model = LongformerForMaskedLM(config)
new_model.load_state_dict(torch.load(MODEL_PATH + "model.pth"))
new_model.eval()
new_model.to(device)

natural_language_sentence = tokenizer.decode(batch["input_ids"][0])
print(f"Sentence: {natural_language_sentence}")

token_logits = new_model(batch["input_ids"][0].unsqueeze(0).to(device))
mask_token_index = torch.where(batch["input_ids"][0] == tokenizer.mask_token_id)[0]
#select the mask tokens
mask_token_logits = token_logits[0, mask_token_index, :]
top_5_tokens = torch.topk(mask_token_logits, 1, dim=1).indices

labels = batch["labels"][0][mask_token_index].tolist()
predictions = top_5_tokens.tolist()

for i, (label, prediction) in enumerate(zip(labels, predictions)):
    print(f"Original: {tokenizer.decode([label])} - Prediction: {tokenizer.decode(prediction)}")
    
    