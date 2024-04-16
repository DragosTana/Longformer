from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torch
from copy import copy
from model.distil_bert import MyDistilBertForMaskedLM
from model.config import Config
from trainer import Trainer
from data import WikiDataset    
    
dataset = WikiDataset(tokenizer_name="distilbert-base-uncased", max_seq_len=512, num_workers=16, cache_dir="./data", shuffle=True)
train, test = dataset.split()
datacollator = DataCollatorForLanguageModeling(tokenizer=dataset.tokenizer, mlm=True, mlm_probability=0.15)
train_loader = DataLoader(train, batch_size=8, shuffle=True, collate_fn=datacollator)
test_loader = DataLoader(test, batch_size=8, shuffle=False, collate_fn=datacollator)

config = Config(n_layers=6, dim=768, num_attention_heads=12, vocab_size=30522)
model = MyDistilBertForMaskedLM(config)
#model_state_dict = torch.load("./model/weights/distilbert.pth")
#model.load_state_dict(model_state_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)   
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

trainer = Trainer(
    device="cuda" if torch.cuda.is_available() else "cpu",
    default_root_dir="./model/",
    optimizer=optimizer,
    scheduler=scheduler,
    compute_metrics=None,
    continue_training=False, 
    logger="wandb",
    log=True, 
    max_epochs=1,  
    gradient_accumulation_steps=1,
    project_name="prova_wikipedia",
)

trainer.fit(model, [train_loader], [test_loader])






    
        



