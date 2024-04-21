from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch
from model.distil_bert import MyDistiBertClassification
from model.config import ConfigClassification
from trainer import Trainer
from data import IMDB
    
dataset = IMDB(tokenizer_name="distilbert-base-uncased", max_seq_len=512, num_workers=16, cache_dir="./data", shuffle=True)
train, test = dataset.split()
datacollator = DataCollatorWithPadding(tokenizer=dataset.tokenizer)
train_loader = DataLoader(train, batch_size=8, shuffle=True, collate_fn=datacollator)
test_loader = DataLoader(test, batch_size=8, shuffle=False, collate_fn=datacollator)

config = ConfigClassification(n_layers=6, dim=768, num_attention_heads=12, vocab_size=30522, num_labels=2)
model = MyDistiBertClassification(config, num_labels=2)
model_state_dict = torch.load("./model/weights/distilbert.pth")
model.load_state_dict(model_state_dict, strict=False)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-05, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)    

trainer = Trainer(
    device="cuda" if torch.cuda.is_available() else "cpu",
    default_root_dir="./model/",
    optimizer=optimizer,
    scheduler=scheduler,
    compute_metrics=None,
    continue_training=False, 
    logger="wandb",
    log=True, 
    max_epochs=2,  
    gradient_accumulation_steps=2,
    project_name="Classification_IMDB",
)

trainer.fit(model, [train_loader], [test_loader])
