from transformers import DataCollatorWithPadding, DistilBertForMaskedLM
from torch.utils.data import DataLoader
import torch
from model.longformer import LongformerForSequenceClassification
from model.roberta import RobertaForSequenceClassification
from model.config import LongformerConfig
from transformers import RobertaConfig, AutoModelForMaskedLM, AutoTokenizer
from trainer import Trainer
from data import IMDB

from safetensors.torch import load_file
      
#distil_roberta = AutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base", model_max_length=2048)
#config = RobertaConfig.from_pretrained("distilbert/distilroberta-base")
#model = RobertaForSequenceClassification(config)
#model.load_state_dict(distil_roberta.state_dict(), strict=False)
#print(model)

dataset = IMDB(tokenizer_name=tokenizer, 
               max_seq_len=2048, 
               num_workers=16, 
               cache_dir="./data", 
               shuffle=True, 
               longformer=True,)

train, test = dataset.split()
datacollator = DataCollatorWithPadding(tokenizer=dataset.tokenizer)
train_loader = DataLoader(train, batch_size=2, shuffle=True, collate_fn=datacollator)
test_loader = DataLoader(test, batch_size=2, shuffle=False, collate_fn=datacollator)

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

optimizer = torch.optim.Adam(model.parameters(), lr=5e-05, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

def compute_metrics(data, outputs):
    predictions = torch.argmax(outputs, dim=-1)
    return {"accuracy": (predictions == data["labels"]).float().mean()}

trainer = Trainer(
    device="cuda" if torch.cuda.is_available() else "cpu",
    default_root_dir="./model/",
    optimizer=optimizer,
    scheduler=scheduler,
    compute_metrics=compute_metrics,
    continue_training=False, 
    logger="wandb",
    log=True, 
    max_epochs=1,  
    use_mixed_precision=False,
    gradient_accumulation_steps=8,
    project_name="Classification_IMDB",
)

trainer.fit(model, [train_loader], [test_loader])