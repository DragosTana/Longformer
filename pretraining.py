from transformers import DataCollatorForLanguageModeling
from transformers import DistilBertForMaskedLM
from torch.utils.data import DataLoader
import torch
from model.longformer import LongformerForMaskedLM
from model.config import LongformerConfig  
from trainer import Trainer
from data import WikiDataset    
import collections
from transformers import get_linear_schedule_with_warmup
    
dataset = WikiDataset(tokenizer_name="distilbert-base-uncased", max_seq_len=2048, num_workers=16, cache_dir="./data", shuffle=True, n_docs=100)
train, test = dataset.split()
datacollator = DataCollatorForLanguageModeling(tokenizer=dataset.tokenizer, mlm=True, mlm_probability=0.15)
train_loader = DataLoader(train, batch_size=2, shuffle=True, collate_fn=datacollator)
test_loader = DataLoader(test, batch_size=2, shuffle=False, collate_fn=datacollator)

#select only the first batch
train_loader = [next(iter(train_loader))]

config = LongformerConfig(n_layers=6, 
                          dim=768, 
                          num_attention_heads=12,
                          max_position_embeddings=2048,
                          activation="gelu",
                          dropout=0.2,
                          attention_window=[32, 64, 64, 128, 256, 256],
                          attention_dilation=[1]*6,
                          vocab_size=30522)
model = LongformerForMaskedLM(config)

#try:
#    distil_bert_state_dict = torch.load("./model/weights/distilbert.pth")  
#except:
#    model_hf = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased", cache_dir="./model/weights/")
#    torch.save(model_hf.state_dict(), "./model/weights/distilbert.pth")
#    distil_bert_state_dict = torch.load("./model/weights/distilbert.pth")
#
## Copy the position embeddings
#position_embeddings = distil_bert_state_dict["distilbert.embeddings.position_embeddings.weight"]
#if config.max_position_embeddings % position_embeddings.shape[0] != 0:
#    raise ValueError("The max_position_embeddings is not a multiple of the pretrained position embeddings")    
#n = int(config.max_position_embeddings / position_embeddings.shape[0])
#position_embeddings = position_embeddings.repeat(n, 1)
#model_state_dict = model.state_dict()
#model_state_dict["distilbert.embeddings.position_embeddings.weight"] = position_embeddings
#model.load_state_dict(model_state_dict)
#
## copy the rest of the weights
#new_state_dict = collections.OrderedDict()
#for k, v in distil_bert_state_dict.items():
#    if k in model.state_dict() and v.size() == model.state_dict()[k].size():
#        new_state_dict[k] = v
#model.load_state_dict(new_state_dict, strict=False)

state_dict = torch.load("./model/longformer/LongDistilBertForMaskedLM.model")
model.load_state_dict(state_dict)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=1000)

trainer = Trainer(
    device="cuda" if torch.cuda.is_available() else "cpu",
    default_root_dir="./model/",
    optimizer=optimizer,
    scheduler=scheduler,
    compute_metrics=None,
    continue_training=False, 
    logger="wandb",
    log=False, 
    max_epochs=1000,  
    save_every_n_steps=None,
    gradient_accumulation_steps=1,
    max_grad_norm=None,
    use_mixed_precision=True,
    project_name="MLM_wikipedia",
)

trainer.fit(model, [train_loader])






    
        



