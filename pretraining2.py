import logging
import os
import math
import copy
from typing import Tuple
import torch
from dataclasses import dataclass, field
from transformers import DistilBertForMaskedLM, DistilBertTokenizerFast, DataCollatorForLanguageModeling, Trainer, TrainingArguments, HfArgumentParser
from transformers import DistilBertConfig, PretrainedConfig
from transformers import AutoTokenizer, AutoModelForMaskedLM, RobertaForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
from model.longformer_attention import LongformerSelfAttention
from transformers import AutoTokenizer
from data import WikiDataset    
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from typing import Optional, Union
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

class RobertaLongSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        return super().forward(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)


class RobertaLongForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.roberta.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = RobertaLongSelfAttention(config, layer_id=i)
        
def create_long_model(save_model_to, attention_window, max_pos):
    model = AutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base", return_dict=True)
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base", model_max_length=max_pos)
    config = model.config   
    
       # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.roberta.embeddings.position_embeddings.weight.shape
    max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
    config.max_position_embeddings = max_pos
    config.attention_window = [attention_window] * config.num_hidden_layers
    config.attention_dilation = [1] * config.num_hidden_layers
    config.attention_mode = 'sliding_chunks'
    config.autoregressive = False
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.roberta.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_pos_embed[k:(k + step)] = model.roberta.embeddings.position_embeddings.weight[2:]
        k += step
    model.roberta.embeddings.position_embeddings.weight.data = new_pos_embed
    model.roberta.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.roberta.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = copy.deepcopy(layer.attention.self.query)
        longformer_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
        longformer_self_attn.value_global = copy.deepcopy(layer.attention.self.value)

        layer.attention.self = longformer_self_attn

    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer

def copy_proj_layers(model):
    for i, layer in enumerate(model.distilbert.transformer.layer):
        layer.attention.query_global = copy.deepcopy(layer.attention.q_lin)
        layer.attention.key_global = copy.deepcopy(layer.attention.k_lin)
        layer.attention.value_global = copy.deepcopy(layer.attention.v_lin)
    return model
    
@dataclass
class ModelArguments:
    attention_window: int = field(default=256, metadata={"help": "Size of attention window"})
    max_pos: int = field(default=2048, metadata={"help": "Maximum position"})
    
parser = HfArgumentParser((TrainingArguments, ModelArguments))

training_args, model_args = parser.parse_args_into_dataclasses(look_for_args_file=False, args=[
                    '--output_dir', 'long_distilbert',
                    '--warmup_steps', '500',
                    '--learning_rate', '0.00003',
                    '--weight_decay', '0.01',
                    '--adam_epsilon', '1e-6',
                    '--max_steps', '3000',
                    '--logging_steps', '100',
                    '--save_steps', '500',
                    '--max_grad_norm', '5.0',
                    '--per_gpu_eval_batch_size', '2',
                    '--per_gpu_train_batch_size', '2', 
])

model_path = training_args.output_dir
logger.info(f'Converting distilbert to longformer with attention window {model_args.attention_window} and max pos {model_args.max_pos}')

#model, tokenizer = create_long_model(model_path, model_args.attention_window, model_args.max_pos)
model = RobertaLongForMaskedLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print(model)

#model = AutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base", return_dict=True)
#tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")

#def random_initialization(model):
#    for param in model.parameters():
#        if param.dim() > 1:
#            torch.nn.init.xavier_uniform_(param)
#    return model
#
##model = random_initialization(model)
#
#logger.info(f"Original DistilBERT model: {model}")
#
#raw_data = load_dataset("Salesforce/wikitext", 
#                        "wikitext-103-raw-v1",
#                        trust_remote_code=True, 
#                        num_proc=16, 
#                        cache_dir="data/wikitext-103-raw-v1")
#
#
#test_data = raw_data["test"]
#test_data = raw_data.filter(lambda x: len(x["text"]) > 0)
#
#train_data = raw_data["train"]
#train_data = raw_data.filter(lambda x: len(x["text"]) > 0)
#
#val_data = raw_data["validation"]
#val_data = raw_data.filter(lambda x: len(x["text"]) > 0)
#
#
#def preprocess_function(data):
#    return tokenizer(["".join(x) for x in data["text"]]) 
#
#def group_texts(examples):
#    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
#    total_length = len(concatenated_examples[list(examples.keys())[0]])
#    padding_length = 0
#    remainder = total_length % model_args.max_pos
#    if remainder != 0:
#        padding_length = model_args.max_pos - remainder
#        for k, t in concatenated_examples.items():
#            concatenated_examples[k] = t + [tokenizer.pad_token_id] * padding_length
#        total_length += padding_length
#        concatenated_examples["attention_mask"][-padding_length:] = [0] * padding_length
#    result = {
#        k: [t[i : i + model_args.max_pos] for i in range(0, total_length, model_args.max_pos)]
#        for k, t in concatenated_examples.items()
#    }
#    return result
#
#train_data = train_data.map(preprocess_function, batched=True, num_proc=16, remove_columns=["text"])
#train_data = train_data.map(group_texts, batched=True, num_proc=16)
#
#val_data = val_data.map(preprocess_function, batched=True, num_proc=16, remove_columns=["text"])
#val_data = val_data.map(group_texts, batched=True, num_proc=16)
#
#test_data = test_data.map(preprocess_function, batched=True, num_proc=16, remove_columns=["text"])
#test_data = test_data.map(group_texts, batched=True, num_proc=16)
#
#datacollator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
#
#train_data.set_format(type='torch', columns=['input_ids', 'attention_mask'])
#val_data.set_format(type='torch', columns=['input_ids', 'attention_mask'])
#test_data.set_format(type='torch', columns=['input_ids', 'attention_mask'])
#
#lr_scheduler_args = {"lr_end": 1e-6, "power": 3.0}
#
#
#training_args = TrainingArguments(
#    output_dir=model_path,
#    overwrite_output_dir=True,
#    num_train_epochs=1,
#    per_device_train_batch_size=4,
#    per_device_eval_batch_size=2,
#    gradient_accumulation_steps=32,
#    evaluation_strategy="steps",
#    save_strategy="steps",
#    logging_strategy="steps",
#    logging_dir=model_path,
#    do_train=True,
#    do_eval=True,
#    load_best_model_at_end=True,
#    metric_for_best_model="eval_loss",
#    greater_is_better=False,
#    report_to="wandb",
#    run_name="long_distilbert_wikitext103",
#    logging_steps=1,
#    save_steps=500,
#    eval_steps=500,
#    save_total_limit=3,
#    dataloader_num_workers=16,
#    warmup_steps=5,
#    learning_rate=3e-5,
#    weight_decay=0.01,
#    adam_epsilon=1e-6,
#    lr_scheduler_type="polynomial",
#    lr_scheduler_kwargs=lr_scheduler_args,
#    #max_grad_norm=5.0,
#    fp16=True,
#    fp16_opt_level="O1",
#    )
#
#trainer = Trainer(
#    model=model,
#    args=training_args,
#    data_collator=datacollator,
#    train_dataset = train_data["train"],
#    eval_dataset = val_data["validation"],
#)
#
#trainer.evaluate()  

