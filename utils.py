from transformers.models.roberta.modeling_roberta import RobertaEmbeddings, RobertaConfig
import torch
config = RobertaConfig(
    vocab_size=50265,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
    dropout=0.0,
    max_position_embeddings=514,
)

embeddings = RobertaEmbeddings(config)
batch_size = 2
seq_length = 512
inputs_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
attention_mask = torch.ones_like(inputs_ids)
attention_mask[0, -100:] = 0
output = embeddings(inputs_ids)
print(output.size())

