import unittest
import torch
import os
import sys

# ugly hack to allow imports from parallel directories
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/models'))
if path not in sys.path:
    sys.path.insert(0, path)

from roberta import RobertaEmbeddings as MyRobertaEmbeddings
from roberta import RobertaAttention as MyRobertaAttention
from roberta import RobertaLayer as MyRobertaLayer
from roberta import RobertaEncoder as MyRobertaEncoder
from roberta import RobertaModel as MyRobertaModel
from roberta import RobertaForMaskedLM as MyRobertaForMaskedLM
from roberta import RobertaForSequenceClassification as MyRobertaForSequenceClassification
from transformers import RobertaConfig, RobertaModel, RobertaForMaskedLM, RobertaForSequenceClassification

class TestRobertaComponents(unittest.TestCase):
    
    def setUp(self):
        self.config = RobertaConfig(
            vocab_size=50265,
            max_position_embeddings=514,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            dropout=0.0,
        )

    def test_embeddings(self):
        model = RobertaModel(self.config)
        embeddings = model.embeddings
        my_embeddings = MyRobertaEmbeddings(self.config)
        my_embeddings.load_state_dict(embeddings.state_dict())

        batch_size = 1
        seq_length = self.config.max_position_embeddings - 2
        inputs_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(inputs_ids)
        attention_mask[0, -100:] = 0

        output = embeddings(inputs_ids)
        my_output = my_embeddings(inputs_ids)

        self.assertTrue(torch.allclose(output, my_output, atol=1e-4))

    def test_attention(self):
        model = RobertaModel(self.config)
        attention = model.encoder.layer[0].attention
        my_attention = MyRobertaAttention(self.config)
        my_attention.load_state_dict(attention.state_dict())

        batch_size = 4
        seq_length = 512
        hidden_size = 768
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        attention_mask = torch.ones(batch_size, seq_length)
        attention_mask[0, -100:] = 0
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min

        output = attention(hidden_states, attention_mask, output_attentions=True)
        my_output = my_attention(hidden_states, attention_mask, output_attentions=True)

        self.assertTrue(torch.allclose(output[0], my_output[0], atol=1e-5))
        self.assertTrue(torch.allclose(output[1], my_output[1], atol=1e-5))

    def test_layer(self):
        model = RobertaModel(self.config)
        layer = model.encoder.layer[0]
        my_layer = MyRobertaLayer(self.config)
        my_layer.load_state_dict(layer.state_dict())

        batch_size = 4
        seq_length = self.config.max_position_embeddings - 2
        dim = self.config.hidden_size
        hidden_states = torch.randn(batch_size, seq_length, dim)

        output = layer(hidden_states, output_attentions=True)
        my_output = my_layer(hidden_states, output_attentions=True)

        self.assertTrue(torch.allclose(output[0], my_output[0], atol=1e-4))
        self.assertTrue(torch.allclose(output[1][0], my_output[1][0], atol=1e-4))

    def test_encoder(self):
        model = RobertaModel(self.config)
        encoder = model.encoder
        my_encoder = MyRobertaEncoder(self.config)
        my_encoder.load_state_dict(encoder.state_dict())

        batch_size = 4
        seq_length = 512
        hidden_size = 768
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        attention_mask = torch.ones(batch_size, seq_length)
        attention_mask[0, -100:] = 0
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min

        output = encoder(hidden_states, attention_mask, output_attentions=True, output_hidden_states=True)
        my_output = my_encoder(hidden_states, attention_mask, output_attentions=True, output_hidden_states=True)

        self.assertTrue(torch.allclose(output[0], my_output[0], atol=1e-5))
        self.assertTrue(all(torch.allclose(output[1][i], my_output[1][i], atol=1e-5) for i in range(len(output[1]))))
        self.assertTrue(all(torch.allclose(output[2][i], my_output[2][i], atol=1e-5) for i in range(len(output[2]))))

    def test_model(self):
        model = RobertaModel(self.config)
        my_model = MyRobertaModel(self.config)
        my_model.load_state_dict(model.state_dict())

        batch_size = 2
        seq_length = self.config.max_position_embeddings - 2
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones(input_ids.shape)
        attention_mask[:, -10:] = 0

        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True, output_attentions=True)
        my_outputs = my_model(input_ids, attention_mask=attention_mask, output_hidden_states=True, output_attentions=True)

        self.assertTrue(torch.allclose(outputs[0], my_outputs[0], atol=1e-5))
        self.assertTrue(all(torch.allclose(outputs[1][i], my_outputs[1][i], atol=1e-5) for i in range(len(outputs[1]))))
        self.assertTrue(all(torch.allclose(outputs[2][i], my_outputs[2][i], atol=1e-5) for i in range(len(outputs[2]))))

    def test_roberta_for_mlm(self):
        model = RobertaForMaskedLM(self.config)
        my_model = MyRobertaForMaskedLM(self.config)
        my_model.load_state_dict(model.state_dict())

        input_ids = torch.randint(0, 50265, (4, 512))
        attention_mask = torch.ones(input_ids.shape)
        attention_mask[0, -100:] = 0

        output = model(input_ids, attention_mask)
        my_output = my_model(input_ids, attention_mask)

        self.assertTrue(torch.allclose(output[0], my_output, atol=1e-5))


def test_backprop():
    config = RobertaConfig(
        vocab_size=50265,
        max_position_embeddings=514,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        dropout = 0.0,
    )

    model = RobertaForMaskedLM(config)
    my_model = MyRobertaForMaskedLM(config)
    my_model.load_state_dict(model.state_dict())
    print(model)
    print(my_model)
    
    input_ids = torch.randint(0, 50265, (4, 512))    
    attention_mask = torch.ones(input_ids.shape)
    attention_mask[0, -100:] = 0
    
    output = model(input_ids, attention_mask)
    my_output = my_model(input_ids, attention_mask)
    
    optimizer = torch.optim.Adam(my_model.parameters(), lr=1e-5)
    loss = my_output.sum()
    print("Loss: ", loss)
    loss.backward()
    optimizer.step()
    assert any(param.grad is not None for param in my_model.parameters())
    
    optimizer_original = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_original = output[0].sum()
    print("Loss Original: ", loss_original)
    loss_original.backward()
    optimizer_original.step()
    
    print(torch.allclose(output[0], my_output, atol=1e-4))
    print(torch.allclose(loss, loss_original, atol=1e-4))
    
    output = model(input_ids, attention_mask)
    my_output = my_model(input_ids, attention_mask)
    
    print(torch.allclose(output[0], my_output, atol=1e-2))
    
def test_classifier():
    config = RobertaConfig(
        vocab_size=50265,
        max_position_embeddings=514,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
        hidden_dropout_prob=0.0,
        classifier_dropout=0.0,
        attention_probs_dropout_prob=0.0,
        dropout = 0.0,
    )

    model = RobertaForSequenceClassification(config)
    my_model = MyRobertaForSequenceClassification(config)
    my_model.load_state_dict(model.state_dict())
    print(model)
    print(my_model)
    
    input_ids = torch.randint(0, 50265, (4, 512))
    attention_mask = torch.ones(input_ids.shape)
    attention_mask[0, -100:] = 0
    
    output = model(input_ids, attention_mask)
    my_output = my_model(input_ids, attention_mask)
    
    print(f"Output equal: {torch.allclose(output[0], my_output, atol=1e-5)}")

if __name__ == "__main__":
    unittest.main()
