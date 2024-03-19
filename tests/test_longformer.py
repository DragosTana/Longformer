import unittest
import torch
import torch.nn as nn

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from model.longformer import    LongformerPooler, LongformerLMHead, LongformerEmbeddings,  \
                                LongformerLayer, LongformerEncoder, LongformerForMaskedLM       

from model.config import TransformerConfig

class TestLongformerPooler(unittest.TestCase):
    
    @torch.no_grad()
    def test_forward(self):
        config = TransformerConfig()
        pooler = LongformerPooler(config)
        input_tensor = torch.randn(32, 128, 768) # batch_size, sequence_length, hidden_size
        output_tensor = pooler(input_tensor)
        self.assertEqual(output_tensor.size(), (32, 768))
        
class TestLongformerLMHead(unittest.TestCase):
    
    @torch.no_grad()
    def test_forward(self):
        config = TransformerConfig(vocab_size=50265)
        lm_head = LongformerLMHead(config)
        input_tensor = torch.randn(32, 128, 768) # batch_size, sequence_length, hidden_size
        output_tensor = lm_head(input_tensor)
        self.assertEqual(output_tensor.size(), (32, 128, 50265)) # batch_size, sequence_length, vocab_size
        
class TestLongformerEmbeddings(unittest.TestCase):
    
    @torch.no_grad()
    def test_forward(self):
        config = TransformerConfig()
        embeddings = LongformerEmbeddings(config)
        input_ids = torch.randint(0, config.vocab_size, (32, 128))
        output_tensor = embeddings(input_ids)
        self.assertEqual(output_tensor.size(), (32, 128, 768))
        
class TestLongformerLayer(unittest.TestCase):
    
    @torch.no_grad()
    def test_forward(self):
        config = TransformerConfig()
        layer = LongformerLayer(config)
        hidden_states = torch.randn(32, 128, 768) # batch_size, sequence_length, hidden_size
        output_tensor = layer(hidden_states)
        self.assertEqual(output_tensor.size(), (32, 128, 768))
        
class TestLongformerEncoder(unittest.TestCase):
    
    @torch.no_grad()
    def test_forward_no_pooling(self):
        config = TransformerConfig(num_hidden_layers=2)
        encoder = LongformerEncoder(config, add_pooling_layer=False)
        
        batch_size = 32
        sequence_length = 128
        input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_length))
        output_tensor = encoder(input_ids)
        self.assertEqual(output_tensor.size(), (batch_size, sequence_length, config.model_dim))
        
    @torch.no_grad()
    def test_forward_with_pooling(self):
        config = TransformerConfig(num_hidden_layers=2)
        encoder = LongformerEncoder(config, add_pooling_layer=True)
        
        batch_size = 32
        sequence_length = 128
        input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_length))
        output_tensor = encoder(input_ids)
        self.assertEqual(output_tensor.size(), (batch_size, config.model_dim))
        
    def test_all_parameter_updates(self):
        
        batch_size = 32
        sequence_length = 128
        
        config = TransformerConfig(num_hidden_layers=2)
        encoder = LongformerEncoder(config, add_pooling_layer=True)
        
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
        
        input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_length))
        output = encoder(input_ids)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        
        for param in encoder.parameters():
            self.assertTrue(param.grad is not None)
            
class TestLongformerForMaskedLM(unittest.TestCase):
    
    @torch.no_grad()
    def test_forward(self):
        config = TransformerConfig(num_hidden_layers=2)
        model = LongformerForMaskedLM(config)
        
        batch_size = 32
        sequence_length = 128
        input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_length))
        output_tensor = model(input_ids)
        self.assertEqual(output_tensor.size(), (batch_size, sequence_length, config.vocab_size))
        
if __name__ == "__main__":
    unittest.main()

