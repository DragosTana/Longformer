import unittest
import torch
import torch.nn as nn

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from model.config import TransformerConfig
from model.longformer import LongformerEmbeddings, LongformerLMHead, LongformerSelfAttention

class TestEncoder(unittest.TestCase):

    @torch.no_grad()
    def test_batch_forward(self):
        config = TransformerConfig()
        encoder = Encoder(config)
        
        batch_size = 32
        sequence_length = 128
        input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_length))
        output = encoder(input_ids)
        
        self.assertEqual(output.shape, (batch_size, sequence_length, config.model_dim))
        
class TestDecoder(unittest.TestCase):
    
    @torch.no_grad()
    def test_batch_forward(self):
        config = TransformerConfig()
        decoder = Decoder(config)
        
        batch_size = 32
        sequence_length = 128
        input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_length))
        encoder_hidden_states = torch.randn(batch_size, sequence_length, config.model_dim)
        output = decoder(input_ids, encoder_hidden_states)
        
        self.assertEqual(output.shape, (batch_size, sequence_length, config.model_dim))
        
        
class TestTransformer(unittest.TestCase):
        
        @torch.no_grad()
        @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
        def test_batch_forward(self):
            config = TransformerConfig(num_hidden_layers=2)
            transformer = Transformer(config)
            
            batch_size = 32
            sequence_length = 128
            input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_length))
            target_ids = torch.randint(0, config.trg_vocab_size, (batch_size, sequence_length))
            output = transformer(input_ids, target_ids)
            
            self.assertEqual(output.shape, (batch_size, sequence_length, config.trg_vocab_size))
        
        @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
        def test_all_parameter_updates(self):
            config = TransformerConfig(num_hidden_layers=2)
            transformer = Transformer(config)
            optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)
            
            batch_size = 32
            sequence_length = 128
            input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_length))
            target_ids = torch.randint(0, config.trg_vocab_size, (batch_size, sequence_length))
            output = transformer(input_ids, target_ids)
            
            loss = output.mean()
            loss.backward()
            optimizer.step()
            
            for param in transformer.parameters():
                self.assertIsNotNone(param.grad)
              



