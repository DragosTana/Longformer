import unittest
import torch
import torch.nn as nn

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from model.encoders import Encoder
from model.decoders import Decoder
from model.transformer import Transformer
from model.config import TransformerConfig

class TestEncoder(unittest.TestCase):
    
    def test_batch_forward(self):
        config = TransformerConfig(num_hidden_layers=3)
        encoder = Encoder(config)
        
        batch_size = 32
        sequence_length = 128
        input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_length))
        output = encoder(input_ids)
        
        self.assertEqual(output.shape, (batch_size, sequence_length, config.model_dim))
        
class TestDecoder(unittest.TestCase):
    
    def test_batch_forward(self):
        config = TransformerConfig(num_hidden_layers=3)
        decoder = Decoder(config)
        
        batch_size = 32
        sequence_length = 128
        input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_length))
        encoder_hidden_states = torch.randn(batch_size, sequence_length, config.model_dim)
        output = decoder(input_ids, encoder_hidden_states)
        
        self.assertEqual(output.shape, (batch_size, sequence_length, config.model_dim))
        
        
class TestTransformer(unittest.TestCase):
        
        def test_batch_forward(self):
            config = TransformerConfig(num_hidden_layers=3)
            transformer = Transformer(config)
            
            batch_size = 32
            sequence_length = 128
            input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_length))
            target_ids = torch.randint(0, config.trg_vocab_size, (batch_size, sequence_length))
            output = transformer(input_ids, target_ids)
            
            self.assertEqual(output.shape, (batch_size, sequence_length, config.trg_vocab_size))
        
if __name__=="__main__":
    unittest.main()