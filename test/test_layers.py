import unittest
import torch
import torch.nn as nn

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from model.layers import PositionWiseFeedForward, MultiHeadAttention, EncoderLayer, DecoderLayer, PositionalEncoding

class TestPositionWiseFeedForward(unittest.TestCase):
    def test_output_size(self):
        class Config:
            model_dim = 512
            ffn_dim = 2048
            hidden_dropout_prob = 0.1

        model = PositionWiseFeedForward(Config)

        input_tensor = torch.randn(32, 128, 512)
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.size(), input_tensor.size())
        
class TestMultiHeadAttention(unittest.TestCase):
    def test_output_size(self):
        class Config:
            model_dim = 512
            num_attention_heads = 8
            attention_probs_dropout_prob = 0.1

        model = MultiHeadAttention(Config)

        input_query = torch.randn(32, 128, 512)
        input_key = torch.randn(32, 128, 512)
        input_value = torch.randn(32, 128, 512)
        output_tensor = model(input_query, input_key, input_value)
        self.assertEqual(output_tensor.size(), input_query.size())
        self.assertEqual(output_tensor.size(), input_key.size())
        self.assertEqual(output_tensor.size(), input_value.size())
        
    def test_hidden_size_not_multiple_of_num_attention_heads(self):
        class Config:
            model_dim = 512
            num_attention_heads = 7
            attention_probs_dropout_prob = 0.1

        with self.assertRaises(ValueError):
            MultiHeadAttention(Config)
                  
class TestEncoder(unittest.TestCase):
    def test_output_size(self):
        class Config:
            model_dim = 512
            num_attention_heads = 8
            attention_probs_dropout_prob = 0.1
            ffn_dim = 2048
            hidden_dropout_prob = 0.1
            layer_norm_eps = 1e-5
            
        model = EncoderLayer(Config)

        input_tensor = torch.randn(32, 128, 512)
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.size(), input_tensor.size())
    
class TestDecoder(unittest.TestCase):
    def test_output_size(self):
        class Config:
            model_dim = 512
            num_attention_heads = 8
            attention_probs_dropout_prob = 0.1
            ffn_dim = 2048
            hidden_dropout_prob = 0.1
            

        model = DecoderLayer(Config)

        hidden_states = torch.randn(32, 128, 512)
        encoder_hidden_states = torch.randn(32, 128, 512)
        output_tensor = model(hidden_states, encoder_hidden_states)
        self.assertEqual(output_tensor.size(), hidden_states.size())
        
class TestPositionalEmbedding(unittest.TestCase):
    def test_output_size(self):
        class Config:
            max_position_embeddings = 256
            model_dim = 512

        model = PositionalEncoding(Config)

        input_tensor = torch.randn(32, 128, 512)
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.size(), input_tensor.size())
        

if __name__ == '__main__':
    unittest.main()