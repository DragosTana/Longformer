import unittest
import torch
import torch.nn as nn

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from model.config import Config

class TestConfigFile(unittest.TestCase):
    def test_transformer_config(self):
        config = Config()
        self.assertEqual(config.vocab_size, 30522)
        self.assertEqual(config.model_dim, 768)
        self.assertEqual(config.num_hidden_layers, 12)
        self.assertEqual(config.num_attention_heads, 12)
        self.assertEqual(config.ffn_dim, 3072)
        self.assertEqual(config.attention_probs_dropout_prob, 0.1)
        self.assertEqual(config.hidden_dropout_prob, 0.1)
        self.assertEqual(config.max_position_embeddings, 512)
        self.assertEqual(config.layer_norm_eps, 1e-12)
        self.assertEqual(config.pad_token_id, 0)
        
    def test_types(self):
        config = Config()
        self.assertIsInstance(config.vocab_size, int)
        self.assertIsInstance(config.model_dim, int)
        self.assertIsInstance(config.num_hidden_layers, int)
        self.assertIsInstance(config.num_attention_heads, int)
        self.assertIsInstance(config.ffn_dim, int)
        self.assertIsInstance(config.attention_probs_dropout_prob, float)
        self.assertIsInstance(config.hidden_dropout_prob, float)
        self.assertIsInstance(config.max_position_embeddings, int)
        self.assertIsInstance(config.layer_norm_eps, float)
        self.assertIsInstance(config.pad_token_id, int)
        
if __name__ == '__main__':
    unittest.main()