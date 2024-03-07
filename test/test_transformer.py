import unittest
import torch
import torch.nn as nn

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from model.encoders import Encoder
from model.config import TransformerConfig

class TestEncoder(unittest.TestCase):
    
    def test_batch_forward(self):
        config = TransformerConfig()
        encoder = Encoder(config)
        
        input_ids = torch.randint(0, config.vocab_size, (32, 128))
        
        output = encoder(input_ids)
        
        self.assertEqual(output.size(), (32, 128, config.model_dim))
        
if __name__=="__main__":
    unittest.main()