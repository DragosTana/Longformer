
try:
    from sliding_chunks import  sliding_chunks_matmul_qk, sliding_chunks_matmul_pv, \
                                sliding_chunks_no_overlap_matmul_qk, sliding_chunks_no_overlap_matmul_pv, \
                                mask_invalid_locations, pad_to_window_size
except:
    from .sliding_chunks import  sliding_chunks_matmul_qk, sliding_chunks_matmul_pv, \
                                sliding_chunks_no_overlap_matmul_qk, sliding_chunks_no_overlap_matmul_pv, \
                                mask_invalid_locations, pad_to_window_size
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch
import time

class LongformerSelfAttentionMine(nn.Module):
    def __init__(self, config, layer_id: int):
        super(LongformerSelfAttentionMine, self).__init__()
        if config.dim % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.dim}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.dim // config.num_attention_heads
        self.dim = config.dim
        
        self.query = nn.Linear(self.dim, self.dim)
        self.key = nn.Linear(self.dim, self.dim)
        self.value = nn.Linear(self.dim, self.dim)
        
        self.query_global = nn.Linear(self.dim, self.dim)
        self.key_global = nn.Linear(self.dim, self.dim)
        self.value_global = nn.Linear(self.dim, self.dim)
        
        self.dropout = config.attention_dropout
        
        self.layer_id = layer_id
        self.attention_window = config.attention_window[self.layer_id]
        self.attention_mode = config.attention_mode
        
        assert self.attention_mode in ['sliding_chunks', 'sliding_chunks_no_overlap']
        assert self.attention_window > 0
    
    def forward(self, hidden_states, attention_mask=None):
        
        '''
        Attention mask is a tensor of shape [batch_size, :, :, sequence_length].
        It contains:
            - -ve   no attention (e.g. padding tokens)
            - 0     for local attention
            - +ve   for global attention
        '''
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
            padding_mask = attention_mask < 0  #[batch_size, sequence_length] True for padding tokens
            global_mask = attention_mask > 0   #[batch_size, sequence_length] True for global attention locations
            removed_from_windowed_attention_mask = attention_mask != 0  #[batch_size, sequence_length] True for tokens that are not part of the windowed attention
            
            num_global_tokens = global_mask.sum(dim=1) #[batch_size] number of global tokens in each sequence
            max_num_global_tokens = num_global_tokens.max().item() # max number of global tokens in the batch
            if max_num_global_tokens <= 0:
                global_mask = None
            else:
                # necessary for global attention when num_global_tokens is not the same for all sequences (i.e. question-answering)
                global_mask_nonzero = global_mask.nonzero(as_tuple=True) # tuple of two tensors, containing indices of non-zero(True) elements
                zero_to_max_range = torch.arange(0, max_num_global_tokens, device=global_mask.device) 
                selection_padding_mask = zero_to_max_range < num_global_tokens.unsqueeze(dim=-1) # [batch_size, max_num_global_tokens] 
                selection_padding_mask_nonzero = selection_padding_mask.nonzero(as_tuple=True) # tuple of two tensors, containing indices of non-zero (True) elements
                selection_padding_mask_zeros = (selection_padding_mask == 0).nonzero(as_tuple=True) # tuple of two tensors, containing indices of zero (False) elements
        
        else:
            removed_from_windowed_attention_mask = None
            global_mask = None
            padding_mask = None
            
        batch_size, sequence_length, embed_dim = hidden_states.size()
        assert embed_dim == self.dim
        q = self.query(hidden_states) # [sequence_length, batch_size, dim]
        k = self.key(hidden_states) # [sequence_length, batch_size, dim]
        v = self.value(hidden_states)   # [sequence_length, batch_size, dim]
        q /= math.sqrt(self.attention_head_size)
        
        q = q.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size) # [batch_size, sequence_length, num_attention_heads, attention_head_size]
        k = k.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size) # [batch_size, sequence_length, num_attention_heads, attention_head_size]
        v = v.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size) # [batch_size, sequence_length, num_attention_heads, attention_head_size]

        # calculate local attention weights
        if self.attention_mode == 'sliding_chunks':
            attention_weights = sliding_chunks_matmul_qk(q, k, self.attention_window, padding_value=0) # [batch_size, sequence_lenght, num_attention_heads, 2w+1]
        elif self.attention_mode == 'sliding_chunks_no_overlap':
            attention_weights = sliding_chunks_no_overlap_matmul_qk(q, k, self.attention_window, padding_value=0)
        else:
            raise ValueError(f"Unrecognized attention mode {self.attention_mode}")
        
        # mask out the tokens that are not part of the windowed attention from the attention weights (global tokens and padding tokens)
        if removed_from_windowed_attention_mask is not None:
            removed_from_windowed_attention_mask = removed_from_windowed_attention_mask.unsqueeze(dim=-1).unsqueeze(dim=-1) # [batch_size, sequence_length, 1, 1]
            float_mask = removed_from_windowed_attention_mask.type_as(q).masked_fill(removed_from_windowed_attention_mask, float('-inf')) 
            float_mask = float_mask.repeat(1, 1, 1, 1)
            ones = torch.ones_like(float_mask)  # [batch_size, sequence_length, 1, 1]
            if self.attention_mode == 'sliding_chunks':
                d_mask = sliding_chunks_matmul_qk(ones, float_mask, self.attention_window, padding_value=0) 
            elif self.attention_mode == 'sliding_chunks_no_overlap':
                d_mask = sliding_chunks_no_overlap_matmul_qk(ones, float_mask, self.attention_window, padding_value=0)
            
            attention_weights += d_mask 
            
        assert attention_weights.size() == (batch_size, sequence_length, self.num_attention_heads, 2*self.attention_window + 1)

        if global_mask is not None:
            selected_k = k.new_zeros(batch_size, max_num_global_tokens, self.num_attention_heads, self.attention_head_size)
            selected_k[selection_padding_mask_nonzero] = k[global_mask_nonzero] # from k, select the global tokens

            selected_attn_weights = torch.einsum('blhd,bshd->blhs', q, selected_k) # [batch_size, sequence_length, num_attention_heads, max_num_global_tokens]
            selected_attn_weights[selection_padding_mask_zeros[0], :, :, selection_padding_mask_zeros[1]] = float('-inf')
            attention_weights = torch.cat((selected_attn_weights, attention_weights), dim=-1) # [batch_size, sequence_length, num_attention_heads, sequence_length + max_num_global_tokens]
        attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32)
        if padding_mask is not None:
            attention_weights = torch.masked_fill(attention_weights, padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
            
        attention_probs = F.dropout(attention_weights, p=self.dropout, training=self.training)

        attention = 0
        if global_mask is not None:
            selected_attention_probs = attention_probs.narrow(-1, 0, max_num_global_tokens) # [batch_size, sequence_length, num_attention_heads, max_num_global_tokens]
            selected_v = v.new_zeros(batch_size, max_num_global_tokens, self.num_attention_heads, self.attention_head_size)
            selected_v[selection_padding_mask_nonzero] = v[global_mask_nonzero]
            attention = torch.einsum('blhs,bshd->blhd', selected_attention_probs, selected_v) # [batch_size, sequence_length, num_attention_heads, attention_head_size]
            attention_probs = attention_probs.narrow(-1, max_num_global_tokens, attention_probs.size(-1) - max_num_global_tokens).contiguous() # [batch_size, sequence_length, num_attention_heads, sequence_length]
    
        if self.attention_mode == 'sliding_chunks':
            attention += sliding_chunks_matmul_pv(attention_probs, v, self.attention_window)
        elif self.attention_mode == 'sliding_chunks_no_overlap':
            attention += sliding_chunks_no_overlap_matmul_pv(attention_probs, v, self.attention_window)
        else:
            raise ValueError(f"Unrecognized attention mode {self.attention_mode}")
        
        attention = attention.type_as(hidden_states).transpose(0, 1).reshape(sequence_length, batch_size, embed_dim).contiguous()

        if global_mask is not None:
            selected_hidden_states = hidden_states.new_zeros(max_num_global_tokens, batch_size, embed_dim)
            hidden_states = hidden_states.transpose(0, 1) 
            selected_hidden_states[selection_padding_mask_nonzero[::-1]] = hidden_states[global_mask_nonzero[::-1]] # select the global tokens
            
            q = self.query_global(selected_hidden_states)   # [max_num_global_tokens, batch_size, dim]
            k = self.key_global(hidden_states)     # [max_num_global_tokens, batch_size, dim] 
            v = self.value_global(hidden_states)   # [max_num_global_tokens, batch_size, dim]

            q = q.contiguous().view(max_num_global_tokens, batch_size*self.num_attention_heads, self.attention_head_size).transpose(0, 1) # [batch_size*num_attention_heads, max_num_global_tokens, attention_head_size]
            k = k.contiguous().view(-1, batch_size*self.num_attention_heads, self.attention_head_size).transpose(0, 1) # [batch_size*num_attention_heads, sequence_length, attention_head_size]
            v = v.contiguous().view(-1, batch_size*self.num_attention_heads, self.attention_head_size).transpose(0, 1) # [batch_size*num_attention_heads, sequence_length, attention_head_size]
            attention_weights = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.attention_head_size) 
        
            attention_weights = attention_weights.view(batch_size, self.num_attention_heads, max_num_global_tokens, sequence_length) 
            attention_weights[selection_padding_mask_zeros[0], :, selection_padding_mask_zeros[1], :] = float('-inf')
            if padding_mask is not None:
                attention_weights = attention_weights.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attention_weights = attention_weights.view(batch_size*self.num_attention_heads, max_num_global_tokens, sequence_length)
            attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32)
            attention_probs = F.dropout(attention_weights, p=self.dropout, training=self.training)
            selected_attention = torch.bmm(attention_probs, v) # [batch_size*num_attention_heads, max_num_global_tokens, attention_head_size]
            assert selected_attention.size() == (batch_size*self.num_attention_heads, max_num_global_tokens, self.attention_head_size)
            
            selected_attention = selected_attention.view(batch_size, self.num_attention_heads, max_num_global_tokens, self.attention_head_size)
            nonzero_selected_attention = selected_attention[selection_padding_mask_nonzero[0], :, selection_padding_mask_nonzero[1]]
            attention[global_mask_nonzero[::-1]] = nonzero_selected_attention.view(len(selection_padding_mask_nonzero[0]), -1).type_as(hidden_states)
            
        return attention.transpose(0, 1).contiguous()
    

class LongformerSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        super(LongformerSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)

        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        self.attention_window = config.attention_window[self.layer_id]
        self.attention_dilation = config.attention_dilation[self.layer_id]
        self.attention_mode = config.attention_mode
        self.autoregressive = config.autoregressive
        assert self.attention_window > 0
        assert self.attention_dilation > 0
        assert self.attention_mode in ['tvm', 'sliding_chunks', 'sliding_chunks_no_overlap']
        if self.attention_mode in ['sliding_chunks', 'sliding_chunks_no_overlap']:
            assert not self.autoregressive  # not supported
            assert self.attention_dilation == 1  # dilation is not supported

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        '''
        The `attention_mask` is changed in `BertModel.forward` from 0, 1, 2 to
            -ve: no attention
              0: local attention
            +ve: global attention
        '''
        assert encoder_hidden_states is None, "`encoder_hidden_states` is not supported and should be None"
        assert encoder_attention_mask is None, "`encoder_attention_mask` is not supported and shiould be None"

        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
            key_padding_mask = attention_mask < 0
            extra_attention_mask = attention_mask > 0
            remove_from_windowed_attention_mask = attention_mask != 0

            num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
            max_num_extra_indices_per_batch = num_extra_indices_per_batch.max()
            if max_num_extra_indices_per_batch <= 0:
                extra_attention_mask = None
            else:
                # To support the case of variable number of global attention in the rows of a batch,
                # we use the following three selection masks to select global attention embeddings
                # in a 3d tensor and pad it to `max_num_extra_indices_per_batch`
                # 1) selecting embeddings that correspond to global attention
                extra_attention_mask_nonzeros = extra_attention_mask.nonzero(as_tuple=True)
                zero_to_max_range = torch.arange(0, max_num_extra_indices_per_batch,
                                                 device=num_extra_indices_per_batch.device)
                # mask indicating which values are actually going to be padding
                selection_padding_mask = zero_to_max_range < num_extra_indices_per_batch.unsqueeze(dim=-1)
                # 2) location of the non-padding values in the selected global attention
                selection_padding_mask_nonzeros = selection_padding_mask.nonzero(as_tuple=True)
                # 3) location of the padding values in the selected global attention
                selection_padding_mask_zeros = (selection_padding_mask == 0).nonzero(as_tuple=True)
        else:
            remove_from_windowed_attention_mask = None
            extra_attention_mask = None
            key_padding_mask = None

        hidden_states = hidden_states.transpose(0, 1)
        seq_len, bsz, embed_dim = hidden_states.size()
        assert embed_dim == self.embed_dim
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        q /= math.sqrt(self.head_dim)

        q = q.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        # attn_weights = (bsz, seq_len, num_heads, window*2+1)
        if self.attention_mode == 'tvm':
            q = q.float().contiguous()
            k = k.float().contiguous()
            attn_weights = diagonaled_mm_tvm(q, k, self.attention_window, self.attention_dilation, False, 0, False)
        elif self.attention_mode == "sliding_chunks":
            attn_weights = sliding_chunks_matmul_qk(q, k, self.attention_window, padding_value=0)
        elif self.attention_mode == "sliding_chunks_no_overlap":
            attn_weights = sliding_chunks_no_overlap_matmul_qk(q, k, self.attention_window, padding_value=0)
        else:
            raise False
        mask_invalid_locations(attn_weights, self.attention_window, self.attention_dilation)
        if remove_from_windowed_attention_mask is not None:
            # This implementation is fast and takes very little memory because num_heads x hidden_size = 1
            # from (bsz x seq_len) to (bsz x seq_len x num_heads x hidden_size)
            remove_from_windowed_attention_mask = remove_from_windowed_attention_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
            # cast to float/half then replace 1's with -inf
            float_mask = remove_from_windowed_attention_mask.type_as(q).masked_fill(remove_from_windowed_attention_mask, -10000.0)
            repeat_size = 1 if isinstance(self.attention_dilation, int) else len(self.attention_dilation)
            float_mask = float_mask.repeat(1, 1, repeat_size, 1)
            ones = float_mask.new_ones(size=float_mask.size())  # tensor of ones
            # diagonal mask with zeros everywhere and -inf inplace of padding
            if self.attention_mode == 'tvm':
                d_mask = diagonaled_mm_tvm(ones, float_mask, self.attention_window, self.attention_dilation, False, 0, False)
            elif self.attention_mode == "sliding_chunks":
                d_mask = sliding_chunks_matmul_qk(ones, float_mask, self.attention_window, padding_value=0)
            elif self.attention_mode == "sliding_chunks_no_overlap":
                d_mask = sliding_chunks_no_overlap_matmul_qk(ones, float_mask, self.attention_window, padding_value=0)

            attn_weights += d_mask
        assert list(attn_weights.size())[:3] == [bsz, seq_len, self.num_heads]
        assert attn_weights.size(dim=3) in [self.attention_window * 2 + 1, self.attention_window * 3]

        # the extra attention
        if extra_attention_mask is not None:
            selected_k = k.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_k[selection_padding_mask_nonzeros] = k[extra_attention_mask_nonzeros]
            # (bsz, seq_len, num_heads, max_num_extra_indices_per_batch)
            selected_attn_weights = torch.einsum('blhd,bshd->blhs', (q, selected_k))
            selected_attn_weights[selection_padding_mask_zeros[0], :, :, selection_padding_mask_zeros[1]] = -10000
            # concat to attn_weights
            # (bsz, seq_len, num_heads, extra attention count + 2*window+1)
            attn_weights = torch.cat((selected_attn_weights, attn_weights), dim=-1)
        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
        if key_padding_mask is not None:
            # softmax sometimes inserts NaN if all positions are masked, replace them with 0
            attn_weights_float = torch.masked_fill(attn_weights_float, key_padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
        v = v.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        attn = 0
        if extra_attention_mask is not None:
            selected_attn_probs = attn_probs.narrow(-1, 0, max_num_extra_indices_per_batch)
            selected_v = v.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_v[selection_padding_mask_nonzeros] = v[extra_attention_mask_nonzeros]
            # use `matmul` because `einsum` crashes sometimes with fp16
            # attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
            attn = torch.matmul(selected_attn_probs.transpose(1, 2), selected_v.transpose(1, 2).type_as(selected_attn_probs)).transpose(1, 2)
            attn_probs = attn_probs.narrow(-1, max_num_extra_indices_per_batch, attn_probs.size(-1) - max_num_extra_indices_per_batch).contiguous()

        if self.attention_mode == 'tvm':
            v = v.float().contiguous()
            attn += diagonaled_mm_tvm(attn_probs, v, self.attention_window, self.attention_dilation, True, 0, False)
        elif self.attention_mode == "sliding_chunks":
            attn += sliding_chunks_matmul_pv(attn_probs, v, self.attention_window)
        elif self.attention_mode == "sliding_chunks_no_overlap":
            attn += sliding_chunks_no_overlap_matmul_pv(attn_probs, v, self.attention_window)
        else:
            raise False

        attn = attn.type_as(hidden_states)
        assert list(attn.size()) == [bsz, seq_len, self.num_heads, self.head_dim]
        attn = attn.transpose(0, 1).reshape(seq_len, bsz, embed_dim).contiguous()

        # For this case, we'll just recompute the attention for these indices
        # and overwrite the attn tensor. TODO: remove the redundant computation
        if extra_attention_mask is not None:
            selected_hidden_states = hidden_states.new_zeros(max_num_extra_indices_per_batch, bsz, embed_dim)
            selected_hidden_states[selection_padding_mask_nonzeros[::-1]] = hidden_states[extra_attention_mask_nonzeros[::-1]]

            q = self.query_global(selected_hidden_states)
            k = self.key_global(hidden_states)
            v = self.value_global(hidden_states)
            q /= math.sqrt(self.head_dim)

            q = q.contiguous().view(max_num_extra_indices_per_batch, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # (bsz*self.num_heads, max_num_extra_indices_per_batch, head_dim)
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # bsz * self.num_heads, seq_len, head_dim)
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # bsz * self.num_heads, seq_len, head_dim)
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            assert list(attn_weights.size()) == [bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len]

            attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
            attn_weights[selection_padding_mask_zeros[0], :, selection_padding_mask_zeros[1], :] = -10000.0
            if key_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    -10000.0,
                )
            attn_weights = attn_weights.view(bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len)
            attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
            attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
            selected_attn = torch.bmm(attn_probs, v)
            assert list(selected_attn.size()) == [bsz * self.num_heads, max_num_extra_indices_per_batch, self.head_dim]

            selected_attn_4d = selected_attn.view(bsz, self.num_heads, max_num_extra_indices_per_batch, self.head_dim)
            nonzero_selected_attn = selected_attn_4d[selection_padding_mask_nonzeros[0], :, selection_padding_mask_nonzeros[1]]
            attn[extra_attention_mask_nonzeros[::-1]] = nonzero_selected_attn.view(len(selection_padding_mask_nonzeros[0]), -1).type_as(hidden_states)

        context_layer = attn.transpose(0, 1)
        if output_attentions:
            if extra_attention_mask is not None:
                # With global attention, return global attention probabilities only
                # batch_size x num_heads x max_num_global_attention_tokens x sequence_length
                # which is the attention weights from tokens with global attention to all tokens
                # It doesn't not return local attention
                # In case of variable number of global attantion in the rows of a batch,
                # attn_weights are padded with -10000.0 attention scores
                attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
            else:
                # without global attention, return local attention probabilities
                # batch_size x num_heads x sequence_length x window_size
                # which is the attention weights of every token attending to its neighbours
                attn_weights = attn_weights.permute(0, 2, 1, 3)
        outputs = (context_layer, attn_weights) if output_attentions else (context_layer,)
        return outputs  

def boh():
    import torch
    import random
    import numpy as np
    from torch import nn
    
    def reproducibility(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    def generate_attention_mask(attention_mask):
        no_attention_mask = (attention_mask == 0).long() * -10000  # Padding tokens (-10000 to mask out)
        local_attention_mask = (attention_mask == 1).long() * 0    # Local attention tokens (0 to keep them)
        global_attention_mask = (attention_mask == 2).long() * 10000 # Global attention tokens (10000 to enhance them)
        converted_attention_mask = no_attention_mask + local_attention_mask + global_attention_mask
        converted_attention_mask = converted_attention_mask.unsqueeze(1).unsqueeze(2)
        return converted_attention_mask
     
    def generate_attention_mask_2(attention_mask):
        dtype = torch.float32
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.to(dtype=dtype)
        attention_mask = (1.0 - attention_mask) * torch.finfo(dtype).min
        return attention_mask
    
    
    reproducibility(420)

    from config import LongformerConfig
    config = LongformerConfig(attention_window=[8]*6, 
                              attention_dilation=[1]*6, 
                              autoregressive=False, 
                              attention_mode='sliding_chunks', 
                              attention_dropout=0.0,
                              dropout=0.0,)
    
    attention_1 = LongformerSelfAttention(config, 0)
    attention_2 = LongformerSelfAttention2(config, 0)
    attention_2.load_state_dict(attention_1.state_dict())
    
    hidden_states = torch.randn(2, 512, 768)
    attention_mask = torch.ones(2, 512, dtype=torch.long)
    attention_mask[:, 0] = 2
    attention_mask[:, -200:] = 0
    
    attention_mask = generate_attention_mask_2(attention_mask)
    
    output_1 = attention_1(hidden_states, attention_mask)
    output_2 = attention_2(hidden_states, attention_mask)
    
    print(torch.allclose(output_1, output_2, atol=1e-4))
    print(output_1)
    print(output_2)
    

class Config:
    def __init__(self, dim, num_attention_heads, attention_dropout, attention_window, attention_mode):
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.attention_window = attention_window
        self.attention_mode = attention_mode
        self.attention_dilation = [1]*6
        self.autoregressive = False
        
        
def measure_execution_time(layer, sequence_lengths, device, num_iterations=10):
    layer.to(device)
    layer.eval()
    
    times = []
    
    for seq_len in sequence_lengths:
        hidden_states = torch.rand(1, seq_len, 768).to(device)
        attention_mask = torch.ones(1, 1, 1, seq_len).to(device)
        
        # Warm-up
        with torch.no_grad():
            _ = layer(hidden_states, attention_mask)
        
        elapsed_times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()  # Ensure all GPU operations are completed
            start_time = time.time()
            with torch.no_grad():
                _ = layer(hidden_states, attention_mask)
            torch.cuda.synchronize()  # Ensure all GPU operations are completed
            end_time = time.time()
            
            elapsed_times.append(end_time - start_time)
        
        avg_time = sum(elapsed_times) / num_iterations
        times.append(avg_time)
    
    return times

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import gc 
    from layers import MultiHeadAttention
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sequence_lengths = [128, 256, 512, 1024, 2048, 4096]
    config = Config(dim=768, num_attention_heads=12, attention_dropout=0.1, attention_window=[64], attention_mode='sliding_chunks')

    layer = MultiHeadAttention(config=config)
    times_original = measure_execution_time(layer, sequence_lengths, device)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    layer = LongformerSelfAttention(config, layer_id=0)
    times_mine = measure_execution_time(layer, sequence_lengths, device)
    
    
    plt.plot(sequence_lengths, times_mine, label='Longformer', color='red', marker='o')
    plt.plot(sequence_lengths, times_original, label='Full Attention', color='blue', marker='o')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.show()
    