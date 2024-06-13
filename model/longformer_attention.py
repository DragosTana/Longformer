from torch import nn
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

class LongformerSelfAttention(nn.Module):
    def __init__(self, config, layer_id: int):
        super(LongformerSelfAttention, self).__init__()
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
    

