import torch
import torch.nn.functional as F
from functools import lru_cache
from typing import Union

def _get_invalid_locations_mask_fixed_dilation(seq_len: int, w: int, d: int):
    diagonals_list = []
    for j in range(-d * w, d, d):
        diagonal_mask = torch.zeros(seq_len, device='cpu', dtype=torch.uint8)
        diagonal_mask[:-j] = 1
        diagonals_list.append(diagonal_mask)
    return torch.stack(diagonals_list, dim=-1)

@lru_cache()
def _get_invalid_locations_mask(w: int, d: Union[torch.Tensor,int], device: str):
    if isinstance(d, int):
        affected_seq_len = w * d
        mask = _get_invalid_locations_mask_fixed_dilation(affected_seq_len, w, d)
        mask = mask[None, :, None, :]
    else:
        affected_seq_len = w * d.max()
        head_masks = []
        d_list = d.cpu().numpy().tolist()
        for d in d_list:
            one_head_mask = _get_invalid_locations_mask_fixed_dilation(affected_seq_len, w, d)
            head_masks.append(one_head_mask)
        mask = torch.stack(head_masks, dim=-2)
        mask = mask[None, :, :, :]

    return affected_seq_len, mask.bool().to(device)


def mask_invalid_locations(input_tensor: torch.Tensor, w: int, d: Union[torch.Tensor, int]) -> torch.Tensor:
    affected_seq_len, mask = _get_invalid_locations_mask(w, d, input_tensor.device)
    seq_len = input_tensor.size(1)
    beginning_input = input_tensor[:, :affected_seq_len, :, :w+1]
    mask = mask[:, :seq_len].expand(beginning_input.size())
    beginning_input.masked_fill_(mask, -float('inf'))


def _skew(x, direction, padding_value):
    '''Convert diagonals into columns (or columns into diagonals depending on `direction`'''
    x_padded = F.pad(x, direction, value=padding_value)
    x_padded = x_padded.view(*x_padded.size()[:-2], x_padded.size(-1), x_padded.size(-2))
    return x_padded


def _skew2(x, padding_value):
    '''shift every row 1 step to right converting columns into diagonals'''
    # X = B x C x M x L
    B, C, M, L = x.size()
    x = F.pad(x, (0, M + 1), value=padding_value)  # B x C x M x (L+M+1)
    x = x.view(B, C, -1)  # B x C x ML+MM+M
    x = x[:, :, :-M]  # B x C x ML+MM
    x = x.view(B, C, M, M + L)  # B x C, M x L+M
    x = x[:, :, :, :-1]
    return x


def _chunk(x, w):
    '''convert into overlapping chunkings. Chunk size = 2w, overlap size = w'''

    # non-overlapping chunks of size = 2w
    x = x.view(x.size(0), x.size(1) // (w * 2), w * 2, x.size(2))

    # use `as_strided` to make the chunks overlap with an overlap size = w
    chunk_size = list(x.size())
    chunk_size[1] = chunk_size[1] * 2 - 1

    chunk_stride = list(x.stride())
    chunk_stride[1] = chunk_stride[1] // 2
    return x.as_strided(size=chunk_size, stride=chunk_stride)


def banded_gemm_qk(q: torch.Tensor, k: torch.Tensor, w: int, padding_value: float):
    '''Matrix multiplicatio of query x key tensors using with a sliding window attention pattern.
    This implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer)
    with an overlap of size w'''
    bsz, seqlen, num_heads, head_dim = q.size()
    assert seqlen % (w * 2) == 0
    assert q.size() == k.size()

    chunks_count = seqlen // w - 1
    q = q.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)
    k = k.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)
    chunk_q = _chunk(q, w)
    chunk_k = _chunk(k, w)
    chunk_attn = torch.einsum('bcxd,bcyd->bcxy', (chunk_q, chunk_k))  # multiply
    diagonal_chunk_attn = _skew(chunk_attn, direction=(0, 0, 0, 1), padding_value=padding_value)
    diagonal_attn = diagonal_chunk_attn.new_empty((bsz * num_heads, chunks_count + 1, w, w * 2 + 1))

    # copy parts from diagonal_chunk_attn into the compined matrix of attentions
    # - copying the main diagonal and the upper triangle
    diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, :w + 1]
    diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, :w + 1]
    # - copying the lower triangle
    diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[:, :, - (w + 1):-1, w + 1:]
    diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, :w - 1, 1 - w:]

    # separate bsz and num_heads dimensions again
    diagonal_attn = diagonal_attn.view(bsz, num_heads, seqlen, 2 * w + 1).transpose(2, 1)
    
    mask_invalid_locations(diagonal_attn, w, 1) 
    return diagonal_attn


def banded_gemm_pv(prob: torch.Tensor, v: torch.Tensor, w: int):
    '''Same as sliding_chunks_matmul_qk but for prob and value tensors. It is expecting the same output
    format from sliding_chunks_matmul_qk'''
    bsz, seqlen, num_heads, head_dim = v.size()
    assert seqlen % (w * 2) == 0
    assert prob.size()[:3] == v.size()[:3]
    assert prob.size(3) == 2 * w + 1
    chunks_count = seqlen // w - 1
    # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size 2w
    chunk_prob = prob.transpose(1, 2).reshape(bsz * num_heads, seqlen // w, w, 2 * w + 1)

    # group bsz and num_heads dimensions into one
    v = v.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)

    # pad seqlen with w at the beginning of the sequence and another w at the end
    padded_v = F.pad(v, (0, 0, w, w), value=-1)

    # chunk padded_v into chunks of size 3w and an overlap of size w
    chunk_v_size = (bsz * num_heads, chunks_count + 1, 3 * w, head_dim)
    chunk_v_stride = padded_v.stride()
    chunk_v_stride = chunk_v_stride[0], w * chunk_v_stride[1], chunk_v_stride[1], chunk_v_stride[2]
    chunk_v = padded_v.as_strided(size=chunk_v_size, stride=chunk_v_stride)

    skewed_prob = _skew2(chunk_prob, padding_value=0)

    context = torch.einsum('bcwd,bcdh->bcwh', (skewed_prob, chunk_v))
    return context.view(bsz, num_heads, seqlen, head_dim).transpose(1, 2)


def pad_to_window_size(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                       one_sided_window_size: int, pad_token_id: int):
    '''A helper function to pad tokens and mask to work with the sliding_chunks implementation of Longformer selfattention.
    Input:
        input_ids = torch.Tensor(bsz x seqlen): ids of wordpieces
        attention_mask = torch.Tensor(bsz x seqlen): attention mask
        one_sided_window_size = int: window size on one side of each token
        pad_token_id = int: tokenizer.pad_token_id
    Returns
        (input_ids, attention_mask) padded to length divisible by 2 * one_sided_window_size
    '''
    w = int(2 * one_sided_window_size)
    seqlen = input_ids.size(1)
    padding_len = (w - seqlen % w) % w
    input_ids = F.pad(input_ids, (0, padding_len), value=pad_token_id)
    attention_mask = F.pad(attention_mask, (0, padding_len), value=False)  # no attention on the padding tokens
    return input_ids, attention_mask


def indicies_matmul(q, k):
    result = torch.zeros((1, q.size(1), q.size(2), q.size(2)))
    for chuck in range(q.size(1)):
        for i in range(q.size(2)):
            for j in range(q.size(2)):
                ith = q[0, chuck, i].item()
                jth = k[0, chuck, j].item()
                string = str(ith) + str(jth)
                result[0, chuck, i, j] = int(string)
    return result

def visualize_attention():
    w = 2
    chunk_size = 2 * w
    seq_len = 8
    chunks = seq_len // w - 1

    query = torch.arange(1, seq_len + 1, dtype=torch.int)
    key = torch.arange(1, seq_len + 1, dtype=torch.int)
        
    query = query[None, :, None]
    key = key[None, :, None]
    q = _chunk(query, w)
    k = _chunk(key, w)
    result = indicies_matmul(q, k)    
    
    print("Query: ")
    print(query)
    
    print("Query after chunking: ")
    print(q)
    
    print("After matmul: ")
    print(result)
    
    print("padding: ")
    print(F.pad(result, (0, 0, 0, 1), value=0))
    
    skewd = _skew(result, direction=(0, 0, 0, 1), padding_value=0)
    
    print("After skew: ")
    print(skewd)

    #diagonal_attn = skewd.new_empty((1, chunks + 1, w, w * 2 + 1))
    diagonal_attn = torch.zeros((1, chunks + 1, w, w * 2 + 1))
    
    print("------------------------------------------------------")
    diagonal_attn[:, :-1, :, w:] = skewd[:, :, :w, :w + 1]
    print("Selected diagonals: ")
    print(skewd[:, :, :w, :w + 1])
    print("Diagonal attention: ")
    print(diagonal_attn)
    
    print("------------------------------------------------------")
    diagonal_attn[:, -1, :, w:] = skewd[:, -1, w:, :w + 1]
    print("Selected diagonals: ")
    print(skewd[:, -1, w:, :w + 1])
    print("Diagonal attention: ")
    print(diagonal_attn)
    
    print("------------------------------------------------------")
    diagonal_attn[:, 1:, :, :w] = skewd[:, :, - (w + 1):-1, w + 1:]
    print("Selected diagonals: ")
    print(skewd[:, :, - (w + 1):-1, w + 1:])
    print("Diagonal attention: ")
    print(diagonal_attn)
    
    print("------------------------------------------------------")
    diagonal_attn[:, 0, 1:w, 1:w] = skewd[:, 0, :w - 1, 1 - w:]
    print("Selected diagonals: ")
    print(skewd[:, 0, :w - 1, 1 - w:])
    print("Diagonal attention: ")
    print(diagonal_attn)
    
        
    diagonal_attn = diagonal_attn.view(1, 1, seq_len, 2*w + 1).transpose(2, 1)
    
    mask_invalid_locations(diagonal_attn, w, 1)
    
    print("Final attention: ")
    print(diagonal_attn)
    
    
    print("------------------------------------------------------")
    print(_get_invalid_locations_mask_fixed_dilation(8, 2, 1))

    
if __name__ == "__main__":
    visualize_attention()
    
    