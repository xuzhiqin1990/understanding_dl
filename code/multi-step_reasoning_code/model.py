import torch
import torch.utils.data as Data
from torch import nn
import numpy as np
import torch.nn.init as init



def get_attn_pad_mask(seq_q, seq_k):
    '''
        seq_q: [batch_size, seq_len]
        seq_k: [batch_size, seq_len]
        seq_len could be src_len or it could be tgt_len
        seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

def get_attn_subsequence_mask(seq, device):
    """
    seq: [batch_size, tgt_len]
    """
    batch_size, tgt_len = seq.size()
    attn_shape = (batch_size, tgt_len, tgt_len)
    subsequence_mask = torch.triu(torch.ones(attn_shape, dtype=torch.uint8, device=device), diagonal=1)
    return subsequence_mask

def attn_mask(X_input, device):
    '''
        X_input: [batch_size, tgt_len]
    '''
    dec_self_attn_pad_mask = get_attn_pad_mask(X_input, X_input) # [batch_size, tgt_len, d_model] 遮挡padding部分
    dec_self_attn_subsequence_mask = get_attn_subsequence_mask(X_input, device) # [batch_size, tgt_len, d_model] 遮挡未来时刻的词
    # 两个mask之和只要有一个为1的地方，就为1
    dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0) # [batch_size, tgt_len, d_model] 

    return dec_self_attn_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()

        self.n_head = args.n_heads
        self.d_k = args.d_k
        self.d_v = args.d_v
        self.d_model = args.d_model
    
        self.W_Q = nn.Linear(args.d_model, args.d_k * args.n_heads, bias=False)
        self.W_K = nn.Linear(args.d_model, args.d_k * args.n_heads, bias=False)
        self.W_V = nn.Linear(args.d_model, args.d_v * args.n_heads, bias=False)
        self.fc = nn.Linear(args.n_heads * args.d_v, args.d_model, bias=False)
        self.layernorm = nn.LayerNorm(args.d_model)

        # 设置初始化
        bound_qkv = (6 / (args.d_model))**(args.std_rate)
        bound_fc = (6 / (args.n_heads * args.d_v))**(args.std_rate)
        init.uniform_(self.W_Q.weight, -bound_qkv, bound_qkv)
        init.uniform_(self.W_K.weight, -bound_qkv, bound_qkv)
        init.uniform_(self.W_V.weight, -bound_qkv, bound_qkv)
        init.uniform_(self.fc.weight, -bound_fc, bound_fc)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # scores : [batch_size, n_heads, len_q, len_k]
        attn = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        masked_attn = attn.masked_fill(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
        softmax_attn = nn.Softmax(dim=-1)(masked_attn)
        qkv = torch.matmul(softmax_attn, V)  # [batch_size, n_heads, len_q, d_v] 

        qkv_out = qkv.transpose(1, 2).reshape(batch_size, -1, self.n_head * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(qkv_out)  # [batch_size, len_q, d_model]

        return self.layernorm(output + residual), softmax_attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, args):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(args.d_model, args.d_feedforward, bias=False),
            nn.ReLU(),
            nn.Linear(args.d_feedforward, args.d_model, bias=False)
        )
        self.layernorm=nn.LayerNorm(args.d_model)

        # 设置初始化
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                input_size=layer.weight.size(1)
                bound = (6.0 / (input_size))**(args.std_rate)
                init.uniform_(layer.weight, -bound, bound)

    def forward(self, hidden_state):
        '''
        hidden_state: [batch_size, seq_len, d_model]
        '''
        residual = hidden_state
        output = self.fc(hidden_state)
        return self.layernorm(output + residual) # [batch_size, seq_len, d_model]

class DecoderLayer(nn.Module):
    def __init__(self, args):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)

    def forward(self, hidden_state, dec_self_attn_mask):
        '''
            hidden_state: [batch_size, tgt_len, d_model]
            dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        '''
        # Attention层
        # hidden_state: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        hidden_state, dec_self_attn = self.dec_self_attn(hidden_state, hidden_state, hidden_state, dec_self_attn_mask)

        # 非线性层
        hidden_state = self.pos_ffn(hidden_state)  # [batch_size, tgt_len, d_model]
        return hidden_state, dec_self_attn


class Decoder(nn.Module):
    def __init__(self, args, device):
        super(Decoder, self).__init__()
        self.device = device
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layers)])

    def forward(self, hidden_state, dec_self_attn_mask):
        '''
            hidden_state: [batch_size, tgt_len]
        '''
        dec_self_attns = []
        for layer in self.layers:
            # hidden_state: [batch_size, tgt_len, d_model]
            # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
            hidden_state, dec_self_attn = layer(hidden_state, dec_self_attn_mask)
   
            dec_self_attns.append(dec_self_attn)

        return hidden_state, dec_self_attns

class Embedding(nn.Module):
    def __init__ (self, args, device):
        super(Embedding, self).__init__()
        self.device = device
        self.tgt_emb = nn.Embedding(args.vocab_size, args.d_model)
        self.pos_emb = nn.Embedding(args.seq_len, args.d_model)


        # 设置初始化
        bound1 = (6 / (args.vocab_size))**(args.std_rate)
        bound2 = (6 / (args.seq_len))**(args.std_rate)
        init.uniform_(self.tgt_emb.weight, -bound1, bound1)
        init.uniform_(self.pos_emb.weight, -bound2, bound2)

    def forward(self, X_input):
        seq_len = X_input.size(1)
        pos = torch.arange(seq_len, dtype = torch.long, device = self.device)
        pos = pos.unsqueeze(0).expand_as(X_input)

        tgt_emb = self.tgt_emb(X_input)
        pos_emb = self.pos_emb(pos)
        emb = tgt_emb + pos_emb

        return emb

class myGPT(nn.Module):
    def __init__(self, args, device):
        super(myGPT, self).__init__()

        self.device = device
        self.embedding = Embedding(args, device)
        self.decoder = Decoder(args, device)
        self.projection = nn.Linear(args.d_model, args.vocab_size)

        # 设置初始化
        bound = (6 / (args.d_model))**(args.std_rate)
        init.uniform_(self.projection.weight, -bound, bound)

    def forward(self, X_input):
        """
            dec_inputs: [batch_size, tgt_len]
        """
        hidden_state = self.embedding(X_input)

        dec_self_attn_mask = attn_mask(X_input, self.device)

        hidden_state, dec_self_attns = self.decoder(hidden_state, dec_self_attn_mask)
    
        dec_logits = self.projection(hidden_state)

        return dec_logits.view(-1, dec_logits.size(-1)), dec_self_attns




