from dataclasses import dataclass, field
import torch
from torch import nn
import math

@dataclass
class BertConfig:
    """
    Configuration class to store the configuration of a `BertModel`.
    It is used to instantiate an BERT model according to the specified arguments,
    defining the model architecture.
    Parameters:
        vocab_size: int
            Vocabulary size of `inputs_ids` in `BertModel`.
        hidden_size: int
            Size of the encoder layers and the pooler layer.
        num_hidden_layers: int
            Number of hidden layers in the Transformer encoder.
        num_attention_heads: int
            Number of attention heads for each attention layer in
            the Transformer encoder.
        intermediate_size: int
            The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
        hidden_dropout_prob: float
            The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob: float
            The dropout ratio for the attention probabilities.
        max_position_embeddings: int
            The maximum value of the position index.
        type_vocab_size: int
            The vocabulary size of `token_type_ids`.
            The vocabulary size of `token_type_ids`.
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]
            0 corresponds to a sentence A token,
            1 corresponds to a sentence B token.
        initializer_range: float
            The standard deviation of the truncated_normal_initializer for
            initializing all weight matrices.
    """
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    position_embedding_type: str = "absolute"
    pad_token_id: int = 0    


class BertEmbeddings(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(
        self,
        input_ids,
        token_type_ids,
    ):
        """
        input_ids is shape (B, T) where T is either 128 or 512
        """
        T = input_ids.size(1)
        pos = torch.arange(0, T, dtype=torch.long, device = input_ids.device)
        pos_emb = self.position_embeddings(pos) # (T, 768)
        word_emb = self.word_embeddings(input_ids) # (B,T,768)
        tok_type_emb = self.token_type_embeddings(token_type_ids) # (B,T,768)
        x = pos_emb + word_emb + tok_type_emb
        x = self.norm(x)
        x = self.dropout(x)
        return x
    

class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        # heads are parallel streams, and outputs get concatenated.
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.hidden_size, config.hidden_size * 3)
        # output projection
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.n_head = config.num_attention_heads
        self.hidden_size = config.hidden_size

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality
        
        # calculate query, key, value for all heads in batch
        # C is hidden_size, which is 768 in BERT
        # nh is "number of heads", which is 12 in BERT
        # hs is "head size", which is C // nh = 768 // 12 = 64 in BERT
    
        qkv = self.c_attn(x) # (B, T, 3*C)
        q, k, v = qkv.split(self.hidden_size, dim=2) # (B, T, C) x 3
        # (B, T, C) -> (B, T, nh, C/nh) = (B, T, nh, 64) --transpose(1,2)--> (B, nh, T, 64)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, 64)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, 64)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, 64)
        

        # attention multiplies the head_size dimension (T,64) x (64,T) = (T,T)
        # (B, nh, T, 64) x (B, nh, 64, T) -> (B, nh, T, T)
        att = q @ k.transpose(2, 3)
        # scale q, k, v by 1/sqrt(64), the head size
        scale = 1.0 / math.sqrt(k.size(-1))
        att = att * scale
        # att describes the relation between the tokens in the sequence
        # how much token 0 should be a mixture of tokens 0 through T
        att = nn.functional.softmax(att, dim=-1)
        # Randomly sets some attention weights to zero during training,
        # meaning certain key-value pairs are ignored for that forward pass.
        # This prevents the model from over-relying on specific attention patterns.
        att = self.dropout(att)

        # re-mix the value tokens, by multiplying each token by the corresponding
        # weights in the attention matrix. Do this across all 64 dimensions
        y = att @ v # (B, nh, T, T) x (B, nh, T, 64) -> (B, nh, T, 64)
        
        # (B, nh, T, 64) -> (B, T, nh, 64) -> (B, T, nh*64 = 12*64 = 768)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.c_proj(y)
        # y = self.norm(y)
        return y
    
class BertBlock(nn.Module):
    def __init__(self, config):
        """
        Pre-LN helps stabilize training
        """
        super().__init__()
        self.scale_factor = 4
        self.attention = ScaledDotProductAttention(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * self.scale_factor),
            nn.GELU(), # apprixmiate tanh?
            nn.Linear(config.hidden_size * self.scale_factor, config.hidden_size),
        )
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, x):
        x = x + 
        h = self.attention(x)
        h = self.dropout(h)
        x = self.norm1(x + h)
        h = self.mlp(x)
        h = self.dropout(h)
        x = self.norm2(x + h)
        return x