from dataclasses import dataclass, field
import torch
from torch import nn
import math
from typing import Optional


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

        pre_layer_norm: bool
            Whether to use post layer norm or pre layer norm.
            pre-LN: Layer normalization is applied to the input of the attention block and ffn block, but not to the residual connection.
            post-LN: Layer normalization is applied after the residual connection.

            post-LN is the default in BERT, but pre-LN may improve training stability
            in deep transformer models due to gradient amplification.
            https://arxiv.org/abs/2002.04745
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
    pre_layer_norm: bool = False


class BertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position, and token_type embeddings.

    The default BERT model uses post-LN configuration, where LayerNorm is applied after the residual connection.
    When pre-LN is used, LayerNorm is applied to the input of the attention block and ffn block, but not to the residual connection.
    Because LayerNorm is applied at the input of the attention block, the embedding LayerNorm is redundant.
    The Transformer's first LayerNorm can handle scaling and stabilization directly from the combined embeddings (post-Dropout).

    Args:
        config: BertConfig
            Configuration for the BERT model.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.pre_layer_norm = getattr(config, "pre_layer_norm", False)
        if not self.pre_layer_norm:
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings, dtype=torch.long).expand((1, -1)),
            persistent=False,
        )

    def forward(self, input_ids: torch.LongTensor, token_type_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        """
        Forward pass for the embeddings.

        Args:
            input_ids: torch.Tensor
                Tensor of input token IDs.
            token_type_ids: torch.Tensor
                Tensor of token type IDs.

        Returns:
            torch.Tensor: Combined embeddings.
        """
        T = input_ids.size(1)

        position_ids = self.position_ids[:, :T]
        if token_type_ids is None:
            token_type_ids = torch.zeros(T, dtype=torch.long, device=self.position_ids.device)

        word_emb = self.word_embeddings(input_ids)  # (B,T,768)
        tok_type_emb = self.token_type_embeddings(token_type_ids)  # (B,T,768)
        x = word_emb + tok_type_emb

        if self.position_embedding_type == "absolute":
            position_emb = self.position_embeddings(position_ids)
            x += position_emb

        # LayerNorm stabilizes the combined embeddings by ensuring zero mean and unit variance
        if not self.pre_layer_norm:
            x = self.norm(x)

        # Regularizes the embeddings before they enter the Transformer stack,
        # preventing overfitting to specific embedding patterns.
        x = self.dropout(x)
        return x


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.

    Args:
        config: BertConfig
            Configuration for the BERT model.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0

        # heads are parallel streams, and outputs get concatenated.
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.hidden_size, config.hidden_size * 3)
        # output projection
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout_attn = nn.Dropout(config.attention_probs_dropout_prob)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.hidden_size = config.hidden_size  # 768
        self.n_head = config.num_attention_heads  # 12
        self.head_size = config.hidden_size // config.num_attention_heads  # 64

    def forward(self, x: torch.Tensor, attention_mask: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass for the attention mechanism.

        Args:
            x: torch.Tensor
                Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying attention.
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality

        # calculate query, key, value for all heads in batch
        # C is hidden_size, which is 768 in BERT
        # nh is "number of heads", which is 12 in BERT
        # hs is "head size", which is C // nh = 768 // 12 = 64 in BERT

        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.hidden_size, dim=2)  # (B, T, C) x 3
        # (B, T, C) -> (B, T, nh, C/nh) = (B, T, nh, 64) --transpose(1,2)--> (B, nh, T, 64)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, 64)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, 64)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, 64)

        # attention multiplies the head_size dimension (T,64) x (64,T) = (T,T)
        # (B, nh, T, 64) x (B, nh, 64, T) -> (B, nh, T, T)
        att = q @ k.transpose(2, 3)
        att = att / math.sqrt(self.head_size)

        # attention mask is a binary mask of shape (B,T) that is 1 for positions we want to attend to
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        # Broadcast to (B, nh, T, T) by applying it to the key dimension
        # Mask out padding by setting scores to -inf where attn_mask is 0
        attn = attn.masked_fill(attention_mask == 0, float("-inf"))  # (B, nh, T, T)

        # att describes the relation between the tokens in the sequence
        # how much token 0 should be a mixture of tokens 0 through T
        att = nn.functional.softmax(att, dim=-1)
        # Randomly sets some attention weights to zero during training,
        # meaning certain key-value pairs are ignored for that forward pass.
        # This prevents the model from over-relying on specific attention patterns.
        att = self.dropout_attn(att)

        # re-mix the value tokens, by multiplying each token by the corresponding
        # weights in the attention matrix. Do this across all 64 dimensions
        y = att @ v  # (B, nh, T, T) x (B, nh, T, 64) -> (B, nh, T, 64)

        # (B, nh, T, 64) -> (B, T, nh, 64) -> (B, T, nh*64 = 12*64 = 768)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.c_proj(y)
        y = self.dropout(y)
        return y


class BertFFN(nn.Module):
    """
    Feed-Forward Network (FFN) used in BERT.

    Args:
        config: BertConfig
            Configuration for the BERT model.
    """

    def __init__(self, config):
        super().__init__()
        self.scale_factor = 4
        self.layers = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * self.scale_factor),
            # approximate GELU with tanh is sufficient for BERT
            nn.GELU(approximate="tanh"),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size * self.scale_factor, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )

    def forward(self, x):
        """
        Forward pass for the feed-forward network.

        Args:
            x: torch.Tensor
                Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the feed-forward network.
        """
        return self.layers(x)


class BertBlock(nn.Module):
    """
    Single block of the BERT model, consisting of attention and feed-forward layers.

    Args:
        config: BertConfig
            Configuration for the BERT model.
    """

    def __init__(self, config):
        """
        BERT uses post-LN where layer normalization
        is applied after the residual connection,
        as opposed to pre-LN where layer normalization is applied
        to the input of the attention block and ffn block, but not to the
        residual connection.
        """
        super().__init__()
        self.attention = ScaledDotProductAttention(config)
        self.ffn = BertFFN(config)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pre_layer_norm = getattr(config, "pre_layer_norm", False)

    def forward(self, x):
        """
        Forward pass for the BERT block.
        This

        Args:
            x: torch.Tensor
                Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying attention and feed-forward layers.
        """
        if self.pre_layer_norm:
            x += self.attention(self.ln1(x))
        else:
            x = self.ln1(x + self.attention(x))

        if self.pre_layer_norm:
            x += self.ffn(self.ln2(x))
        else:
            x = self.ln2(x + self.ffn(x))
        return x
