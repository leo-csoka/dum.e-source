import math
import json
import os
import mlx.core as mx
import mlx.nn as nn
import tiktoken as tk

# --    TOKENIZER     --
class Tokenizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.dim = self.cfg.token_vector_len
        self.max_sequence_length = self.cfg.context_length
        self.encoding = tk.get_encoding("gpt2")
        self.token_embeddings = nn.Embedding(self.cfg.vocab_size, self.dim)
        self.token_positions = nn.Embedding(self.max_sequence_length, self.dim)

    def encode(self, text):
        token_ids = self.encoding.encode(text, allowed_special=set())
        if len(token_ids) > self.max_sequence_length:
            raise ValueError(
                f"text produces {len(token_ids)} tokens, "
                f"but the context length is {self.max_sequence_length}"
            )
        return token_ids

    def decode(self, token_ids):
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return self.encoding.decode(token_ids)

    def embed(self, text):
        token_ids = self.encode(text)
        token_ids = mx.array([token_ids])
        positions = mx.arange(token_ids.shape[1])

        token_vectors = self.token_embeddings(token_ids)
        position_vectors = self.token_positions(positions)
        return token_vectors + position_vectors

    def unembed(self, embedding):
        if embedding.ndim not in (2, 3):
            raise ValueError(
                "embedding must be shaped [sequence, width] or [batch, sequence, width]"
            )
        if embedding.shape[-1] != self.dim:
            raise ValueError(
                f"embedding width must be {self.dim}, got {embedding.shape[-1]}"
            )

        last_embedding = embedding[-1] if embedding.ndim == 2 else embedding[:, -1, :]
        logits = mx.matmul(last_embedding, self.token_embeddings.weight.T)
        return mx.argmax(logits, axis=-1)

    def __call__(self, text):
        return self.embed(text)


# --    LAYERNORM     --
class LayerNorm(nn.Module):
    def __init__(self, config, eps=1e-6):
        super().__init__()
        self.cfg = config
        self.dim = self.cfg.token_vector_len
        self.gamma = mx.ones(self.dim)
        self.beta = mx.zeros(self.dim)
        self.eps = eps

    def __call__(self, x):
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / mx.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


# --    GATED SWIGLU MLP     --
class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.dim = self.cfg.token_vector_len
        self.hidden_dim = round(self.cfg.hidden_upscale * self.dim)

        self.W_gate = mx.random.normal((self.dim, self.hidden_dim)) * 0.02
        self.b_gate = mx.zeros(self.hidden_dim)

        self.W_up = mx.random.normal((self.dim, self.hidden_dim)) * 0.02
        self.b_up = mx.zeros(self.hidden_dim)

        self.W_out = mx.random.normal((self.hidden_dim, self.dim)) * 0.02
        self.b_out = mx.zeros(self.dim)

    def __call__(self, x):
        gate = mx.matmul(x, self.W_gate) + self.b_gate
        up = mx.matmul(x, self.W_up) + self.b_up
        hidden = up * (gate * mx.sigmoid(gate))
        return mx.matmul(hidden, self.W_out) + self.b_out


# --    ATTENTION     --
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.dim = self.cfg.token_vector_len
        self.num_heads = self.cfg.num_heads
        self.head_dim = self.cfg.head_dim
        if self.num_heads * self.head_dim != self.dim:
            raise ValueError("token_vector_len must be divisible by head_dim")
        self.Wq = mx.random.normal((self.dim, self.dim)) * 0.02
        self.Wk = mx.random.normal((self.dim, self.dim)) * 0.02
        self.Wv = mx.random.normal((self.dim, self.dim)) * 0.02
        self.Wo = mx.random.normal((self.dim, self.dim)) * 0.02
        self.bq = mx.zeros(self.dim)
        self.bk = mx.zeros(self.dim)
        self.bv = mx.zeros(self.dim)
        self.bo = mx.zeros(self.dim)

    def __call__(self, x):
        if x.ndim != 3:
            raise ValueError("attention expects input shaped [batch, sequence, width]")

        batch_size, sequence_length, _ = x.shape
        Q = mx.matmul(x, self.Wq) + self.bq
        K = mx.matmul(x, self.Wk) + self.bk
        V = mx.matmul(x, self.Wv) + self.bv

        Q = mx.transpose(mx.reshape(Q, (batch_size, sequence_length, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        K = mx.transpose(mx.reshape(K, (batch_size, sequence_length, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        V = mx.transpose(mx.reshape(V, (batch_size, sequence_length, self.num_heads, self.head_dim)), (0, 2, 1, 3))

        scores = mx.matmul(Q, mx.transpose(K, (0, 1, 3, 2))) / math.sqrt(self.head_dim)
        causal_mask = mx.triu(mx.ones((sequence_length, sequence_length)), 1)
        scores = mx.where(causal_mask == 1, -float("inf"), scores)
        attention_weights = mx.softmax(scores, axis=-1)
        context = mx.matmul(attention_weights, V)
        context = mx.transpose(context, (0, 2, 1, 3))
        context = mx.reshape(context, (batch_size, sequence_length, self.dim))

        return mx.matmul(context, self.Wo) + self.bo


# --    TRANSFORMER BLOCK     --
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = LayerNorm(config)
        self.attention = Attention(config)
        self.ffn_norm = LayerNorm(config)
        self.perceptron = SwiGLU(config)

    def __call__(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.perceptron(self.ffn_norm(x))


# --    LANGUAGE MODEL     --
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.tokenizer = Tokenizer(config)
        self.blocks = [TransformerBlock(config) for _ in range(config.num_blocks)]
        self.final_norm = LayerNorm(config)

    def __call__(self, token_ids):
        if token_ids.ndim != 2:
            raise ValueError("token_ids must be shaped [batch, sequence]")
        if token_ids.shape[1] > self.cfg.context_length:
            raise ValueError(
                f"sequence length must be at most {self.cfg.context_length}"
            )

        positions = mx.arange(token_ids.shape[1])
        x = self.tokenizer.token_embeddings(token_ids)
        x = x + self.tokenizer.token_positions(positions)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return mx.matmul(x, self.tokenizer.token_embeddings.weight.T)


def load_checkpoint(checkpoint_dir):
    from config import Config

    with open(os.path.join(checkpoint_dir, "config.json"), encoding="utf-8") as file:
        config = Config(**json.load(file))
    model = Transformer(config)
    model.load_weights(os.path.join(checkpoint_dir, "weights.npz"))
    mx.eval(model.parameters())
    return model