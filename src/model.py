"""Contains the definition of Alexandria model"""

import torch
import torch.nn as nn


class AlexandriaMultiheadAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.n_heads = config["n_heads"]
        self.d_head = config["d_model"] // self.n_heads
        self.seq_len = config["seq_len"]
        self.mask = torch.tril(torch.ones(self.seq_len, self.seq_len))
        self.W_v = nn.Linear(
            in_features=config["d_model"], out_features=config["d_model"], bias=False
        )
        self.W_k = nn.Linear(
            in_features=config["d_model"], out_features=config["d_model"], bias=False
        )
        self.W_q = nn.Linear(
            in_features=config["d_model"], out_features=config["d_model"], bias=False
        )

    def forward(self, x, attention_mask=None):

        batch_size, seq_len, _ = x.shape

        V = self.W_v(x)
        K = self.W_k(x)
        Q = self.W_q(x)

        # Divide the projection in the n_heads heads
        # Pass from (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_head)
        # E.g.: (32, 256, 256) -> (32, 256, 8, 32)
        # Q,K and V need to be (batch_size, n_heads, seq_len, d_head)

        V = V.view(batch_size, seq_len, self.n_heads, self.d_head)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_head)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_head)
        V = V.transpose(1, 2)  # (batch_size, n_heads, seq_len, d_head)
        Q = Q.transpose(1, 2)  # (batch_size, n_heads, seq_len, d_head)
        K = K.transpose(1, 2)  # (batch_size, n_heads, seq_len, d_head)

        alphas = (
            Q @ K.transpose(-2, -1) / torch.sqrt(self.d_head)
        )  # (batch_size, n_heads, seq_len, d_head) @ (batch_size, n_heads, d_head, seq_len)
        # alphas is (batch_size, n_heads, seq_len, seq_len)

        alphas = alphas.masked_fill(self.mask == 0, -torch.inf)

        # Apply the attention_mask (not the causal) to generate the last alpha representation
        if attention_mask:
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(
                2
            )  # for broadcasting: (batch_size, 1, 1, seq_len)
            alphas = alphas.masked_fill(padding_mask == 0, -torch.inf)

        Y = (
            torch.softmax(alphas) @ V
        )  # (batch_size, n_heads, seq_len, seq_len) @ (batch_size, n_heads, seq_len, d_head)
        return Y  # (batch_size, n_heads, seq_len, d_head)


class AlexandriaTransformerBlock(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.d_model = config["d_model"]
        self.d_ff = config["d_ff"]
        self.norm1 = nn.LayerNorm(self.d_model)
        self.attention = AlexandriaMultiheadAttention(config)
        self.W_o = nn.Linear(self.d_model, self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.first_projection = nn.Linear(self.d_model, self.d_ff)
        self.relu_activation = nn.ReLU()
        self.second_projection = nn.Linear(self.d_ff, self.d_model)

    def forward(self, x, attention_mask=None):

        batch_size, seq_len, d_model = x.shape

        normalized_x = self.norm1(x)
        Y = self.attention(normalized_x, attention_mask)  # (batch_size, n_heads, seq_len, d_head)
        # Check the concat operation, in this case, I might need only to execute a view
        Y = Y.transpose(1, 2)
        Y = Y.view(batch_size, seq_len, d_model)
        # Input to the W_o
        outputs = self.W_o(Y)
        outputs = outputs + x
        residual_outputs = outputs
        norm_outputs = self.norm2(outputs)
        outputs = self.first_projection(norm_outputs)
        outputs = self.relu_activation(outputs)
        outputs = self.second_projection(outputs)
        outputs = outputs + residual_outputs

        return outputs


class AlexandriaModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.d_model = config["d_model"]
        self.vocab_size = config["vocab_size"]
        self.token_embeddings = nn.Embedding(config["vocab_size"], config["d_model"])
        self.positional_embeddings = nn.Embedding(config["seq_len"], config["d_model"])
        self.attention_blocks = nn.ModuleList(
            [AlexandriaTransformerBlock(config) for _ in range(config["n_layers"])]
        )
        self.norm = nn.LayerNorm(self.d_model)
        self.W_out = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, x):

        x = x['input_ids']
        attention_mask = x['attention_mask']

        _, seq_len, _ = x.shape

        token_emb = self.token_embeddings(x)
        pos_emb = self.positional_embeddings(torch.arange(seq_len))
        x = token_emb + pos_emb
        for transformer in self.attention_blocks:
            x = transformer(x, attention_mask)
        x_norm = self.norm(x)
        outputs = self.W_out(x_norm)

        return outputs
