from transformers.transformer import Transformer

import torch
import torch.nn as nn
import torch.optim as optim


class TransformerTrainer:
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    ):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_length = max_seq_length
        self.dropout = dropout

        self.transformer_model = Transformer(
            src_vocab_size,
            tgt_vocab_size,
            d_model,
            num_heads,
            num_layers,
            d_ff,
            max_seq_length,
            dropout,
        )

    def prepare_data(self):
        src_data = torch.randint(1, self.src_vocab_size, (64, self.max_seq_length))
        tgt_data = torch.randint(1, self.tgt_vocab_size, (64, self.max_seq_length))
        return src_data, tgt_data

    def train(self, src_data, tgt_data):
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(
            self.transformer_model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
        )

        self.transformer_model.train()

        for epoch in range(100):
            optimizer.zero_grad()
            output = self.transformer_model(src_data, tgt_data[:, :-1])
            loss = criterion(
                output.contiguous().view(-1, self.tgt_vocab_size),
                tgt_data[:, 1:].contiguous().view(-1),
            )
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
