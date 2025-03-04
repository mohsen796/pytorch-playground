import torch
import torch.nn as nn

from transformers.core import PositionalEncoding
from transformers.decoder import DecoderLayer
from transformers.encoder import EncoderLayer


class Transformer(nn.Module):
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
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layer = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layer = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_len = tgt.size(1)
        nopeak_mask = (
            1
            - torch.triu(
                torch.ones((1, seq_len, seq_len), device=tgt.device), diagonal=1
            )
        ).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedding = self.dropout(
            self.positional_encoding(self.encoder_embedding(src))
        )
        tgt_embedding = self.dropout(
            self.positional_encoding(self.decoder_embedding(tgt))
        )

        enc_output = src_embedding
        for enc_layer in self.encoder_layer:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedding
        for dec_layer in self.decoder_layer:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
