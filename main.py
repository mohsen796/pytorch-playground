# Check that MPS is available


from transformers.training import TransformerTrainer


def train_transformer():
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1

    trainer = TransformerTrainer(
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    )
    src_data, tgt_data = trainer.prepare_data()
    trainer.train(src_data, tgt_data)


if __name__ == "__main__":
    train_transformer()
