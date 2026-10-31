"""Encoder-decoder Transformer (Vaswani et al., 2017) implemented on top of PyTorch layers."""
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal position encodings added to the token embeddings."""

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model=256, nhead=4, num_layers=4, dim_ff=1024, dropout=0.3, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        # One embedding table shared by encoder, decoder and output layer (shared EN/DE vocabulary).
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model)
        self.drop = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers, norm=nn.LayerNorm(d_model))
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.out.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, mean=0.0, std=d_model ** -0.5)
        with torch.no_grad():
            self.embed.weight[pad_id].zero_()

    def _embed(self, tokens):
        return self.drop(self.pos(self.embed(tokens) * math.sqrt(self.d_model)))

    def encode(self, src):
        src_mask = src.eq(self.pad_id)
        memory = self.encoder(self._embed(src), src_key_padding_mask=src_mask)
        return memory, src_mask

    def _decode(self, memory, src_mask, tgt_in):
        causal = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1), device=tgt_in.device, dtype=torch.bool)
        h = self.decoder(
            self._embed(tgt_in),
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_in.eq(self.pad_id),
            memory_key_padding_mask=src_mask,
        )
        return self.out(h)

    def forward(self, src, tgt_in):
        memory, src_mask = self.encode(src)
        return self._decode(memory, src_mask, tgt_in)

    def decode_step(self, memory, src_mask, ys):
        """Logits for the next token given the prefix ys [B, t]."""
        return self._decode(memory, src_mask, ys)[:, -1]
