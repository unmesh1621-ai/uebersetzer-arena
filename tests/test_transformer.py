import torch

from src.transformer_scratch.decode import beam_decode, greedy_decode, sample_decode
from src.transformer_scratch.model import Seq2SeqTransformer

V, PAD, BOS, EOS = 20, 0, 2, 3


def small_model(dropout=0.0):
    torch.manual_seed(0)
    return Seq2SeqTransformer(V, d_model=32, nhead=2, num_layers=2, dim_ff=64, dropout=dropout, pad_id=PAD)


def test_output_shape():
    m = small_model().eval()
    src = torch.randint(4, V, (2, 7))
    tgt = torch.randint(4, V, (2, 5))
    assert m(src, tgt).shape == (2, 5, V)


def test_causal():
    m = small_model().eval()
    src = torch.randint(4, V, (1, 6))
    tgt = torch.randint(4, V, (1, 5))
    out1 = m(src, tgt)
    tgt2 = tgt.clone()
    tgt2[0, 3] = (tgt2[0, 3] + 1 - 4) % (V - 4) + 4
    out2 = m(src, tgt2)
    assert torch.allclose(out1[0, :3], out2[0, :3], atol=1e-5)
    assert not torch.allclose(out1[0, 3], out2[0, 3], atol=1e-5)


def test_src_padding_ignored():
    m = small_model().eval()
    src = torch.randint(4, V, (1, 6))
    tgt = torch.randint(4, V, (1, 4))
    padded = torch.cat([src, torch.full((1, 3), PAD)], dim=1)
    assert torch.allclose(m(src, tgt), m(padded, tgt), atol=1e-5)


def _copy_data():
    seqs = [[5, 6, 7], [8, 9, 10, 11], [12, 13], [14, 15, 16]]
    maxlen = 4
    src = torch.tensor([s + [PAD] * (maxlen - len(s)) for s in seqs])
    tgt = [[BOS] + s + [EOS] for s in seqs]
    tl = max(len(t) for t in tgt)
    tgt = torch.tensor([t + [PAD] * (tl - len(t)) for t in tgt])
    return seqs, src, tgt


def _overfit():
    m = small_model()
    seqs, src, tgt = _copy_data()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD)
    m.train()
    for _ in range(300):
        logits = m(src, tgt[:, :-1])
        loss = loss_fn(logits.reshape(-1, V), tgt[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m.eval(), seqs, src


def test_greedy_overfits_copy_task():
    m, seqs, src = _overfit()
    assert greedy_decode(m, src, BOS, EOS, max_len=8) == seqs


def test_beam1_equals_greedy_and_beam4_works():
    m, seqs, src = _overfit()
    g = greedy_decode(m, src, BOS, EOS, max_len=8)
    assert beam_decode(m, src, BOS, EOS, max_len=8, beam=1) == g
    assert beam_decode(m, src, BOS, EOS, max_len=8, beam=4) == seqs


def test_sample_low_temperature_matches_greedy():
    m, seqs, src = _overfit()
    gen = torch.Generator().manual_seed(42)
    assert sample_decode(m, src, BOS, EOS, max_len=8, temperature=0.01, generator=gen) == seqs
