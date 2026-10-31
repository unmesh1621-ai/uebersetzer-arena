"""Greedy, beam-search and temperature-sampling decoders for Seq2SeqTransformer."""
import torch


def _strip(seq, eos):
    return seq[: seq.index(eos)] if eos in seq else seq


@torch.no_grad()
def _autoregressive(model, src, bos, eos, max_len, pick):
    memory, src_mask = model.encode(src)
    ys = torch.full((src.size(0), 1), bos, dtype=torch.long, device=src.device)
    done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
    for _ in range(max_len):
        nxt = pick(model.decode_step(memory, src_mask, ys).float())
        nxt = nxt.masked_fill(done, eos)
        ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
        done |= nxt.eq(eos)
        if done.all():
            break
    return [_strip(row[1:], eos) for row in ys.tolist()]


def greedy_decode(model, src, bos, eos, max_len):
    return _autoregressive(model, src, bos, eos, max_len, lambda logits: logits.argmax(-1))


def sample_decode(model, src, bos, eos, max_len, temperature, generator):
    def pick(logits):
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, 1, generator=generator).squeeze(1)

    return _autoregressive(model, src, bos, eos, max_len, pick)


def _lp(length, alpha):
    return ((5 + length) / 6) ** alpha


@torch.no_grad()
def beam_decode(model, src, bos, eos, max_len, beam=4, alpha=0.6):
    """Per-sentence beam search with GNMT length normalisation ((5+len)/6)^alpha."""
    results = []
    for i in range(src.size(0)):
        memory, src_mask = model.encode(src[i : i + 1])
        beams = [([bos], 0.0)]
        finished = []
        for _ in range(max_len):
            ys = torch.tensor([b[0] for b in beams], device=src.device)
            logp = torch.log_softmax(
                model.decode_step(memory.expand(len(beams), -1, -1), src_mask.expand(len(beams), -1), ys).float(), -1
            )
            cand = []
            for b, (seq, score) in enumerate(beams):
                top = logp[b].topk(beam)
                cand += [(seq + [t], score + s) for s, t in zip(top.values.tolist(), top.indices.tolist())]
            cand.sort(key=lambda c: c[1], reverse=True)
            beams = []
            for rank, (seq, score) in enumerate(cand):
                if seq[-1] == eos:
                    if rank < beam:  # only hypotheses that would have survived in the beam may finish
                        finished.append((seq, score / _lp(len(seq) - 1, alpha)))
                else:
                    beams.append((seq, score))
                if len(beams) == beam:
                    break
            if not beams:
                break
            # Scores only decrease, so the best live beam can at most reach its score / lp(max_len).
            if finished and max(f[1] for f in finished) >= beams[0][1] / _lp(max_len, alpha):
                break
        if not finished:
            finished = [(seq, score / _lp(len(seq) - 1, alpha)) for seq, score in beams]
        best = max(finished, key=lambda f: f[1])[0]
        results.append(_strip(best[1:], eos))
    return results
