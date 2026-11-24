"""Readouts the probe compares, written to run inside an NDIF trace (registered with nnsight).

Every readout here unembeds with the model's own final norm and head; they differ in which state
they unembed:

- ``active``: the state entering the layer: only the block being built (the starter's lens);
- ``mixed``: that state mixed with the finished blocks by ``final_input``, the mix the model
  itself applies before its head;
- ``mlp``: the input of the layer's MoE, after the block mix and, in passes 2 and 3, the message
  from the previous pass;
- ``out``: the layer's output, mixed with the finished blocks as they stand after the layer.
"""

import torch

READOUTS = ("active", "mixed", "mlp", "out")


def llm(model):
    return model.model.language_model


def unembed(model, x, dtype):
    """Logits of states ``x`` [..., hidden] under the final norm and head (softcap left out: it
    keeps the order)."""
    return model.lm_head(llm(model).norm(x).to(dtype)).float()


def mixed(model, blocks, partial):
    """The model's final block mix: finished ``blocks`` [b, s, n, hidden] and ``partial`` [b, s, hidden]."""
    return llm(model).final_input(blocks, partial)[0]


def ranks(logits, variant_ids, max_rank: int = 30000):
    """Rank (0 = best) of each variant token in each row of ``logits`` [rows, vocab]: [rows, variants]."""
    target = logits[:, variant_ids]  # [rows, variants]
    better = (logits.unsqueeze(1) > target.unsqueeze(-1)).sum(-1)
    return better.clamp(max=max_rank).to(torch.int16)
