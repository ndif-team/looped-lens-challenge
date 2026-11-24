# Findings

What we know, with the evidence for each claim. Newest at the top of each section.

## The model (Inkling 2 Mini)

- **No single residual stream.** Layers sum into 6-layer *blocks*, and at layers 0, 6, 12 and 18 the finished block is frozen into `block_states`. Each sublayer's input is a learned-query softmax mix of the frozen blocks plus the active block. The output is `final_input(block_states, hidden) → norm → lm_head`, with logit softcap 30.
  - Source: `transformers/models/inkling2_loop/modeling_inkling2_loop.py`, `Inkling2LoopTextModel.forward`, `Inkling2LoopBlockAttentionResidual`.
- **Passes 2 and 3 restart from the prelude's state.** The previous pass reaches them only through four *messages* per layer (query, key, value, MoE), blended in by rank-64 tanh gates. Lens readouts therefore drop at the start of each pass and rise at its end.
  - Source: the same file, and <https://nnsight.net/blog/2026/11/20/opening-the-box-of-inkling-2-mini-using-nnsight-ndif/>.
- **A layer's call is `layer(hidden, block_states, [messages], ...)`,** so on NDIF `hidden, blocks = layer.inputs[0][:2]`. `layer.output` at visit v equals the input of visit v+1 within a pass. The probe checked this at 43 of 43 visits.
- **Some NDIF replicas split the model over two GPUs.** `block_states` and `hidden` can then be on different devices, so move tensors to one device before `torch.cat`.
  - Seen: a probe request failed with `Expected all tensors to be on the same device … cuda:1 … cuda:0`.
- **At the chat read position (`<|message_model|>`), the model's next token is `<|content_thinking|>`.** The `final_input`-mixed readout late in the network fills with that format prediction. On reasoning prompts the last visits show the topic (`arithmetic`, `week`) but not the answer.

## The scoring

- **Each scored dataset is one family** (`evaluator/evaluate.py`, `_questions`).
- **A question's readouts need distinct layer labels,** so there are at most 10 readouts, one per visit. Each can come from any computation at that visit, at any position.
- **Practice (10 questions) is a poor guide.** A lens at 0.90 on practice scored 0.660 on the hidden sets (see [leaderboard.md](leaderboard.md)).

## Lenses

- **Starter logit lens** (active block, read position, visits 12, 24, 34–36, 43–47): 0.60 on practice and 0.662 on the leaderboard.
- **Plain-format prompts** score 0 at the read position (the final `.`) on our first probe batch. At the subject's last token, about 8 from the end, they score 1.00 for sport questions. Being checked on the full dev set.
