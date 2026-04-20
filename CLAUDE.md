# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Course project implementing a JEPA-based self-supervised model for 12-lead ECG signals, adapted from [ECG-JEPA (arxiv 2410.08559)](https://arxiv.org/abs/2410.08559). Keep the paper and project proposal with the repository documentation. The design rationale for every architectural decision is in `design_backlog.md` — read it before making structural changes.

**Current scope:** core architecture only (`src/`). Training loop, dataloaders, EMA, and downstream evaluators do not exist yet and will be added separately.

## Running

No build step. Run smoke tests directly with Python:

```bash
# Verify all shapes and module wiring (no flash-attn required)
python3 -c "
import torch
from src.configs import dev_preset
from src.tokenizer import ECGTokenizer
from src.encoder import ECGEncoder
from src.predictor import ECGPredictor

tok_cfg, enc_cfg, pred_cfg = dev_preset()
enc_cfg.use_flash = False
pred_cfg.use_flash = False

tokenizer = ECGTokenizer(tok_cfg)
encoder   = ECGEncoder(enc_cfg, tokenizer)
predictor = ECGPredictor(pred_cfg)

signal  = torch.randn(2, 12, 1000)
patches = tokenizer.patchify(signal)
vis = torch.arange(0, 20, 2)
msk = torch.arange(1, 20, 2)
ctx = encoder(patches, visible_indices=vis)
pred = predictor(ctx, vis, msk)
print(ctx.shape, pred.shape)  # [2,120,384]  [2,12,10,384]
"
```

## Architecture

See `src/README.md` for the full integration guide (shapes, training-loop pseudo-code, gotchas). Summary:

**Three-component JEPA:**
- `ECGTokenizer` — 2-layer 1D CNN per patch → D-dim token. Also owns `patchify(signal)`.
- `ECGEncoder` — transformer over C×Q visible tokens (or all C×N for target / inference). 2D sinusoidal pos embed is added *before* masking so dropped tokens retain their true position.
- `ECGPredictor` — receives context encoder output, reshapes to `(bs×C, Q, D)` to process each lead independently, appends learnable mask tokens, runs transformer, returns predictions at masked positions only as `(bs, C, M, encoder_embed_dim)`.

**Masking is synchronized across leads.** `visible_indices` and `masked_indices` are 1D LongTensors of temporal indices applied identically to every lead.

**Positional embeddings:**
- Encoder: 2D sincos — splits `embed_dim` in half, first half = lead index, second half = temporal index.
- Predictor: 1D sincos — temporal positions only.

## Config system

```python
from src.configs import dev_preset, final_preset
tok_cfg, enc_cfg, pred_cfg = dev_preset()   # 6L/8H/D=384 encoder, ~11M params
tok_cfg, enc_cfg, pred_cfg = final_preset() # 12L/16H/D=768 encoder, ~86M params
```

`presets.py` enforces dim consistency (tokenizer output dim == encoder dim == predictor `encoder_embed_dim`, patch sizes match, etc.). If assembling configs manually, call `_check_consistency` or expect silent shape errors downstream.

## Flash Attention flag

`use_flash: bool` is set on both `EncoderConfig` and `PredictorConfig`. When `True`, `flash_attn.flash_attn_func` is used and `return_attn=True` will raise. When `False`, standard dot-product attention runs and attention weights are returned per block as `list[Tensor(bs, H, S, S)]`. The flag is frozen at construction — swap by rebuilding the module with the same weights loaded.

Flash Attention requires fp16/bf16 inputs; use `torch.autocast` when enabled.

## Key constraints

- `signal length T` must be divisible by `patch_size` (default 50). Trim/pad upstream.
- `embed_dim % num_heads == 0` enforced in presets.
- Predictor's `forward` takes `(context_tokens, visible_indices, masked_indices)` — not a boolean mask.
- Target encoder must have `requires_grad_(False)` and be EMA-updated by the training loop.
- Tokenizer ownership: each `ECGEncoder` holds its own `ECGTokenizer`. If the same tokenizer instance is shared between student and teacher, its weights are not EMA-updated.
