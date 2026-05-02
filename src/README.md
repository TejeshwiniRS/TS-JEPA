# ECG-JEPA — Core Architecture

This package implements the core architecture of a Joint-Embedding Predictive Architecture (JEPA) for 12-lead ECG, adapted from [Kim, 2026 (arxiv 2410.08559)](https://arxiv.org/abs/2410.08559).

**Scope of this package:** model architectures only — tokenizer, encoder, predictor, and shared transformer building blocks. The training loop, dataloaders, EMA bookkeeping, loss, and downstream evaluators are **not** included and are the responsibility of the caller.

See `/Users/shadyali/TS-JEPA/design_backlog.md` for the full design rationale. This README focuses on the interface a training-loop author needs.

---

## Architecture overview

JEPA pretraining uses three networks:

- **Context encoder** (student): sees only the *visible* patches. Gradients flow here.
- **Target encoder** (teacher): sees *all* patches. Updated as an EMA of the context encoder — no gradients.
- **Predictor**: takes the context encoder's output plus per-lead learnable mask tokens and predicts the target encoder's representation at the masked positions.

At inference the context (student) encoder is used alone; its per-patch outputs are average-pooled into a single D-dim vector.

```
           ┌──────────────────┐
 patches → │ Context Encoder  │→ (bs, C*Q, D)
           └──────────────────┘         │
                                        ▼
                                 ┌────────────┐        predictions
                                 │ Predictor  │────────→ (bs, C, M, D)
                                 └────────────┘              │
                                                             │
           ┌──────────────────┐                              │   L1 loss
 patches → │ Target Encoder   │→ (bs, C*N, D) ── gather ─────┤
           │ (EMA, no grad)   │                              │
           └──────────────────┘                              ▼
```

---

## Module summary

| File | Purpose |
| --- | --- |
| `configs/tokenizer_config.py` | `TokenizerConfig` dataclass |
| `configs/encoder_config.py` | `EncoderConfig` dataclass |
| `configs/predictor_config.py` | `PredictorConfig` dataclass |
| `configs/presets.py` | `dev_preset()` and `final_preset()` factory functions returning consistent config triples |
| `pos_encoding.py` | `get_1d_sincos_pos_embed`, `get_2d_sincos_pos_embed` (non-learnable) |
| `tokenizer.py` | `ECGTokenizer` — FFN patch tokenizer, also exposes `patchify(signal)` |
| `modules/attention.py` | `MultiHeadAttention` with `use_flash` flag |
| `modules/transformer.py` | `TransformerBlock` (Pre-LN, shared by encoder and predictor) |
| `encoder.py` | `ECGEncoder` — used as both context and target encoder |
| `predictor.py` | `ECGPredictor` — per-lead masked prediction |

---

## Config system

One dataclass per component, plus named presets in `configs/presets.py`:

```python
from src.configs import dev_preset, final_preset

tok_cfg, enc_cfg, pred_cfg = dev_preset()    # small, for debugging/ablations
tok_cfg, enc_cfg, pred_cfg = final_preset()  # paper-scale, for full runs
```

**Consistency constraints** enforced by `_check_consistency` inside the preset factories:

- `tokenizer.embed_dim == encoder.embed_dim` (token output dim feeds the encoder)
- `encoder.embed_dim == predictor.encoder_embed_dim` (predictor's input projection dim)
- `tokenizer.patch_size == encoder.patch_size`
- `encoder.num_patches == predictor.num_patches`
- `encoder.num_leads == predictor.num_leads`
- `embed_dim % num_heads == 0` in both encoder and predictor

If you hand-assemble configs (not via a preset), you are responsible for these invariants.

### Dev vs final config

| | Encoder (dev / final) | Predictor (dev / final) |
| --- | --- | --- |
| `embed_dim` | 384 / 768 | 192 / 384 |
| `depth` | 6 / 12 | 3 / 6 |
| `num_heads` | 8 / 16 | 6 / 12 |
| Approx. params | ~22M / ~86M | ~5M / ~22M |

Use `dev_preset()` until the training loop is working end-to-end, then swap in `final_preset()` for the actual pretraining run.

---

## Input / output shapes

### `ECGTokenizer`

- `patchify(signal)` — `(bs, C, T)` → `(bs, C, N, patch_size)`, where `N = T // patch_size`. Raises if `T` is not divisible by `patch_size`.
- `forward(patches)` — `(bs, C, N, patch_size)` → `(bs, C, N, embed_dim)`.

### `ECGEncoder`

- `forward(patches, visible_indices=None, return_attn=False)`
  - `patches`: `(bs, C, N, patch_size)`
  - `visible_indices`: `LongTensor (Q,)` of temporal indices ∈ [0, N), or `None` for no masking.
  - Returns `(bs, C*num_visible, embed_dim)` where `num_visible = Q` if `visible_indices` is given, else `N`.
  - If `return_attn=True`, also returns `list[Tensor(bs, H, S, S)]` (one per block). Requires `use_flash=False`.
- `forward_all(patches)` — convenience for the target encoder and inference. Equivalent to `forward(patches, visible_indices=None)`.

### `ECGPredictor`

- `forward(context_tokens, visible_indices, masked_indices, return_attn=False)`
  - `context_tokens`: `(bs, C*Q, encoder_embed_dim)` — the context encoder's output.
  - `visible_indices`: `LongTensor (Q,)` of temporal indices that were visible.
  - `masked_indices`: `LongTensor (M,)` of temporal indices that were masked; `M = N - Q`.
  - Returns `(bs, C, M, encoder_embed_dim)` — predictions at masked positions, already projected back to the encoder's embedding dim so they can be compared directly with target encoder outputs.

Masking is **synchronized across leads**: the same `visible_indices` and `masked_indices` apply to every lead in the batch.

---

## Training-loop integration guide

Below is pseudo-code showing how to wire these modules into a JEPA training step. The training loop author should own masking, EMA, and loss.

```python
import torch
import torch.nn.functional as F
from src.configs import dev_preset
from src.tokenizer import ECGTokenizer
from src.encoder import ECGEncoder
from src.predictor import ECGPredictor

tok_cfg, enc_cfg, pred_cfg = dev_preset()

# Each encoder owns its own tokenizer. If you prefer a single shared tokenizer,
# construct one and pass the same instance to both — but remember that EMA will
# then need to skip the tokenizer parameters on the target side (or accept that
# the tokenizer is effectively "tied" between student and teacher).
tokenizer_ctx = ECGTokenizer(tok_cfg)
tokenizer_tgt = ECGTokenizer(tok_cfg)

context_encoder = ECGEncoder(enc_cfg, tokenizer_ctx)
target_encoder  = ECGEncoder(enc_cfg, tokenizer_tgt)
predictor       = ECGPredictor(pred_cfg)

# Initialize target encoder as a copy of the context encoder.
target_encoder.load_state_dict(context_encoder.state_dict())
for p in target_encoder.parameters():
    p.requires_grad_(False)

def random_mask_indices(num_patches: int, mask_ratio: float, device):
    """Synchronized random masking: one (visible, masked) split per batch,
    applied to every lead and every sample in the batch."""
    num_mask = int(round(num_patches * mask_ratio))
    perm = torch.randperm(num_patches, device=device)
    masked = perm[:num_mask].sort().values
    visible = perm[num_mask:].sort().values
    return visible, masked

def training_step(signal):                        # signal: (bs, C, T)
    patches = tokenizer_ctx.patchify(signal)      # (bs, C, N, patch_size)

    mask_ratio = 0.6 + 0.1 * torch.rand(1).item() # uniform in [0.6, 0.7)
    vis_idx, msk_idx = random_mask_indices(
        enc_cfg.num_patches, mask_ratio, device=signal.device
    )

    # Student / context path.
    ctx_tokens = context_encoder(patches, visible_indices=vis_idx)
    # Predictor reshapes each lead independently and appends mask tokens.
    preds = predictor(ctx_tokens, vis_idx, msk_idx)   # (bs, C, M, D)

    # Teacher / target path — no gradients, LayerNorm stabilization is common.
    with torch.no_grad():
        tgt_all = target_encoder.forward_all(patches) # (bs, C*N, D)
        tgt_all = F.layer_norm(tgt_all, (tgt_all.size(-1),))
        tgt_all = tgt_all.reshape(
            signal.size(0), enc_cfg.num_leads, enc_cfg.num_patches, -1
        )
        targets = tgt_all.index_select(dim=2, index=msk_idx)  # (bs, C, M, D)

    loss = (preds - targets).abs().mean()
    return loss

# EMA update (call after optimizer.step())
@torch.no_grad()
def ema_update(momentum: float):
    for p_tgt, p_ctx in zip(target_encoder.parameters(), context_encoder.parameters()):
        p_tgt.data.mul_(momentum).add_(p_ctx.data, alpha=1.0 - momentum)
```

### Notes on the above

- **Mask ratio** — the paper sweeps uniformly in `(0.6, 0.7)` per batch with random masking. Fix or schedule as you prefer.
- **Target LayerNorm** — applying `layer_norm` to the teacher output (across the D axis) before the loss is a standard stability trick borrowed from the reference implementation; drop it if you want strict paper fidelity.
- **Loss** — the paper uses L1 (`abs().mean()`), averaged over masked positions and leads. You may also try L2; both are common.
- **EMA momentum** — the paper uses a schedule from `ema_0 = 0.996` at step 0 to `ema_1 = 1.0` at the final step (see design backlog for the exact formula). Keep the target encoder in `.eval()` if it contains dropout layers.
- **Mixed precision / bf16** — Flash Attention requires fp16 or bf16 inputs. Wrap the forward passes in `torch.autocast` or cast inputs manually.

---

## Inference guide

```python
context_encoder.eval()
with torch.no_grad():
    patches = tokenizer_ctx.patchify(signal)           # (bs, C, N, patch_size)
    tokens = context_encoder.forward_all(patches)      # (bs, C*N, D)
    ecg_embedding = tokens.mean(dim=1)                 # (bs, D)
```

Use `ecg_embedding` as the input to a downstream classifier / regressor / segmentation head. For segmentation or feature regression, you may instead want the per-patch tensor reshaped to `(bs, C, N, D)` (simply reshape `tokens` before pooling).

---

## Flash attention vs standard attention

Controlled per model by `use_flash` on `EncoderConfig` and `PredictorConfig`:

- `use_flash=True` (default): uses `flash_attn.flash_attn_func`. Fast and memory-efficient. **Cannot** return attention weights — passing `return_attn=True` will raise.
- `use_flash=False`: manual scaled dot-product. Slower, more memory. Supports `return_attn=True` in both encoder and predictor forwards, which returns a list of per-block attention tensors for downstream analysis.

The flag is fixed at construction time. To inspect attention maps, build a **separate** encoder with `use_flash=False` and load pretrained weights into it — the parameter shapes are identical between backends.

Flash Attention requires fp16/bf16 tensors; make sure your autocast context is active when calling a flash-enabled model.

---

## Known gotchas

1. **`patch_size` must divide the raw signal length `T`.** `patchify` raises otherwise. Trim or pad the signal upstream in your dataloader.
2. **Positional embeddings assume fixed `N` and `C`.** They are precomputed at construction time; changing `num_patches` or `num_leads` requires rebuilding the model.
3. **Predictor takes explicit `visible_indices` / `masked_indices`, not a boolean mask.** Keeps gather operations simple and unambiguous.
4. **Masking is synchronized across leads.** All leads in a batch share the same visible/masked temporal indices. If you want per-lead masking, it requires changes to both encoder and predictor.
5. **Target encoder must be kept out of the optimizer.** Either via `requires_grad_(False)` or by passing only context encoder parameters into the optimizer. EMA updates must happen under `torch.no_grad()`.
6. **Tokenizer ownership.** Each encoder instantiated here owns its own tokenizer. If you prefer shared tokenizer weights between student and teacher, pass the same `ECGTokenizer` instance to both; then the tokenizer is not EMA-updated (its weights remain those of the student). Either approach is defensible — pick one and document it in your training script.
7. **Attention maps across batches.** When `use_flash=False` and `return_attn=True`, returned attention tensors have shape `(bs, num_heads, seq_len, seq_len)` for the encoder and `(bs*num_leads, num_heads, seq_len, seq_len)` for the predictor (because the predictor reshapes leads into the batch axis).

---

## Extension points

- **CroPA.** Add a masked-attention variant in `modules/attention.py` and expose a `use_cropa` flag on `EncoderConfig`. The encoder would need to build the CroPA mask from `(num_leads, num_patches)` and pass it through transformer blocks.
- **Multi-block masking.** Replace `random_mask_indices` in the training loop with a sampler that picks overlapping consecutive blocks. No model-side changes needed.
- **Different tokenizers.** `ECGEncoder.__init__` takes any `nn.Module` that exposes `embed_dim`, `patch_size`, and `forward(patches)` with the `(bs, C, N, patch_size)` → `(bs, C, N, D)` contract. Swap in a linear tokenizer, deeper FFN, or a CNN as needed.
- **Variable lead counts.** The transformer itself handles variable sequence lengths. The constraints are (a) positional embeddings are sized for `num_leads` at construction and (b) lead-count changes after training require handling in the pooling and downstream heads.

---

## Verifying the build (smoke test)

A minimal check that does not require flash-attn; paste into a scratch file:

```python
import torch
from src.configs import dev_preset
from src.tokenizer import ECGTokenizer
from src.encoder import ECGEncoder
from src.predictor import ECGPredictor

tok_cfg, enc_cfg, pred_cfg = dev_preset()
enc_cfg.use_flash = False       # allow running on CPU without flash-attn
pred_cfg.use_flash = False

tokenizer = ECGTokenizer(tok_cfg)
encoder   = ECGEncoder(enc_cfg, tokenizer)
predictor = ECGPredictor(pred_cfg)

bs, C, T = 2, enc_cfg.num_leads, enc_cfg.num_patches * enc_cfg.patch_size
signal = torch.randn(bs, C, T)
patches = tokenizer.patchify(signal)                  # (2, 12, 50, 50)

N = enc_cfg.num_patches
vis = torch.arange(0, N, 2)                           # Q=25 visible
msk = torch.arange(1, N, 2)                           # M=25 masked

ctx_tokens = encoder(patches, visible_indices=vis)    # (2, 12*25, 384)
preds = predictor(ctx_tokens, vis, msk)               # (2, 12, 25, 384)
all_tokens = encoder.forward_all(patches)             # (2, 12*50, 384)

print(ctx_tokens.shape, preds.shape, all_tokens.shape)
```

Expected output:
```
torch.Size([2, 300, 384]) torch.Size([2, 12, 25, 384]) torch.Size([2, 600, 384])
```
