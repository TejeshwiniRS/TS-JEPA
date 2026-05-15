# EEG transfer learning: ECG-JEPA → BCIC-IV 2a

Tests whether ECG-pretrained JEPA representations transfer to EEG, by fine-tuning three pretrained checkpoints on the BCIC-IV 2a motor imagery benchmark and comparing to **EEG-FM-Bench Table 2** (arXiv 2508.17742).

## Layout

```
eeg_transfer/
├── data/
│   ├── bcic-2a/                  # raw .mat files (A0[1-9]T.mat)
│   └── processed/                # produced by preprocessing
├── preprocessing/
│   ├── README.md                 # protocol details + run instructions
│   └── preprocess_bcic2a.py      # single entry-point script
├── shared/                       # helpers reused by all three notebooks
│   ├── data.py                   # BCICDataset
│   ├── metrics.py                # bacc / wf1 / kappa via sklearn
│   └── train.py                  # fit / evaluate / scheduler helpers
├── notebooks/
│   ├── 01_dev_preset_finetune.ipynb       # src/ arch, embed_dim 384, 6 layers
│   ├── 02_final_preset_finetune.ipynb     # src/ arch, embed_dim 768, 12 layers
│   └── 03_official_ecg_jepa_finetune.ipynb # ecg_jepa/ arch, paper checkpoint
└── results/
    ├── dev_preset/{metrics.json, curves.png, best_model.pt}
    ├── final_preset/{...}
    └── official/{...}
```

## How to run

1. **Preprocess** (one-time, ~30 s):
   ```bash
   /home/aimakeradmin/shady-env/bin/python3 eeg_transfer/preprocessing/preprocess_bcic2a.py
   ```
   Produces `eeg_transfer/data/processed/{train,val,test}.npz` plus `meta.json` and `norm_stats.npz`.

2. **Run the three notebooks** (each ~3-10 min on a GPU for 30 epochs):
   - `notebooks/01_dev_preset_finetune.ipynb`
   - `notebooks/02_final_preset_finetune.ipynb`
   - `notebooks/03_official_ecg_jepa_finetune.ipynb`

   Each notebook is self-contained and follows the same 9-step structure:
   imports → load data → build encoder + load checkpoint → classifier head →
   optimizer + scheduler → train → test → save → comparison vs Table 2.

## Comparison target

EEG-FM-Bench Table 2 BCIC-2a (full-parameter single-task fine-tuning, average pooling head):

| Model | B-Acc |
|---|---|
| EEGPT (best) | 44.07 |
| CsBrain | 36.23 |
| BENDR | 35.21 |
| CBraMod | 33.71 |
| REVE | 32.73 |
| LaBraM | 28.50 |
| BIOT | 22.08 |
| chance | 25.00 |

All seven baselines were pretrained on EEG; we test whether ECG-pretrained JEPAs land in the same range.

## Adapter strategy (no edits to `src/` or `ecg_jepa/`)

Both architectures are channel-agnostic where it counts:
- **Tokenizer** (`nn.Linear(patch_size, embed_dim)`) is per-patch — transfers 1:1.
- **Transformer blocks** are pure self-attention — shape-agnostic.
- **2D sincos positional embeddings** are parameter-free — regenerated for the new (22, 20) shape at `__init__`.

For the `src/` checkpoints: the `pos_embed` buffer is non-persistent, so it isn't even in the state dict — `load_state_dict(state, strict=False)` returns empty `missing` and `unexpected` lists.

For the official ECG-JEPA checkpoint: `pos_embed` is a frozen `nn.Parameter` of shape `(8*50, 768)`. We pop it from the state dict before loading; `__init__` already populated the new `(22*20, 768)` parameter with sincos. We also wrap the encoder to bypass a hardcoded `assert x.shape[2] == 2500` in `representation()`.

## Protocol — matches EEG-FM-Bench Appendix B

- AdamW, weight_decay 0.01
- Linear warmup 3 epochs → cosine anneal over 27 → 30 total
- Differential LR: backbone 1e-4, head 1e-3 (1/10 ratio)
- Batch 128, grad clip 1.0
- Subject split: train = 1-5, val = 6-7, test = 8-9
- Best model selected on val balanced accuracy

## Limitation

Only T (training-session) `.mat` files are present in `data/bcic-2a/`. EEG-FM-Bench used T+E and reports 2784/1152/1152 trials; we have ~1337/490/501. Direct accuracy comparison remains valid — only the absolute training-data scale differs.
