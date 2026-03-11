# DATA 37100 — Final Project

**Question:** How do diffusion timestep count (T) and prediction target (eps vs x0) jointly affect training dynamics and sample quality on MNIST?

**Models:** Diffusion (DDPM) + Transformer (character-level LM)
**Dataset:** MNIST (28×28 grayscale)

---

## Setup

```bash
pip install torch torchvision matplotlib pandas numpy
```

**Dependencies:** Python 3.12, PyTorch 2.4.1, torchvision 0.19.1, matplotlib, pandas, numpy

---

## Run Commands (from repo root)

### 1. Download data (one-time)

```bash
python -c "from torchvision import datasets; datasets.MNIST(root='./data/bigdata', download=True)"
```

### 2. Diffusion baseline

```bash
python final/starter/src/diffusion_baseline.py \
    --dataset mnist --epochs 1 --T 200 --data-dir ./data/bigdata/MNIST
```

**Runtime:** ~0.8 min on Apple MPS

### 3. Diffusion two-knob grid experiment (6 runs)

```bash
python final/starter/src/diffusion_baseline.py \
    --dataset mnist --epochs 1 \
    --grid "T=100,200,400;target=eps,x0" --data-dir ./data/bigdata/MNIST
```

**Runtime:** ~5 min total (6 runs) on Apple MPS

### 4. Transformer baseline

```bash
python final/starter/src/transformer_baseline.py --steps 600 --sample-len 300 --eval-every 50
```

**Runtime:** ~0.2 min on Apple MPS

### 5. Visualize sample grids

```bash
python final/tools/visualize_samples.py \
    --results-csv ./final/untrack/outputs/diffusion/results.csv
```

---

## Expected Outputs

```
final/untrack/outputs/
├── diffusion/
│   ├── results.csv                        # 6-run manifest
│   ├── ds-mnist_T-{100,200,400}_target-{eps,x0}_b2-0.02_ch-64/
│   │   ├── run_args.json
│   │   ├── summary.json
│   │   └── train_log.csv
│   ├── training_curves.png
│   ├── final_loss_heatmap.png
│   ├── runtime_comparison.png
│   └── sample_grid_all_runs.png
└── transformer/
    └── bsz-64_steps-600_dm-128_nh-4_nl-2_temp-0.8_topP-0.95/
        ├── run_args.json, summary.json, train_log.csv
        └── sample.txt
```

---

## Hardware Assumptions

- **Primary:** Apple MPS (M-series Mac)
- **Also works on:** CPU (auto-detected), CUDA GPU
- All runs complete in under 1 minute each. Total wall-clock: **~7 minutes**.

## Seed

All diffusion runs use `--seed 42` for reproducibility.

## Analysis

Open `analysis/final_project_analysis_template.ipynb` — all cells are pre-executed with outputs.
