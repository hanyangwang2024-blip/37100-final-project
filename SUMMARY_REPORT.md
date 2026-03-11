# Diffusion Timesteps and Prediction Targets: A Two-Knob Study on MNIST

DATA 37100 — Final Project Summary Report

---

## 1. Question & Motivation

**Core question:** How do diffusion timestep count (T) and prediction target (eps vs x0) jointly affect training loss dynamics and sample quality on MNIST?

Denoising Diffusion Probabilistic Models (DDPMs) have two fundamental design choices that interact in non-obvious ways: the number of diffusion timesteps T (controlling the granularity of the noise schedule) and the prediction target (whether the network predicts the added noise ε or the clean image x₀). While both are well-studied independently, their *interaction* under limited training budgets is less understood.

This project investigates this interaction through a controlled 3×2 grid experiment, complemented by a Transformer character-LM baseline to satisfy the two-model-family requirement.

---

## 2. Methods

### Models

| Model | Family | Architecture | Key config |
|-------|--------|-------------|-----------|
| DDPM | Diffusion | UNet, base_ch=64, time_emb=128 | Linear β schedule (0.0001→0.02) |
| Tiny GPT | Transformer | d_model=128, 4 heads, 2 layers | Char-level LM, 37-char vocab |

### Dataset

**MNIST** (60,000 training images, 28×28 grayscale). Chosen for fast iteration (~45s per diffusion run on Apple MPS) and well-understood visual failure modes.

### Controlled Experiment Design

**Knob 1:** T ∈ {100, 200, 400} — diffusion timesteps
**Knob 2:** target ∈ {eps, x0} — prediction parameterisation

This yields a 3×2 grid of **6 runs**, each training for 1 epoch (468 gradient steps) with seed=42 on Apple MPS.

### Transformer Baseline

600 training steps on a 6-sentence corpus about transformers. Config: temp=0.8, top_p=0.95, eval every 50 steps (13 data points). This confirms pipeline correctness for a second model family but is not the focus of the controlled experiment.

---

## 3. Results

### 3.1 Training Loss Dynamics

The central finding is that **eps and x0 respond to T in opposite directions:**

| T | eps final loss | x0 final loss | eps runtime | x0 runtime |
|---|---------------|---------------|-------------|-------------|
| 100 | 0.1042 | **0.0258** | 42.7 s | 42.1 s |
| 200 | 0.0886 | 0.0503 | 45.8 s | 47.3 s |
| 400 | **0.0594** | 0.1114 | 53.4 s | 51.6 s |

- **eps loss decreases with larger T** (0.104 → 0.059). With more timesteps, each individual noise step is smaller, making per-step noise prediction smoother and easier to learn.
- **x0 loss increases with larger T** (0.026 → 0.111). With more timesteps, the model is asked to reconstruct clean images from inputs at higher noise levels (larger t values), where x_t ≈ pure Gaussian noise.

This inversion is visible in the training curve plots (see `training_curves.png` in the notebook § 4) and the final-loss heatmap (`final_loss_heatmap.png`, notebook § 4).

### 3.2 Sample Quality

Sample grids (see `sample_grid_all_runs.png`, notebook § 5, 8×8 per run) show:

- **T=100, eps:** Coarse, noisy shapes — the network has too few denoising steps and each step requires a large correction.
- **T=100, x0:** Faint digit outlines visible — with low noise, x₀ still carries substantial signal.
- **T=200:** Intermediate quality for both targets; this is the "sweet spot" for 1-epoch training.
- **T=400, eps:** Best eps samples — many small steps, each prediction is smooth.
- **T=400, x0:** Blurry/degraded — the prediction task is too hard from heavily corrupted inputs.

All samples are partially formed (blob-like digits) due to the 1-epoch constraint, which is a known confound acknowledged in the limitations.

### 3.3 Runtime

Training time scales approximately linearly with T: 40–48s (T=100) → 50s (T=400), a ~25% increase. The eps vs x0 choice does not significantly affect runtime. Total wall-clock time for all 6 diffusion runs: ~4.5 minutes.

### 3.4 Transformer Baseline

- Train loss: 3.426 → 0.054 in 600 steps (sharp descent, 13 eval points).
- Val loss: 3.426 → 0.055 (nearly identical to train throughout, gap < 0.005).
- Sample output: garbled fragments ("Transformers replace rge + rTresitanlad stabilize training.").
- Runtime: 24.8s on Apple MPS.

The near-identical train/val loss reveals pure memorisation on the tiny 330-character corpus. With 600 steps (vs the initial 300), loss drops further to 0.054 but sample quality does not improve — the model has fully memorised the data but cannot compose novel sequences. The training curve (see `transformer_loss_curve.png`) shows smooth exponential decay with no train/val divergence.

---

## 4. Failure Modes & Limitations

### Failure Mode 1: T=400 + x0 — Loss Inversion

**Evidence:** x0 loss at T=400 (0.111) is *worse* than at T=100 (0.026) — the trend inverts relative to eps.

**Mechanism:** The linear noise schedule (β₁=0.0001 → β₂=0.02) with T=400 means that at large timestep values t, x_t ≈ pure Gaussian noise. Predicting the clean image x₀ from near-pure noise is an ill-posed regression problem. The network cannot learn a reliable mapping in 468 gradient steps.

**Implication:** The x0 parameterisation is inherently more sensitive to T than eps. Practitioners using x0 with large T should either (a) use a cosine schedule to keep signal-to-noise ratio higher, or (b) train for significantly more epochs.

### Failure Mode 2: Under-training Artifacts (All Runs)

**Evidence:** All 6 sample grids show blob-like shapes rather than sharp digits.

**Cause:** 1 epoch = 468 steps over 60,000 images. Each image is seen only once. The UNet has not learned fine-grained denoising at all noise levels.

**Mitigation:** Running 5–10 epochs typically produces recognisable MNIST digits. The 1-epoch constraint was chosen for speed/scope but must be acknowledged as a confound.

### Failure Mode 3: Transformer Memorisation

**Evidence:** Train ≈ val loss throughout 600 steps (gap < 0.005); sample output is garbled mixed fragments even at loss=0.054.

**Cause:** The training corpus is only ~330 characters. The model memorises token co-occurrences but cannot compose unseen sequences. Temperature-based sampling exposes the lack of generalisation.

### Limitations

- All results are from 1-epoch training; the T × target interaction may differ qualitatively at convergence.
- The linear β schedule was held fixed; a cosine schedule could shift the optimal T for x0.
- The Transformer experiment serves as a pipeline check, not a deep behavioural study (corpus too small).

---

## 5. Conclusions

1. **eps and x0 respond to T in opposite directions** — this is the key finding. eps benefits from larger T (smoother per-step task), while x0 degrades (harder regression from noisier inputs).

2. **T=200 is the sweet spot for 1-epoch training**, balancing loss magnitude for both targets.

3. **Low per-step loss does not guarantee good samples.** T=100 x0 achieves the lowest raw loss (0.026) but does not produce the best visual samples — accumulated error over too few denoising steps still degrades quality.

4. **Runtime scales linearly with T** (~25% increase from T=100 to T=400) without proportional quality gains at 1 epoch.

5. **The x0 parameterisation is more fragile** than eps under limited training: it requires more careful schedule tuning or more training epochs when T is large.

6. **Next step:** Rerun at 5 epochs to separate T×target effects from under-training confounds; test a cosine noise schedule; evaluate Transformer decoding knobs on a larger corpus.

---

## Reproducibility

All experiments can be reproduced from the repo root with the commands listed in `README.md`. Seed is fixed at 42 for all diffusion runs. The transformer uses seed 0 (default). Total wall-clock time for all experiments: ~7 minutes on Apple MPS (M-series). Pre-run outputs are saved under `./untrack/outputs/final/`. The analysis notebook (`final/analysis/final_project_analysis_template.ipynb`) contains all cells pre-executed with embedded figures.

---
