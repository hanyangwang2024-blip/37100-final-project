# DATA 37100 — Final Project Proposal (Required)

Length: ~1 page

## 1) Model families (pick at least two)

Check **at least two**:

☐ GAN (DCGAN)
☑ Transformer
☑ Diffusion

**Justification:** Diffusion and Transformer represent two fundamentally different generative paradigms — iterative denoising vs. autoregressive token prediction. Comparing their training dynamics and failure modes under controlled conditions reveals how architectural inductive biases shape generation quality. Diffusion is the primary model for the two-knob experiment (T × target), while the Transformer baseline demonstrates a second model family as required.

---

## 2) Dataset

**MNIST** (Tier 1 approved dataset).

MNIST is chosen because: (1) fast training (~45 seconds per run on Apple MPS), allowing all 6 grid runs + baselines within the 20-hour budget; (2) well-understood failure modes (blurry digits, mode collapse) that are easy to diagnose visually; (3) single-channel 28×28 images minimize computational overhead.

---

## 3) Your core question (one sentence)

**"How do diffusion timestep count (T) and prediction target (eps vs x0) jointly affect training loss dynamics and sample quality on MNIST?"**

---

## 4) Controlled variables (two-knob study)

- **Knob 1:** Diffusion timesteps T ∈ {100, 200, 400}
- **Knob 2:** Prediction target ∈ {eps (predict noise), x0 (predict clean image)}

Planned settings (3 × 2 = **6 runs**, all 1 epoch, seed=42):

| Run | T   | target |
|-----|-----|--------|
| 1   | 100 | eps    |
| 2   | 100 | x0     |
| 3   | 200 | eps    |
| 4   | 200 | x0     |
| 5   | 400 | eps    |
| 6   | 400 | x0     |

---

## 5) Evidence & evaluation plan

**Quantitative:**
- Training loss curves (per-step loss at 50-step intervals for all 6 runs)
- Final-step loss comparison table + heatmap
- Runtime per run (seconds)

**Qualitative:**
- Side-by-side sample grids (8×8 images per run) at the final training step
- Visual comparison of digit sharpness, artifacts, and mode coverage across T and target settings

---

## 6) Risks & fallback plan

| Risk | Mitigation |
|------|-----------|
| 1-epoch training may be too short to show meaningful T × target differences | Increase to 3–5 epochs if initial results are inconclusive |
| x0 target may fail to converge at high T | This is itself an interesting failure mode to document |
| Transformer corpus too small for meaningful decoding analysis | Acknowledge as a limitation; the transformer serves as a second-family baseline, not the primary experiment |

---

**Instructor approval required before proceeding.**
