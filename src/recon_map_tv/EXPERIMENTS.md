# MAP-TV Experimentation Methodology

---

## Parameter Landscape

| Parameter | Importance | Reason |
|---|---|---|
| `beta` (β) | **Critical** | Single biggest lever — controls regularization strength |
| `n_outer` | **High** | Must be long enough for outer loop to converge |
| `n_inner` | **Medium** | Controls quality of TV prox solve per outer step |
| `tau` / `sigma` | **Low** | Symmetric 0.25/0.25 is solid; rarely worth changing |
| `theta` | **Ignore** | 1.0 is theoretically optimal for standard CP |
| `convergence_tol` | **Ignore** | 1e-4 is fine |
| `background` | **Physics** | Fixed by scanner model, not a tuning parameter |
| `eps` | **Ignore** | Numerical floor only |

---

## Phase 0 — Reference Baseline (run once)

Run plain MLEM first to anchor visual comparison. This is β=0 — noisy but unbiased.

```yaml
# recon_mlem/configs/base_config.yml
iterations: 60   # match n_outer so comparison is fair
```

---

## Phase 1 — β Coarse Sweep

β is the only parameter worth sweeping broadly. Fix everything else at defaults.

```yaml
n_outer:  60
n_inner:  30
tau:      0.25
sigma:    0.25
theta:    1.0
```

| Run | β | Expected behavior |
|---|---|---|
| P1-A | `0.001` | Near-MLEM, barely regularized, noisy |
| P1-B | `0.01` | Slight smoothing, noise reduced |
| P1-C | `0.1` | Moderate — likely sweet spot region |
| P1-D | `1` | Current default — moderate-to-strong |
| P1-E | `10` | Strong — expect some staircase artifacts |
| P1-F | `100` | Over-regularized, blocky/cartoon-like |

**Goal:** identify which order of magnitude gives the best tradeoff (noise vs sharpness of rods).
One or two adjacent runs will show cleanly resolved rods — that is the target decade for Phase 2.

---

## Phase 2 — β Fine Sweep

Use the winning decade from Phase 1. Example below assumes P1-C and P1-D were best (0.1–1).
β response is logarithmic — use log spacing.

| Run | β |
|---|---|
| P2-A | `0.05` |
| P2-B | `0.1` |
| P2-C | `0.2` |
| P2-D | `0.5` |
| P2-E | `1.0` |

---

## Phase 3 — n_inner Sensitivity

Fix best β from Phase 2. Check whether 30 inner iterations is overkill or insufficient.

| Run | β | n_inner |
|---|---|---|
| P3-A | best β | `5` |
| P3-B | best β | `10` |
| P3-C | best β | `20` |
| P3-D | best β | `30` (current default) |
| P3-E | best β | `50` |

**What to look for:** if P3-B and P3-D look identical, n_inner=10 is sufficient (3× faster inner loop).
If P3-C through P3-E all look the same, n_inner=20 is the ceiling — no reason to go higher.

---

## Phase 4 — n_outer / Convergence Check

Fix best β and best n_inner from above.

| Run | β | n_inner | n_outer |
|---|---|---|---|
| P4-A | best | best | `20` |
| P4-B | best | best | `40` |
| P4-C | best | best | `60` (current default) |
| P4-D | best | best | `100` |

Check the convergence diff printed in the terminal log. If it hits `1e-4` at outer iteration 30,
running to 60 is wasted compute. If it never hits `1e-4`, more outer iterations are needed.

---

## Phase 5 — τ/σ Step Sizes (optional)

Only run after Phases 1–4 are complete. Stability constraint: `τ × σ < 0.125`.

| Run | tau | sigma | τσ | Character |
|---|---|---|---|---|
| P5-A | `0.20` | `0.20` | 0.040 | Conservative — slower inner convergence |
| P5-B | `0.25` | `0.25` | 0.063 | Current default |
| P5-C | `0.30` | `0.30` | 0.090 | Aggressive — faster per inner step |
| P5-D | `0.50` | `0.20` | 0.100 | Asymmetric — stronger primal push |

P5-C is the most useful to try — still stable, may converge the inner loop faster,
allowing a reduction in n_inner.

---

## Total Run Count

```
Phase 0:  1 run   (MLEM baseline)
Phase 1:  6 runs  (β coarse)
Phase 2:  5 runs  (β fine)
Phase 3:  5 runs  (n_inner)
Phase 4:  4 runs  (n_outer)
Phase 5:  4 runs  (tau/sigma — optional)
─────────────────────────────────────────
Total:   21–25 runs
```

---

## What to Measure Each Run

Use `view_npz.py` for visualization. Track for every run:

1. **Rod visibility** — can the smallest rods in the Derenzo phantom be resolved?
2. **Background noise** — eyeball variance in a uniform background region
3. **Convergence iteration** — what outer iter did the diff drop below tol? (printed in terminal)
4. **Runtime** — SLURM wall time from the `.out` log

---

## Practical Notes

- Run Phases 1 and 2 first — β alone gives 80% of the quality gain
- Use `save_every: 5` to see the convergence trajectory, not just the final image
- Keep all output `.npz` files — `view_npz.py` already tags filenames with parameter values
- When in doubt, β too low is better than β too high — staircase artifacts from over-regularization
  can look misleadingly clean while erasing real small-scale structure
