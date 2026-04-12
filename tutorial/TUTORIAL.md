# masertrack Tutorial

A step-by-step walkthrough of the masertrack pipeline using synthetic data with a known answer. After completing this tutorial, you should understand how to run the pipeline on your own data, interpret the outputs, and iterate to improve results.

## Prerequisites

Install the required packages:

    pip3 install numpy pandas scipy matplotlib plotly

## Tutorial data

The `tutorial/` directory contains synthetic VLBI maser observations with a **known parallax** so you can verify the pipeline recovers the correct answer.

**Source parameters (ground truth):**

| Parameter | Value |
|-----------|-------|
| RA | 06:10:00.000 |
| Dec | +20:30:00.000 |
| True parallax | 0.500 mas |
| True distance | 2.000 kpc |
| True PM (RA) | −2.00 mas/yr |
| True PM (Dec) | +1.00 mas/yr |

**Observations:** 6 epochs over 13 months with a 4-station array (beam 1.8 × 0.8 mas, channel spacing 0.42 km/s). Four maser features at different velocities, plus 2 sidelobe artifacts per epoch.

| Feature | V_LSR (km/s) | Spots | Flux (Jy) | Epochs | Expected grade |
|---------|-------------|-------|-----------|--------|---------------|
| F1 | −35.0 | 5 | 10 | all 6 | A |
| F2 | −42.0 | 4 | 5 | all 6 | A |
| F3 | −28.0 | 5 | 3 | all 6 | A |
| F4 | −50.0 | 3 | 2 | 4 of 6 | B |

The `tutorial/expected_output/` directory contains the correct pipeline output for comparison.

---

## Part 1: Spot identification (`masertrack_identify.py`)

### What this step does

Each epoch has a list of Gaussian-fitted emission peaks (spots) from AIPS MFIT/SAD. `masertrack_identify.py` cleans this list and groups spots into maser features: spatially and spectrally coherent emission groups representing individual maser clouds.

### Run it

Process all 6 epochs. The `--n-stations 4` flag sets the sidelobe threshold (1/N scaling, Steinberg 1972):

    for f in tutorial/input/e*_normal.txt; do
      python masertrack_identify.py "$f" --n-stations 4 \
        --outdir tutorial/output/identify --no-plots
    done

To see diagnostic plots, remove `--no-plots` (or add `--show-plots` to open windows). For a guided first run with step-by-step confirmation:

    python masertrack_identify.py tutorial/input/e01_normal.txt \
      --guided --n-stations 4

### What to expect

For each epoch, the console shows the 6-step pipeline:

    Step 1 - Parse: 19 spots, V = [-50.4, -27.2] km/s
    Step 2 - Deduplicate: 19 -> 19 (0 merged)
    Step 3 - Sidelobe flagging: 2 spots flagged
    Step 4 - Feature grouping: 4 features from 17 spots
    Step 5 - Multi-peak split: 0 split(s) -> 4 features
    Step 6 - Summarize: 4 features, 4 with >= 3 channels

The 2 flagged spots are the planted sidelobe artifacts (faint spots with distorted beams near the brightest feature). Epochs e03 and e06 show only 3 features because F4 is absent.

### Output files

Two text files per epoch:

- **`e01_normal_features.txt`** — one row per feature with summary properties (position, velocity, flux, errors)
- **`e01_normal_spots.txt`** — one row per spot with its feature assignment (`feature_id` column)

The feature catalog is used by `masertrack_match.py` for cross-epoch matching. The spot table provides the individual channel positions used for parallax fitting.

### Check your results

Compare `tutorial/output/identify/e01_normal_features.txt` to `tutorial/expected_output/identify/e01_normal_features.txt`. You should see 4 features with velocities near −50, −42, −35, and −28 km/s. The positions and fluxes should match within rounding.

### Iterating

If sidelobe flagging is too aggressive or too permissive:

1. Inspect the Step 3 output — it lists each flagged spot with its ID and reason
2. Create an exclude file (`echo "42" > exclude.txt`) to remove additional spots
3. Create an include file to un-flag incorrectly flagged spots
4. Re-run: `python masertrack_identify.py input.txt --exclude exclude.txt`

---

## Part 2: Cross-epoch matching (`masertrack_match.py`)

### What this step does

`masertrack_match.py` tracks the same physical maser feature across multiple epochs. It matches features by position and velocity, tracks velocity drift, assigns quality grades (A through F), and produces channel-level position tables for parallax fitting.

### Input: epoch table

The epoch table (`tutorial/epoch_table.txt`) lists each observation:

    # code  date        mjd    dec_year    n_stations
      e01   2022-03-15  59653  2022.2016   4
      e02   2022-06-01  59731  2022.4152   4
      ...

The code must match the filenames: `e01` → expects `e01_normal_features.txt` and `e01_normal_spots.txt` in the input directory.

### Run it

    python masertrack_match.py tutorial/epoch_table.txt \
      --input-dir tutorial/output/identify \
      --outdir tutorial/output/match --no-plots

### What to expect

    Step 1 - 6 epochs (0 flagged, 6 unflagged for grading)
    Step 2 - Alignment (NORMAL: 6 epochs, no shift)
    Step 3 - Matching
      4 groups, 4 with >=3 epochs
        Grade A: 3
        Grade B: 1

The group summary table shows each matched group:

    GID   V_LSR  Nep  Nch  Gr     X     Y   Pattern
      1   -50.0    4    3   B  -4.0  -5.1   12*45*
      2   -42.2    6    4   A  -8.1   6.0   123456
      3   -35.0    6    5   A   4.9   3.0   123456
      4   -28.0    6    5   A   1.9 -10.0   123456

The pattern column shows which epochs are present (`12*45*` means epochs 1, 2, 4, 5 are matched; `*` = missing). G1 is Grade B because it appears in only 4 of 6 unflagged epochs.

### Output files

| File | Content |
|------|---------|
| `match_normal_groups.txt` | Group catalog (one row per matched group) |
| `match_normal_spots_tracked.txt` | Channel positions across epochs (parallax fitting input) |
| `match_normal_features_tracked.txt` | Feature positions across epochs (comparison) |
| `match_normal_id_mapping.txt` | Cross-reference: group ID ↔ per-epoch feature ID |
| `match_corrections_template.txt` | Editable template for manual corrections |

### Grading system

| Grade | Criterion | Meaning |
|-------|-----------|---------|
| A | Present in all unflagged epochs | Best for parallax |
| B | ≥ 2/3 of unflagged epochs, ≥ 4 total | Good |
| C | ≥ 4 epochs | Usable with caution |
| D | 3 epochs | Marginal (5 params from 6 data points) |
| E | 2 epochs | Insufficient for parallax |
| F | 1 epoch | Not matched |

### Iterating with corrections

If a match is incorrect (wrong feature grabbed at one epoch), use the corrections file:

1. Open `match_corrections_template.txt` — all groups and flagged epochs are listed
2. Uncomment a line to apply a correction, e.g.:
   `GROUP  G1  e04  EXCLUDE` removes epoch e04 from group G1
3. Re-run with `--corrections match_corrections_template.txt`

---

## Part 3: Parallax fitting (`masertrack_fit.py`)

### What this step does

`masertrack_fit.py` fits the tracked channel positions for trigonometric parallax (π) and proper motion (μ). It runs individual, group, and feature fits, computes error floors, runs bootstrap uncertainty estimation, and detects outlier groups.

### Run it

    python masertrack_fit.py \
      tutorial/output/match/match_normal_spots_tracked.txt \
      --ra 06:10:00.000 --dec +20:30:00.000 \
      --outdir tutorial/output/fit --no-plots

For production runs, omit `--no-plots` to get diagnostic PNGs and interactive HTML. Add `--quick` to skip bootstrap (faster but no empirical uncertainties).

### What to expect

The key result is the group fit (Step 2) — common parallax across all channels in each group:

    Step 2 — Group fitting (common π)
      G1: 3 ch, pi=0.474 ± 0.049 mas
      G2: 3 ch, pi=0.496 ± 0.016 mas
      G3: 5 ch, pi=0.503 ± 0.019 mas
      G4: 5 ch, pi=0.500 ± 0.034 mas

And the combined result:

    Summary — normal PR:
          Method    Grade   pi (mas)   sigma_pi   D_mode
        wt. mean      A+B     0.4985     0.0126      2.0

The true parallax is 0.500 mas (D = 2.000 kpc). The pipeline recovers this within the uncertainties.

### Understanding the fitting steps

| Step | What it does | Why |
|------|-------------|-----|
| 1. Individual | Fit π + μ per channel independently | Diagnostic: all channels in a group should give consistent π |
| 2. Group | Common π across channels, independent μ per channel | **Primary result** — most reliable (Burns+2015) |
| 3. Bootstrap | 10,000 epoch resamples | Empirical uncertainty (replaces formal σ) |
| 4. Acceleration | BIC test for quadratic motion | Detects non-linear proper motion |
| 5. Feature | Flux-weighted centroids | Comparison only (unreliable for variable masers) |
| 5b. Cauchy | Outlier-tolerant reweighting | Downweights outlier data points |
| 6. Distance | 1/π with optional priors | Distance estimate |
| 7. Combined | Shared π across grade sets (A, A+B, A+B+C) | Best overall parallax |
| 8. Epoch outliers | Flag >3σ residual epochs | Catches systematic errors |
| 9. Chauvenet | Flag outlier groups | Catches groups at different distances |

### Output files

| File | Content |
|------|---------|
| `fit_final_parallax.txt` | Best parallax per mode with per-group breakdown |
| `fit_normal_results.txt` | Detailed per-group results (π, μ, error floors, χ²) |
| `fit_normal_individual.txt` | Per-channel individual fits |
| `export/pmpar/` | pmpar-format input files for cross-validation |
| `export/bessel/` | BeSSeL-format input files |
| `export/vera/` | VERA pipeline format |
| `export/kadai/` | KaDai format |

### Check your results

Compare `tutorial/output/fit/fit_final_parallax.txt` to `tutorial/expected_output/fit/fit_final_parallax.txt`. The combined A+B parallax should be approximately 0.50 ± 0.01 mas.

---

## Normal vs inverse phase referencing

The tutorial data above uses **normal phase referencing** (normal PR). This section explains the differences when using **inverse phase referencing** (inverse PR), which requires additional alignment steps.

### Normal PR

In normal PR, the quasar provides the phase/rate solutions that are applied to the maser data. The quasar is assumed stationary in the ICRF frame, so the maser positions are directly in a common reference frame across epochs. No alignment is needed — the parallax and proper motion are fitted directly from the maser positions.

The pipeline for normal PR is straightforward: `identify → match → fit`.

### Inverse PR

In inverse PR, the maser provides the phase/rate solutions that are applied to the quasar data. This is used when the quasar is too faint for direct fringe detection — the maser, being brighter, serves as the phase reference. The quasar's apparent position then reflects the inverse of the maser's motion (parallax + proper motion).

The complication: the phase reference (the maser) is itself moving. If the reference spot changes between epochs (e.g., a different physical cloudlet becomes the brightest channel), the coordinate origin shifts. `masertrack_match.py` handles this through three alignment layers applied in Step 2:

1. **Quasar-based shifts** — the measured quasar position encodes the inverse of the maser's motion. The shift at each epoch relative to epoch 0 is `−(Q(i) − Q(0))`. These shifts are specified in the epoch table as additional columns.

2. **Reference spot correction** — if the brightest maser channel used for fringe-finding shifts between epochs (common when the array lacks Doppler tracking), the coordinate origin moves. The code identifies the reference velocity at epoch 0 and locates the same velocity in subsequent epochs, applying the positional difference as a correction.

3. **Normal PR anchor** — when both PR modes are available, the inverse data is shifted into the normal PR coordinate frame. The reference maser position at (0,0) in inverse PR has a known position in normal PR (measured from the phase tracking center). Adding this offset puts both modes in the same frame for direct comparison.

### When to use each mode

- **Normal PR**: standard choice when the quasar is bright enough. Simpler pipeline, fewer systematics.
- **Inverse PR**: necessary when the quasar is faint. More complex alignment, but can sometimes yield tighter parallax fits because the maser signal is stronger and provides better phase solutions.
- **Both modes**: when possible, run both and compare. Consistency between the two modes is a strong validation of the parallax result.

To run both modes, name your input files with both `_normal_` and `_inverse_` suffixes. `masertrack_match.py` auto-detects and processes both, including cross-checking the alignment.

---

## Reading diagnostic plots

When you run without `--no-plots`, the pipeline generates PNG and interactive HTML plots at each stage.

### identify plots

| Plot | What to look for |
|------|-----------------|
| `*_s1.png` | Raw spots — check spatial distribution and spectrum shape |
| `*_s3.png` | Sidelobe flagging — red X marks flagged spots; colored rings show individual criteria. Are the right spots flagged? |
| `*_s6.png` | Feature catalog — are features well-separated? Any blends? |
| `*_summary.png` | 9-panel overview — error comparisons, size distribution, flux distribution |
| `*_interactive.html` | Open in browser — hover for details, click legend to toggle features |

### match plots

| Plot | What to look for |
|------|-----------------|
| `*_diagnostic.png` | Rows 1-2: RA/Dec vs time for A/B groups — should show smooth proper motion. Row 3: residuals — points far from zero indicate mismatches. Red rings = flagged epochs. |
| `*_dashboard.png` | Grade distribution, velocity coverage, velocity drift |
| `*_comparison.png` | Normal vs inverse sky maps (when both modes available) |
| `*_interactive.html` | Toggle individual groups, hover for epoch details |

### fit plots

| Plot | What to look for |
|------|-----------------|
| `G{id}_parallax.png` | 3×3 per group — top: PM vectors; middle: RA/Dec vs time with parallax sinusoid; bottom: residuals. Residuals should scatter randomly around zero. |
| `final_parallax_overview.png` | Matrix with one row per group: parallax curve, bootstrap histogram, distance PDF |
| `final_proper_motions.png` | PM vectors colored by V_LSR with AU-scale axes |
| `pi_vs_vlsr.png` | Per-channel parallax vs velocity — all channels should give consistent π |
| `*_interactive.html` | Per-channel toggling, hover for fit parameters |

---

## Common workflow summary

1. **Run identify** on each epoch → inspect sidelobe flagging → iterate with `--exclude`/`--include` if needed
2. **Run match** → inspect group table and diagnostic plots → edit corrections file → re-run if needed
3. **Run fit** → inspect per-group parallax plots → check consistency across groups and channels
4. **If σ_π/π > 20%**: go back to match and try to improve spot selection (more careful matching, exclude problematic epochs)
5. **Final result**: the combined A+B parallax from `fit_final_parallax.txt`
