# masertrack — Detailed Code Breakdown

What the code does, why it does it that way, and the physical reasoning behind each choice.

---

# Part 1: masertrack_identify.py

## File structure

```
Lines     Section
------    -------
1-30      Module docstring (credits, version, description)
32-60     Imports (graceful failure with install instructions)
63-90     Rainbow colormap definition
93-115    Default parameters (DEFAULTS dictionary)
118-140   BeamInfo data class
143-260   MaserTrackIdentify constructor
263-370   Step 1 - Parse
373-430   Step 2 - Deduplicate
433-560   Step 3 - Sidelobe flagging
563-700   Step 4 - Feature grouping (chain-linking)
703-760   Step 5 - Multi-peak splitting
763-890   Step 6 - Summarize
893-920   run() - execute all steps
923-970   save() and fixed-width writer
973-1200  Plotting functions (PNG + HTML)
1203-1280 Click-to-exclude interactive mode
1283-1350 Guided mode
1353-1430 Command-line interface
```

## Default parameters (DEFAULTS dictionary)

All tunable parameters live in one dictionary at the top. The key design principle: **no hard-coded spatial values in mas**. All distances are expressed as multiples of the beam size, and all velocity ranges as multiples of the channel spacing. This is what makes the code array-agnostic.

| Parameter | Default | Source / Reasoning |
|-----------|---------|-------------------|
| Dedup radius | 1.0 × r_geo | Burns used ~1.0 mas; r_geo gives same scale for VERA |
| Sidelobe zone | 0.5–3.0 × bmaj | First sidelobe at 1.0–1.5 × beam (Miyoshi+ 2024), second/third extend to ~3× |
| Sidelobe flux | 1/N + 0.10 | Steinberg (1972) 1/N scaling + margin for post-CLEAN residuals |
| Sidelobe beam tol | 50% | Empirical; sidelobe Gaussian fits are typically >50% distorted |
| Link radius | 1.0 × r_geo | Same as dedup radius for consistency |
| Max vel gap | 2.5 channels | Bridges 1 missing channel; 2+ missing = real break |
| Min dip channels | 2 | Burns' MultiPeakCheck value; 1 would be too sensitive to noise |
| N brightest | 3 | Burns' ThreeStrongest; balances precision vs robustness |

## BeamInfo class

Holds four numbers: `bmaj`, `bmin`, `pa`, `chan_spacing`. The computed property `geo_mean = sqrt(bmaj × bmin)` gives the radius of a circle with the same area as the beam ellipse. This is the standard spatial search radius used throughout the pipeline (deduplication, feature linking).

For VERA at 22 GHz: `sqrt(2.1 × 0.75) = 1.26 mas`.
For VLBA at 22 GHz: `sqrt(1.4 × 0.3) = 0.65 mas`.

## Step 1 — Parse

**Input format:** The parser auto-detects the number of columns and maps them to known MFIT output columns. It handles 3 to 16 columns, with graceful defaults for missing ones:

- No flux column → all fluxes set to 1.0 (uniform weights)
- No error columns → errors set to 0.0
- No beam columns → beam prompted or defaulted

**Channel spacing detection:** Takes all unique velocity values, computes pairwise differences, histograms them, and picks the mode. This correctly handles gaps in the velocity coverage.

**Beam auto-detection:** If >50% of rows have non-NaN beam columns, uses the median. Otherwise prompts or defaults. The median is robust to outlier fits (e.g., sidelobe spots with distorted beams).

## Step 2 — Deduplicate

**Why duplicates exist:** The MFIT script (Imai 2004) divides images into overlapping subimage tiles (NOVER parameter). A spot near a tile boundary gets fitted in multiple tiles independently. These are the same emission, not independent detections.

**How it works:** For each velocity channel, brightest-first search for partners within the merge radius. Flux-weighted average position, but **peak flux kept** (not mean). The brightest fit is the one where the spot was best centered in its tile.

## Step 3 — Sidelobe flagging

**Physics:** For N stations, peak sidelobe ~ 1/N (Steinberg 1972). For VERA (N=4), sidelobes reach 25–50% of the main lobe. After CLEANing, residuals are lower but can still be fitted by SAD. The first sidelobe appears at ~1.0–1.5 × beam FWHM from the main lobe.

**Criteria (all must be met):**

1. **Flux ratio**: spot flux < (1/N + 10%) × channel peak. For VERA: 35%. For VLBA: 20%.
2. **Spatial zone**: spot is between 0.5 × bmaj and 3.0 × bmaj from the brightest spot in the same channel.
3. **Beam shape** (secondary): fitted beam deviates >50% from median (catches sidelobes even outside the spatial zone).

**Important limitation:** Only checks against the *brightest* spot per channel. Conservative by design — the real sidelobe filter is multi-epoch consistency in `masertrack_match.py`.

**Spots are flagged, not removed.** The `sidelobe` column in the spot table shows which spots were flagged and why. You can override by re-running with an `--include` file.

## Step 4 — Feature grouping (chain-linking)

This is the core algorithm, adapted from Burns' `Spots2Features.f90` with two extensions: **bidirectional search** and **gap bridging**.

**Algorithm:**

1. Sort all unflagged spots by flux (brightest first). Bright spots seed features.
2. For each unassigned spot, start a new feature.
3. Grow the chain in **both** velocity directions from the seed.
4. Within each direction, search for the nearest (in velocity) unassigned spot that is within the spatial linking radius AND within the velocity gap limit.
5. **The reference position walks**: after linking spot N, the next search is centered on spot N's position. This allows features with gradual spatial drift across their velocity extent to be correctly linked.
6. When no more candidates are found, move to the next unassigned spot.

**Gap bridging (max_vel_gap = 2.5 channels):** If a feature's emission dipped below the detection threshold in one channel, the chain looks ahead up to 2.5 channel spacings. This bridges one missing channel without connecting truly separate features.

**Bidirectional improvement over Burns:** Burns' Fortran code sorted by file position and searched forward only. If the brightest spot is in the middle of a feature, his code misses the lower-velocity half. Our bidirectional search grows the chain from the seed in both directions.

## Step 5 — Multi-peak splitting

Burns' `MultiPeakCheck.f90` algorithm. Walks through each feature's flux profile and splits when a valley is found: at least `min_dip` consecutive drops followed by `min_dip` consecutive rises.

Why `min_dip = 2`? A single-channel dip could be noise. Two consecutive drops span ~0.8 km/s, comparable to thermal maser linewidths — that is a real valley between physically distinct features.

## Step 6 — Summarize

For each feature, computes summary properties from the N brightest spots (default N=3, or fewer if the feature has fewer spots).

**Two velocity columns:**

- `vlsr_peak`: velocity of the single brightest spot. This is the channel to track for parallax fitting (Burns et al. 2015 showed that tracking individual channels is more reliable than flux-weighted feature positions).
- `vlsr_avg`: flux-weighted average velocity of the N brightest spots. This is the kinematic centroid, useful for outflow modeling.

**Three error columns per axis:**

- `x_err_formal`: flux-weighted average of MFIT fitting errors. Statistical only; underestimates true uncertainty.
- `x_err_scatter`: standard deviation of ALL spot positions in the feature. Shows the spatial extent.
- `x_err_wmean`: weighted standard error of the mean position. Best estimate of the true positional uncertainty of the feature centroid. Only computed for ≥3 spots (too few points gives unreliable scatter estimates). Computed as:

```
sigma = sqrt(sum(w_i^2 * (x_i - x_mean)^2) / (1 - sum(w_i^2)))
```

where `w_i = I_i / sum(I)` are normalized flux weights.

**Features with fewer than 3 spots** are not discarded. Properties come from however many spots exist, with formal errors used when scatter-based errors are unreliable.

## Guided mode

The `--guided` flag wraps the pipeline in an interactive shell. After each step, it shows diagnostic plots and asks for confirmation before proceeding. At the end, it prompts for an output filename.

**Typical two-pass workflow:**

1. First run: `python masertrack_identify.py --guided input.txt --n-stations 4`. Inspect the flagged spots and feature groupings.
2. If any spots need manual overriding, create an exclude file and re-run: `python masertrack_identify.py --guided input.txt --exclude mylist.txt`.

## Exclude file format

One integer index per line. Lines starting with `#` are comments. Blank lines are skipped. Indices refer to row numbers in the raw (Step 1) DataFrame.

```
# Exclude these sidelobe artifacts
42
107
# And this blended spot
58
```

## Why both a feature catalog and a spot table?

Burns et al. (2015, MNRAS 453, 3163) tested three approaches to parallax fitting for their VERA observations of S235AB-MIR:

- **Individual fitting** (same channel tracked across epochs): reliable
- **Group fitting** (multiple channels, common parallax, independent proper motions per channel): most reliable
- **Feature fitting** (flux-weighted position of 3 brightest spots): failed for features with complex velocity structure or burst behavior

Their conclusion: when masers have variable components, flux-weighted feature positions give unreliable parallaxes because changing relative brightnesses shift the centroid. Individual channel positions are stable.

**For `masertrack_match.py`:** Match features by position and velocity across epochs (using the feature catalog), then within each matched group, identify the specific channels that persist and feed those individual spot positions to `masertrack_fit.py`. This gives the best of both worlds — feature grouping for identification, channel-level tracking for astrometry.

## Input data: where the MFIT spot files come from

The MFIT pipeline (Imai 2004, Section 8 of pipelines.pdf) runs AIPS task SAD on image cubes to find and Gaussian-fit maser emission peaks. The raw SAD output is converted to a text file with physical coordinates by the `csad` Fortran program (Imai 2004, Section A.3). The `csad` converter reads image headers (`csad.header`) and SAD output files listed in `csad.list` to produce the ordered text file that this pipeline expects.

Imai also developed `mfident`, a simpler feature identification program with two hardcoded radii: 1/10 beam for deduplication and 1/2 beam for feature separation. `masertrack_identify.py` replaces `mfident` with a more flexible approach using auto-detected beam parameters, configurable thresholds, sidelobe flagging, and Burns' chain-linking algorithm with multi-peak splitting.

---

# Part 2: masertrack_match.py

## Concepts: spots, features, and groups

- **Spot**: A single Gaussian fit to one emission peak in one velocity channel at one epoch. The raw output of AIPS SAD/MFIT.
- **Feature**: A spatially and spectrally coherent group of spots — the same maser cloud across adjacent channels. Output of `masertrack_identify.py`. Exists within a single epoch.
- **Group**: The same physical maser feature tracked across multiple epochs. Output of `masertrack_match.py`. This is what gets parallax-fitted.

The naming follows the hierarchy from the literature: spots → features (Sanna et al. 2010), with "group" as the cross-epoch container.

## Normal vs inverse phase referencing

VERA's dual-beam system observes the maser (A-beam) and a reference quasar (B-beam) simultaneously.

**Normal PR**: Quasar fringes provide phase/rate solutions → applied to maser data → maser positions are in the ICRF frame, relative to the phase tracking center. No alignment needed between epochs (the quasar provides an absolute reference).

**Inverse PR**: Maser's brightest channel provides phase/rate solutions → applied to quasar data → quasar position is measured relative to the maser. Maser spots are self-calibrated with positions relative to the reference channel at (0,0). Used when the quasar is too faint for direct fringe detection (Imai et al. 2012).

The code processes both modes independently and compares results when both are available.

## Default parameters

| Parameter | Default | Reasoning |
|-----------|---------|-----------|
| Match radius | 3.0 × r_geo | Generous to allow for proper motion + position uncertainty. At 5 mas/yr and 1 yr between epochs, a maser can move ~5 mas. 3 × 1.26 = 3.78 mas base radius, plus `max_pm × dt` expands it for distant epochs. |
| Vel tolerance | 4 channels | 4 × 0.425 = 1.7 km/s. H₂O masers commonly show velocity drift of 1–3 km/s/yr as relative spot brightness shifts between channels, or from real acceleration. Combined with velocity interpolation, this catches drifts up to ~3 km/s/yr over a 1-year baseline. |
| Max PM | 5.0 mas/yr | Upper limit for Galactic maser proper motions. Solar motion + Galactic rotation contribute ~2–4 mas/yr; internal motions add 1–5 mas/yr for H₂O. |
| Min epochs | 3 | Minimum for π fitting: 5 parameters (π, μ_α cos δ, μ_δ, x₀, y₀) from ≥6 measurements (2 coordinates × ≥3 epochs). |
| Outlier sigma | 3.0 | MAD-based 3σ threshold for trajectory outlier flagging. MAD × 1.4826 gives a robust σ estimate. |

## Step 1 — Parse

Reads the epoch table and discovers feature/spot files by matching `{epoch}_{mode}_features.txt` in the input directory. Auto-detects beam from the first file's header. Flags outlier epochs by fitting linear trends to quasar shifts and identifying 3σ deviants (catches systematic offsets from degraded observations).

If `--corrections` is provided, epoch-level overrides are applied here (e.g., `use_for_pi=0`).

## Step 2 — Align

Three alignment layers for inverse PR data, applied in sequence:

### 2a. Quasar-based shifts

The measured quasar position in inverse PR encodes the maser's absolute motion:

    Q_measured(i) = Q_true - M_maser(i)

The maser's motion relative to epoch 0:

    M(i) - M(0) = -(Q(i) - Q(0))

The shift to apply at each epoch is `-(Q(i) - Q(0))`, computed from JMFIT positions in the epoch table. The RA conversion uses `shift_ra = -(dQ_seconds × 15 × cos(dec) × 1000)` mas. The Dec sign at negative declinations requires a positive sign: `shift_dec = +(dQ_seconds × 1000)` mas, because increasing Dec seconds correspond to more negative Dec values at southern declinations.

### 2b. Reference spot correction

The brightest maser channel used for FRING can change between epochs. VERA lacks Doppler tracking — the LO is manually tuned, so the peak channel in the correlator output can shift. If epoch 0 used channel 417 (−147.1 km/s) but epoch 5 uses channel 420 (−148.3 km/s), the (0,0) origin shifts to a different physical position.

`compute_ref_corrections()` finds epoch 0's reference velocity in each subsequent epoch's spot file. The position of that velocity's spot relative to (0,0) is the correction. Corrections are typically 0.01–0.28 mas and accumulate as the reference velocity drifts.

### 2c. Normal PR anchor

When both PR modes are available, inverse data is shifted into the normal PR coordinate frame. The reference maser at (0,0) in inverse has a known position in normal PR (measured from the phase tracking center). Adding this offset puts both modes in the same frame.

Without normal PR, inverse positions are self-consistent for parallax (only relative epoch-to-epoch positions matter). The absolute origin is arbitrary.

### Cross-check

When both modes are available, features at the same velocity are matched between aligned inverse and normal data. The median and RMS of position differences verify alignment quality. With the normal PR anchor, residuals should be sub-mas.

## Step 3 — Match

### 3a. Growing-chain with velocity drift tracking

The core matching algorithm in `SingleModeMatch.match()`:

1. **Seed selection**: The epoch with the most features seeds matching — more features means more candidates. Starting from the brightest unmatched feature gives reliable anchor positions.

2. **Nearest-in-time ordering**: Other epochs are searched nearest-first in time. This minimizes extrapolation distance — the first matches are most reliable because position and velocity haven't had time to change much.

3. **Position interpolation**: With ≥2 matched epochs, the expected position at the next epoch is linearly interpolated (or extrapolated beyond the current range). This implicitly tracks proper motion — the search center follows the feature's trajectory.

4. **Velocity drift tracking**: The expected velocity is interpolated identically to position. H₂O masers commonly show velocity drifts of 1–3 km/s/yr as the relative brightness of spots within a feature shifts between channels. Without interpolation, a feature drifting 1.7 km/s over a year (like the −148 km/s reference maser in our test data) would be lost. With it, the search velocity tracks the drift.

5. **Match acceptance**: A candidate must satisfy both:
   - Spatial: within `match_radius × beam_geo + max_pm × |dt|` of interpolated position
   - Velocity: within `vel_tolerance × chan_spacing` of interpolated velocity
   - Nearest spatial match wins (greedy)

6. **Uniqueness**: Each feature belongs to one group. Brightest features seed first.

### 3b. Trajectory smoothness check

`_check_trajectories()` fits linear proper motion (RA and Dec independently) to each group with ≥4 epochs. Epochs with position residuals >3σ (MAD-based) are flagged. This catches:

- **Mismatches**: different physical feature accidentally grabbed at one epoch
- **Systematic offsets**: degraded observations (e.g., 3-station VERA epoch r14307a: +1.6 mas RA, +2.9 mas Dec residual)

Flagged epochs get `use_for_pi=0` in output — kept in the group but excluded from parallax fitting.

### 3c. Velocity drift computation

`_compute_vel_drift()` fits linear V_LSR vs time for each group with ≥3 epochs. The slope (km/s/yr) measures either real acceleration of maser cloudlets or apparent drift as peak brightness shifts between channels. Both are scientifically interesting. Reported in group catalog and plots.

### 3d. Grading

`grade_group()` uses the number of unflagged epochs where the group appears divided by total unflagged epochs. Flagged epochs are excluded from the denominator — a group in all 8 good epochs out of 9 total gets Grade A.

### 3e. Corrections application

`apply_corrections()` processes the corrections file after matching: removes specified epochs from groups, recalculates statistics and grades. The template is auto-generated listing all groups with trajectory flags as pre-populated suggestions.

## Step 4 — Save

### Error floor

Spot position errors from SAD/MFIT can be zero due to output precision truncation (4 decimal places in mas = 0.1 μas precision, while real errors are ~1–10 μas). The code applies a floor following Reid et al. (1988):

    error_floor = max(beam_geo / (2 × SNR), 0.001 mas)

For VERA with beam_geo = 1.28 mas and SNR = 100: floor = 6.4 μas. For brightest spots (SNR ~400): floor = 1.6 μas. Feature errors use the wmean from `masertrack_identify.py` with the same floor.

### Tracked spots vs tracked features

**Spots** (`*_spots_tracked.txt`): One row per velocity channel × epoch within each group. Raw positions with floored errors. Primary parallax fitting input — track the same `vlsr` across epochs.

**Features** (`*_features_tracked.txt`): One row per feature × epoch. Flux-weighted centroid positions with wmean errors. Alternative for feature-level fitting, but Burns et al. (2015) showed this fails for variable masers.

### ID mapping

`*_id_mapping.txt` connects cross-epoch `group_id` to per-epoch `feature_id`. Allows tracing: "Group G21 in epoch r14248a was feature_id 15" — look up that feature in the identify output for full details.

### Corrections template

Auto-generated with all groups listed. Trajectory-flagged epochs as commented-out `EXCLUDE` suggestions. User uncomments desired corrections and re-runs with `--corrections`.

## Plots

### Diagnostic (per mode, 3×2 grid)

| Panel | Content |
|-------|---------|
| (0,0) | Sky map — rainbow by V_LSR, marker shape by epoch |
| (0,1) | V_LSR vs time — shows velocity drift |
| (1,0) | RA vs time for A/B groups — proper motion visible |
| (1,1) | Dec vs time for A/B groups |
| (2,0) | RA residuals from linear fit — **outliers visible here** |
| (2,1) | Dec residuals — red rings on flagged epochs |

The residual panels are the most important for quality control. They subtract the linear proper motion, so what remains should be scatter around zero (plus the parallax sinusoid, which is small). Points far from zero indicate mismatches or systematic errors.

### Dashboard (per mode, 2×2 grid)

Grade distribution (bar chart with counts), statistics text (groups, tracked spots, parameters), velocity coverage (how many epochs per group — shows which velocity ranges are well-sampled), velocity drift distribution (which groups are accelerating).

### Comparison (2×2, when both modes available)

Side-by-side sky maps with same axis limits. Labels show V_LSR of groups common to both modes (matched by velocity within 2 channels and position within 5× beam). Grade histogram comparing both modes. Statistics text box with counts of common, normal-only, and inverse-only groups.

### Interactive HTML (3×2, plotly)

Rows 1–2: Feature-level sky map, velocity tracking, RA/Dec vs time. Row 3: Spot-level sky map and velocity, showing individual spots within each group colored by epoch and sized by flux. Marker opacity encodes time (faint = early epoch, solid = late). Legend toggles linked across all panels.

---

## Credits and references

- Burns, R. A. et al. (2015). MNRAS 453, 3163 — chain-linking, feature vs spot π fitting
- Imai, H. (2004). AIPS pipelines for JVN/VERA — MFIT, csad, mfident
- Imai, H. et al. (2012). PASJ 64, 142 — inverse phase referencing
- Hirota, T. et al. (2020). PASJ 72, 50 — First VERA Astrometry Catalog
- Middelberg, E. & Bach, U. (2008) — VLBI imaging, sidelobe levels
- Miyoshi, M. et al. (2024) — EHT PSF sidelobe analysis
- Reid, M. J. et al. (1988) — thermal positional uncertainty θ/(2×SNR)
- Reid, M. J. et al. (2009). ApJ 693, 397 — BeSSeL π methodology
- Reid, M. J. et al. (2014). ApJ 783, 130 — outlier-tolerant fitting, √N correction
- Sanna, A. et al. (2010). A&A 517, A78 — feature definition, error-weighted positions
- Sato, M. et al. (2010). ApJ 720, 1055 — optimal epoch scheduling
- Steinberg, J. L. (1972) — 1/N sidelobe scaling
- Thompson, Moran & Swenson (2017) — synthesized beam theory

---

# Part 3: masertrack_fit.py

## File structure

```
Lines      Section
------     -------
1-48       Module docstring (credits, version, references, changelog)
50-120     Imports, constants, and default parameters
122-220    Input parsing (tracked files, coordinates)
222-340    Parallax factors (DE440 ephemeris + analytical fallback)
342-515    Core fitting engine (FitResult, design matrices, WLS solver)
517-635    Error floor finding (per-coordinate bisection to χ²_red = 1)
637-705    Cauchy outlier-tolerant fitting
707-750    Bootstrap resampling (epoch-level)
752-890    Distance estimation (1/π default, optional priors)
892-985    Channel data extraction (spots → per-VLSR channel arrays)
988-1290   Fitting functions (individual, group, combined)
1295-1460  Iterative fitting (Imai+2013, kept as diagnostic)
1463-1625  Outlier detection (epoch-level + Chauvenet group-level)
1627-2160  ParallaxFitter class (orchestrates 9-step pipeline)
2160-2400  Output writers (results table, 4 export formats)
2400-2610  Interactive HTML — spot-level (3×3 + pi vs VLSR + PM)
2610-2770  Interactive HTML — feature-level
2770-2920  Proper motion plot (AU axes, Orosz-style scale bars)
2920-3100  Parallax overview matrix (per-group + combined bootstrap)
3100-3240  Per-group 3×3 parallax fit plot
3240-3440  Feature-level 3×3 plot
3440-3600  Summary and diagnostic plots
3600-3750  Publication data + final parallax export
3750-3900  Validation (Reid+2009 Sgr B2 regression test)
3900-4450  CLI, main, self-test
```

## Parallax factors

The parallax of a source at (α, δ) produces an apparent position shift proportional to the Earth–Sun vector projected onto the sky. The code provides two methods:

1. **DE440 ephemeris** (preferred): Uses `astropy.coordinates.get_body_barycentric` for the Earth's barycentric position (X, Y, Z) in ICRS equatorial AU.
2. **Analytical** (fallback): Computes solar ecliptic longitude from the mean anomaly (Astronomical Almanac), Earth–Sun distance from eccentricity, converts to equatorial via obliquity.

**Critical convention:** The input data is in Δα·cos(δ) (not raw Δα), so F_α does NOT include a 1/cos(δ) factor. This matches the BeSSeL/pmpar standard (Reid+2009).

## Design matrices

For a single channel with N epochs: 5 parameters [π, μ_x, x₀, μ_y, y₀], 2N data points (interleaved RA, Dec). For a group with K channels: 1 + 4K parameters (shared π, independent μ and offset per channel). This is the Burns+2015 "group fitting" approach.

## Error floors (Reid+2009)

Systematic astrometric errors exceed formal MFIT uncertainties. The code finds ε_RA and ε_Dec via bisection (scipy.optimize.brentq) such that σ_total² = σ_formal² + ε² gives χ²_red = 1.0 in each coordinate independently. If formal errors already give χ²_red < 1, no floor is needed (ε = 0).

## Bootstrap

Epoch resampling with replacement: for each trial, resample the epoch indices and refit. All channels at a given epoch move together (preserving correlated systematics). The 16th/84th percentiles give the 68% CI. Samples cached to `.npz` for instant reload.

## Chauvenet outlier detection

Statistical replacement for hard-coded velocity exclusions. Computes a robust reference parallax from the weighted mean of all groups, then flags groups where P(|z| > |z_obs|) < 1/(2N). Correctly identifies the −62 km/s group in IRAS 18113 as an outlier without manual intervention.

## Distance estimation

Default is D = 1/π with asymmetric errors — the community standard for VLBI masers (Reid+2019). Optional Galactic disk or exponential priors available for sanity checking. The parallax overview plot shows the 1/π likelihood PDF alongside prior-dependent posteriors in different colors for transparent comparison.

## Feature fitting

Implemented following Burns et al. (2015, MNRAS 453, 3163) and Burns et al. (2017, MNRAS 467, L36, AFGL 5142). Uses flux-weighted feature centroids from `masertrack_match.py`. Feature fits are reported alongside channel-level group fits for comparison, but are never used as the primary result — Burns showed that variable masers give unreliable feature-level parallaxes because changing component brightnesses shift the centroid. The code generates separate 3×3 diagnostic plots for feature fits so the user can visually compare.

## Plots

| Function | Output | Description |
|----------|--------|-------------|
| `plot_parallax_fit` | 3×3 PNG per group | Sky + RA/Dec vs time + parallax ellipse + sinusoids + residuals |
| `plot_feature_fit` | 3×3 PNG per group | Same for feature centroids |
| `plot_final_parallax_overview` | Matrix PNG | Per-group rows: parallax curve, bootstrap histogram, distance PDF |
| `plot_final_proper_motions` | Single PNG | PM vectors with VLSR colors, AU axes, Orosz+2017 scale bars |
| `plot_spots_html` | Interactive HTML | Per-channel toggling with different symbols |
| `plot_features_html` | Interactive HTML | Feature-level with group toggling |

## Credits and references

- Burns, R. A. et al. (2015). MNRAS 453, 3163 — individual/group/feature fitting (S235AB-MIR)
- Burns, R. A. et al. (2017). MNRAS 467, L36 — AFGL 5142, feature fitting limitations
- Reid, M. J. et al. (2009). ApJ 693, 397 — BeSSeL fitting, error floors
- Reid, M. J. et al. (2009). ApJ 705, 1548 — Sgr B2 parallax (validation data)
- Reid, M. J. et al. (2014). ApJ 783, 130 — √N correction, outlier tolerance
- Reid, M. J. et al. (2019). ApJ 885, 131 — Galactic model, distance estimation
- Imai, H. et al. (2013). PASJ 65, 28 — Iterative parallax fitting
- Orosz, G. et al. (2017). MNRAS 468, L63 — IRAS 18113-2503 proper motions
- Bailer-Jones, C. A. L. (2015). PASP 127, 994 — Exponential distance prior
- Andriantsaralaza, M. et al. (2022). A&A 667, A74 — AGB distance priors
- Bland-Hawthorn, J. & Gerhard, O. (2016). ARA&A 54, 529 — Galactic disk model
