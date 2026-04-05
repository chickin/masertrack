# masertrack_identify.py - Detailed Code Breakdown

This document explains every section of the `masertrack_identify.py` pipeline: what the code does, why it does it that way, and what physical reasoning underlies each choice.

## File structure

```
Lines     Section
------    -------
1-95      Module docstring (credits, usage, description)
97-123    Imports (numpy required; pandas, matplotlib optional)
126-167   Default parameters (DEFAULTS dictionary)
170-198   BeamInfo data class
201-280   MaserTrackIdentify constructor and exclusion list loader
283-430   Step 1 - Parse
433-540   Step 2 - Deduplicate
543-680   Step 3 - Sidelobe flagging
683-820   Step 4 - Feature grouping (chain-linking)
823-890   Step 5 - Multi-peak splitting
893-1020  Step 6 - Summarize
1023-1045 run() - execute all steps
1048-1120 save() and fixed-width writer
1123-1215 Plotting functions
1218-1290 Guided mode
1293-1405 Command-line interface
```

---

## Default parameters (DEFAULTS dictionary)

All tunable parameters live in one dictionary at the top. The key design principle: **no hard-coded spatial values in mas**. All distances are expressed as multiples of the beam size, and all velocity ranges as multiples of the channel spacing. This is what makes the code array-agnostic.

| Parameter | Default | Source / Reasoning |
|-----------|---------|-------------------|
| Dedup radius | 1.0 x r_geo | Burns used ~1.0 mas; r_geo gives same scale for VERA |
| Sidelobe zone | 0.5-3.0 x bmaj | First sidelobe at 1.0-1.5 x beam (Miyoshi+ 2024), second/third extend to ~3x |
| Sidelobe flux | 1/N + 0.10 | Steinberg (1972) 1/N scaling + margin for post-CLEAN residuals |
| Sidelobe beam tol | 50% | Empirical; sidelobe Gaussian fits are typically >50% distorted |
| Link radius | 1.0 x r_geo | Same as dedup radius for consistency |
| Max vel gap | 2.5 channels | Bridges 1 missing channel; 2+ missing = real break |
| Min dip channels | 2 | Burns' MultiPeakCheck value; 1 would be too sensitive to noise |
| N brightest | 3 | Burns' ThreeStrongest; balances precision vs. robustness |

---

## BeamInfo class

Holds four numbers: `bmaj`, `bmin`, `pa`, `chan_spacing`. The computed property `geo_mean = sqrt(bmaj x bmin)` gives the radius of a circle with the same area as the beam ellipse. This is the standard spatial search radius used throughout the pipeline (deduplication, feature linking).

For VERA at 22 GHz: `sqrt(2.1 x 0.75) = 1.26 mas`.
For VLBA at 22 GHz: `sqrt(1.4 x 0.3) = 0.65 mas`.

---

## Step 1 - Parse

**Input format:** The parser auto-detects the number of columns and maps them to known MFIT output columns. It handles 3 to 16 columns, with graceful defaults for missing ones:

- No flux column -> all fluxes set to 1.0 (uniform weights)
- No error columns -> errors set to 0.0
- No beam columns -> beam prompted or defaulted

**Channel spacing detection:** Takes all unique velocity values, computes pairwise differences, histograms them, and picks the mode. This correctly handles gaps in the velocity coverage.

**Beam auto-detection:** If >50% of rows have non-NaN beam columns, uses the median. Otherwise prompts or defaults. The median is robust to outlier fits (e.g., sidelobe spots with distorted beams).

---

## Step 2 - Deduplicate

**Why duplicates exist:** The MFIT script (Imai 2004) divides images into overlapping subimage tiles (NOVER parameter). A spot near a tile boundary gets fitted in multiple tiles independently. These are the same emission, not independent detections.

**How it works:** For each velocity channel, brightest-first search for partners within the merge radius. Flux-weighted average position, but **peak flux kept** (not mean). The brightest fit is the one where the spot was best centered in its tile.

---

## Step 3 - Sidelobe flagging

**Physics:** For N stations, peak sidelobe ~ 1/N (Steinberg 1972). For VERA (N=4), sidelobes reach 25-50% of the main lobe. After CLEANing, residuals are lower but can still be fitted by SAD. The first sidelobe appears at ~1.0-1.5 x beam FWHM from the main lobe.

**Criteria (all must be met):**

1. **Flux ratio**: spot flux < (1/N + 10%) x channel peak. For VERA: 35%. For VLBA: 20%.
2. **Spatial zone**: spot is between 0.5 x bmaj and 3.0 x bmaj from the brightest spot in the same channel.
3. **Beam shape** (secondary): fitted beam deviates >50% from median (catches sidelobes even outside the spatial zone).

**Important limitation:** Only checks against the *brightest* spot per channel. Conservative by design -- the real sidelobe filter is multi-epoch consistency in `masertrack_match.py`.

**Spots are flagged, not removed.** The `sidelobe` column in the spot table shows which spots were flagged and why. You can override by re-running with an exclude file.

---

## Step 4 - Feature grouping (chain-linking)

This is the core algorithm, adapted from Burns' `Spots2Features` with two extensions: **bidirectional search** and **gap bridging**.

**Algorithm:**

1. Sort all spots by flux (brightest first). Bright spots seed features.
2. For each unassigned spot, start a new feature.
3. Grow the chain in **both** velocity directions from the seed.
4. Within each direction, search for the nearest (in velocity) unassigned spot that is within the spatial linking radius AND within the velocity gap limit.
5. **The reference position walks**: after linking spot N, the next search is centered on spot N's position. This allows features with gradual spatial drift across their velocity extent to be correctly linked.
6. When no more candidates are found, move to the next unassigned spot.

**Gap bridging (max_vel_gap = 2.5 channels):** If a feature's emission dipped below the detection threshold in one channel, the chain looks ahead up to 2.5 channel spacings. This bridges one missing channel without connecting truly separate features.

**Bidirectional improvement over Burns:** Burns' Fortran code sorted by file position and searched forward only. If the brightest spot is in the middle of a feature, his code misses the lower-velocity half. Our bidirectional search grows the chain from the seed in both directions.

---

## Step 5 - Multi-peak splitting

Burns' `MultiPeakCheck` algorithm. Walks through each feature's flux profile and splits when a valley is found: at least `min_dip` consecutive drops followed by `min_dip` consecutive rises.

Why `min_dip = 2`? A single-channel dip could be noise. Two consecutive drops span ~0.8 km/s, comparable to thermal maser linewidths -- that is a real valley between physically distinct features.

---

## Step 6 - Summarize

For each feature, computes summary properties from the N brightest spots (default N=3, or fewer if the feature has fewer spots).

**Two velocity columns:**

- `vlsr_peak`: velocity of the single brightest spot. This is the channel to track for parallax fitting (Burns et al. 2015 showed that tracking individual channels is more reliable than flux-weighted feature positions).
- `vlsr_avg`: flux-weighted average velocity of the N brightest spots. This is the kinematic centroid, useful for outflow modeling.

**Three error columns per axis:**

- `x_err_formal`: flux-weighted average of MFIT fitting errors. Statistical only; underestimates true uncertainty.
- `x_err_scatter`: standard deviation of ALL spot positions in the feature. Shows the spatial extent. For a 7-spot feature scattered over 0.15 mas, this is 0.15 mas.
- `x_err_wmean`: weighted standard error of the mean position. This is the best estimate of the true positional uncertainty of the feature centroid. For the same 7-spot feature: ~0.06 mas. Computed as:

```
sigma = sqrt(sum(w_i^2 * (x_i - x_mean)^2) / (1 - sum(w_i^2)))
```

where `w_i = I_i / sum(I)` are normalized flux weights.

**Features with fewer than 3 spots** are not discarded. If a feature has 1 spot, all properties come from that single spot. If 2 spots, position is the flux-weighted average of both.

---

## Guided mode

The `--guided` flag wraps the pipeline in an interactive shell. After each step, it shows diagnostic plots and asks for confirmation before proceeding. At the end, it prompts for an output filename.

**Typical two-pass workflow:**

1. First run: `python masertrack_identify.py --guided input.txt --n-stations 4`. Inspect the flagged spots and feature groupings.
2. If any spots need manual overriding, create an exclude file and re-run: `python masertrack_identify.py --guided input.txt --exclude mylist.txt`.

---

## Exclude file format

One integer index per line. Lines starting with `#` are comments. Blank lines are skipped. Indices refer to row numbers in the raw (Step 1) DataFrame.

```
# Exclude these sidelobe artifacts
42
107
# And this blended spot
58
```

---

## Why both a feature catalog and a spot table?

Burns et al. (2015, MNRAS 453, 3163) tested three approaches to parallax fitting for their VERA observations of S235AB-MIR:

- **Individual fitting** (same channel tracked across epochs): reliable
- **Group fitting** (multiple channels, common parallax, independent proper motions per channel): most reliable
- **Feature fitting** (flux-weighted position of 3 brightest spots): failed for features with complex velocity structure or burst behavior

Their conclusion: when masers have variable components, flux-weighted feature positions give unreliable parallaxes because changing relative brightnesses shift the centroid. Individual channel positions are stable.

**For `masertrack_match.py`:** Match features by position and velocity across epochs (using the feature catalog), then within each matched group, identify the specific channels that persist and feed those individual spot positions to `masertrack_fit.py`. This gives the best of both worlds -- feature grouping for identification, channel-level tracking for astrometry.

---

## Input data: where the MFIT spot files come from

The MFIT pipeline (Imai 2004, Section 8 of pipelines.pdf) runs AIPS task SAD on image cubes to find and Gaussian-fit maser emission peaks. The raw SAD output is converted to a text file with physical coordinates by the `csad` Fortran program (Imai 2004, Section A.3). The `csad` converter reads image headers (`csad.header`) and SAD output files listed in `csad.list` to produce the ordered text file that this pipeline expects.

Imai also developed `mfident`, a simpler feature identification program with two hardcoded radii: 1/10 beam for deduplication and 1/2 beam for feature separation. `masertrack_identify.py` replaces `mfident` with a more flexible approach using auto-detected beam parameters, configurable thresholds, sidelobe flagging, and Burns' chain-linking algorithm with multi-peak splitting.

---

## Credits and references

- Burns, R. A. et al. (2015). "A 'water spout' maser jet in S235AB-MIR." MNRAS 453, 3163. -- Feature vs spot parallax fitting; chain-linking algorithm.
- Imai, H. (2004). "AIPS automatic pipelines for JVN/VERA." -- MFIT pipeline; csad converter; mfident.
- Middelberg, E. & Bach, U. (2008). "High Resolution Radio Astronomy Using Very Long Baseline Interferometry." -- VLBI imaging review; sidelobe levels.
- Reid, M. J. et al. (2009a). "Trigonometric Parallaxes of Massive Star-Forming Regions. I." ApJ 693, 397. -- BeSSeL parallax methodology.
- Reid, M. J. & Honma, M. (2014). "Micro-Arcsecond Radio Astrometry." ARA&A 52, 339. -- Review of VLBI maser astrometry.
- Steinberg, J. L. (1972). "Random Array Beamforming." -- 1/N sidelobe scaling law.
- Thompson, A. R., Moran, J. M., & Swenson, G. W. (2017). "Interferometry and Synthesis in Radio Astronomy." -- Synthesized beam theory.
