# masertrack

**VLBI maser spot-to-feature identification, cross-epoch matching, and astrometric fitting.**

| Script | Version | What it does | Input | Output |
|--------|---------|-------------|-------|--------|
| `masertrack_identify.py` | 1.0.0 | Spots → Features (per epoch) | MFIT/SAD spot table | Feature catalog + spot table |
| `masertrack_match.py` | 1.0.0 | Features → Groups (cross-epoch) | Feature catalogs + epoch table | Matched groups + tracked positions |
| `masertrack_fit.py` | — | Groups → Astrometry (planned) | Tracked positions | Parallax + proper motion |

[BSD-3-Clause](LICENSE) | [Cite](CITATION.cff)

`masertrack` is a suite of Python scripts that take the raw output of VLBI spectral-line Gaussian spot fitting (AIPS MFIT/SAD, JMFIT) and produce cleaned maser feature catalogs, cross-epoch matched feature groups, and parallax/proper motion fits.

It works with any VLBI array: VERA, VLBA, LBA, EVN, KaVA, EAVN, or any other interferometric array that produces spectral-line image cubes with fitted emission peaks.

## Requirements

### Required

- Python 3.8 or later
- numpy
- pandas

### Optional (recommended)

- **matplotlib** — PNG diagnostic plots at each pipeline step and summary dashboard. Without this, no plots are generated but the pipeline runs fine and produces text output.
- **plotly** — Interactive HTML plots with hover info, zoom, pan, and legend toggles. Opens in your browser. Highly recommended for inspecting results.
- **adjustText** — Prevents label overlap on feature map plots. A small quality-of-life improvement.

### Install everything

    pip3 install numpy pandas matplotlib plotly adjustText

Or install only what you need:

    pip3 install numpy pandas              # minimum (text output only)
    pip3 install numpy pandas matplotlib   # adds PNG plots
    pip3 install numpy pandas matplotlib plotly adjustText  # full functionality

---

# masertrack_identify.py

## Quick start

For your first time with a new dataset, use guided mode:

    python masertrack_identify.py --guided my_mfit_output.txt --n-stations 4

For batch processing (no plots, no prompts):

    python masertrack_identify.py my_mfit_output.txt -o my_features.txt --no-plots

The `--n-stations` flag is important: it sets the sidelobe flagging threshold based on the theoretical 1/N sidelobe scaling law (Steinberg 1972). Use 4 for VERA, 10 for VLBA, 5 for LBA, etc.

## What does masertrack_identify.py do?

The pipeline converts a raw list of fitted emission peaks (from AIPS MFIT/SAD) into a cleaned catalog of maser features. A "feature" is a spatially and spectrally coherent emission group -- the same physical maser cloud detected across multiple adjacent velocity channels.

### Pipeline overview

    Raw MFIT spots (e.g., 210 spots)
         |
    Step 1: Parse -- read file, detect beam size and channel spacing
         |
    Step 2: Deduplicate -- merge duplicate fits (-> e.g., 100 spots)
         |
    Step 3: Flag sidelobes -- identify dirty beam artifacts (-> e.g., 96 unflagged)
         |
    Step 4: Group features -- chain-link across velocity channels (-> e.g., 31 features)
         |
    Step 5: Split multi-peaks -- separate blended features (-> e.g., 34 features)
         |
    Step 6: Summarize -- compute position, flux, errors per feature
         |
    Feature catalog + Spot table (both output files)

## Input file format

The expected input is the output of Imai's `csad` converter (csad.f, 2004) applied to raw AIPS SAD/MFIT output. The full format has 16 space-separated columns:

    VLSR(km/s)  X(mas)  Xerr  Y(mas)  Yerr  I(Jy/beam)  Ierr  S(Jy)  Serr  BMAJ(mas)  BMAJerr  BMIN(mas)  BMINerr  PA(deg)  PAerr  RMS(Jy/beam)

Reduced formats are also accepted:

- **4 columns**: VLSR, X, Y, I -- works without errors or beam info
- **3 columns minimum**: VLSR, X, Y -- uniform weights, beam must be specified manually

If beam parameters or channel spacing cannot be auto-detected, the pipeline will prompt you (guided mode) or use sensible defaults.

## Output files

Two output files are produced. Both are space-padded fixed-width text, readable by eye and by any data analysis tool.

### Feature catalog (`*_features.txt`)

One row per feature with summary properties. Key columns:

| Column | Description |
|--------|-------------|
| `vlsr_peak` | Velocity of the brightest spot -- the channel to track for parallax fitting |
| `vlsr_avg` | Flux-weighted average velocity -- the kinematic centroid |
| `x`, `y` | Flux-weighted position of the N brightest spots (mas) |
| `x_err_formal` | Formal MFIT position error (mas) |
| `x_err_scatter` | Standard deviation of spot positions -- spatial extent (mas) |
| `x_err_wmean` | Weighted standard error of the mean -- best position uncertainty (mas) |
| `I_peak` | Peak intensity (Jy/beam) |

### Spot table (`*_spots.txt`)

One row per spot with its feature assignment (`feature_id` column). This is the file used by `masertrack_match.py` and `masertrack_fit.py` for parallax fitting -- tracking individual velocity channels across epochs (see Burns et al. 2015, MNRAS 453, 3163).

### Why two output files?

Burns et al. (2015) tested three approaches to parallax fitting for their VERA observations of S235AB-MIR:

- **Individual fitting** (same channel tracked across epochs): reliable
- **Group fitting** (multiple channels, common parallax, independent proper motions per channel): most reliable
- **Feature fitting** (flux-weighted position of 3 brightest spots): failed for features with complex velocity structure or burst behavior

Their conclusion: when masers have variable components, flux-weighted feature positions give unreliable parallaxes because changing relative brightnesses shift the centroid. Individual channel positions are stable.

**For `masertrack_match.py`:** Match features by position and velocity across epochs (using the feature catalog), then within each matched group, identify the specific channels that persist and feed those individual spot positions to `masertrack_fit.py`. This gives the best of both worlds -- feature grouping for identification, channel-level tracking for astrometry.

## Command-line options

### Modes

    --guided          Interactive step-by-step mode (recommended for first pass)
    --no-plots        Disable all plots (for batch/script use)
    --show-plots      Open matplotlib windows (default: save PNG only)

### Beam and array parameters

    --beam-major F    Beam major axis in mas (default: auto-detect)
    --beam-minor F    Beam minor axis in mas (default: auto-detect)
    --beam-pa F       Beam position angle in degrees (default: auto-detect)
    --chan-spacing F   Channel spacing in km/s (default: auto-detect)
    --n-stations N    Number of array stations (sets sidelobe threshold)

### Pipeline tuning

    --dedup-radius F    Dedup merge radius as multiple of beam geometric mean (default: 1.0)
    --sidelobe-flux F   Sidelobe flux ratio threshold (default: set by --n-stations)
    --link-radius F     Feature linking radius as multiple of beam geometric mean (default: 1.0)
    --max-gap F         Max velocity gap to bridge in channel spacings (default: 2.5)
    --min-dip N         Min consecutive drops to trigger peak splitting (default: 2)
    --n-brightest N     Number of brightest spots for position averaging (default: 3)
    --exclude FILE      File listing spot indices to exclude (one per line, # for comments)
    --include FILE      File listing spot indices to force-include (overrides sidelobe flags)

### Output

    -o FILE             Output filename (default: inputname_features.txt)
    --outdir DIR        Output directory (default: .)

## Usage examples

### VERA 22 GHz water maser (4 stations)

    python masertrack_identify.py --guided maser_mfit.dat.txt --n-stations 4

### VLBA 22 GHz water maser (10 stations)

    python masertrack_identify.py --guided maser_mfit.dat.txt --n-stations 10

### Batch processing multiple epochs

    for f in epoch*_mfit.dat.txt; do
        python masertrack_identify.py "$f" --no-plots --n-stations 4
    done

### Re-running with exclusions

If you find a bad spot (say, index 42 is a sidelobe that was not flagged), create an exclude file:

    echo "42" > exclude.txt
    python masertrack_identify.py --guided maser_mfit.dat.txt --exclude exclude.txt

## Using from Python

    from masertrack_identify import MaserTrackIdentify

    m = MaserTrackIdentify("input.txt", n_stations=4)
    m.step1_parse()
    m.step2_deduplicate()
    m.step3_flag_sidelobes()
    m.step4_group_features()
    m.step5_split_multipeaks()
    m.step6_summarize()

    print(m.features)   # pandas DataFrame of features
    print(m.split_df)    # pandas DataFrame of spots with feature IDs
    print(m.beam)        # auto-detected beam parameters

    m.save("output_features.txt")

## Output plots

| File | Content |
|------|---------|
| `*_s1.png` – `*_s6.png` | Step-by-step diagnostics |
| `*_summary.png` | Pipeline dashboard (9 panels) |
| `*_interactive.html` | Interactive plotly (hover, zoom, legend toggles) |

## Testing locally

    # 1. Make sure you have Python 3.8+ and dependencies:
    python3 --version
    pip install numpy pandas matplotlib

    # 2. Run on a test file:
    python3 masertrack_identify.py --guided your_mfit_output.txt --n-stations 4

    # 3. Check the output files:
    less your_mfit_output_features.txt
    less your_mfit_output_spots.txt

    # 4. If plots don't display, try setting the backend:
    #    export MPLBACKEND=TkAgg   (Linux)
    #    export MPLBACKEND=MacOSX  (macOS)

## How the algorithms work

See `CODE_BREAKDOWN.md` for a detailed explanation of each step, the physics behind each parameter choice, and how the algorithms relate to Ross Burns' original Fortran pipeline.

---

# masertrack_match.py

## Quick start

    python masertrack_match.py epoch_table.txt --input-dir ./input --outdir results
    open results/match_inverse_interactive.html

## What does masertrack_match.py do?

Takes per-epoch feature catalogs from `masertrack_identify.py` and finds which features persist across epochs — which detections at similar positions and velocities across different observation dates represent the same physical maser cloud tracked over months to years.

### Pipeline overview

    Per-epoch feature catalogs (e.g., 9 epochs × 2 PR modes = 18 files)
         |
    Step 1: Parse — read epoch table, discover feature/spot files, detect beam
         |
    Step 2: Align — apply inverse PR shifts, reference spot correction, normal PR anchor
         |
    Step 3: Match — growing-chain matching with velocity drift tracking
         |              → trajectory smoothness check (flag outlier epochs)
         |              → velocity drift computation per group
         |              → quality grading A–F
         |
    Step 4: Save — matched groups, tracked spots/features, corrections template
         |
    Output: Group catalog + tracked positions for π fitting

## Input files

### Epoch table (required)

A space-separated text file describing the observations. One row per epoch:

    # epoch      date         mjd      dec_year     n_sta  inv_shift_ra   inv_shift_dec  inv_qso_ra_s     inv_qso_dec_s
      r14094b    2014-04-04   56751    2014.254795  4            0.0000        0.0000     57.84845310      12.585093
      r14163a    2014-06-12   56820    2014.443836  4           -1.1419       -0.1110     57.84853710      12.584982

**Where each column comes from:**

| Column | Description | Source |
|--------|-------------|--------|
| `epoch` | Observation code (must match filenames) | Your naming convention |
| `date` | ISO date | Observation log |
| `mjd` | Modified Julian Date | Observation log or AIPS header |
| `dec_year` | Decimal year | Computed from MJD |
| `n_sta` | Number of stations with fringes | Check SNEDT/SNPLT in AIPS reduction log |
| `inv_qso_ra_s` | Quasar RA seconds from JMFIT | B-beam inverse log: `RA 18 20 XX.XXXXXXXX` |
| `inv_qso_dec_s` | Quasar Dec arcseconds from JMFIT | B-beam inverse log: `DEC -25 28 XX.XXXXXX` |
| `inv_shift_ra/dec` | Maser motion shifts for alignment (mas) | Computed from quasar positions (see below) |

**How to compute the inverse PR shifts from JMFIT quasar positions:**

In your B-beam inverse reduction log, the JMFIT output on the phase reference source gives:

    JMFIT1:  RA 18 20 57.84845310 +/-  0.00000125468
    JMFIT1:  DEC -25 28 12.585093 +/-   0.0000402851

The RA seconds (57.84845310) and Dec arcseconds (12.585093) go in `inv_qso_ra_s` and `inv_qso_dec_s`. The shifts are then:

    dQ = Q(epoch_i) - Q(epoch_0)           # quasar position change vs first epoch
    shift_ra  = -(dQ_ra_seconds × 15 × cos(dec) × 1000) mas
    shift_dec = +(dQ_dec_seconds × 1000) mas

Set epoch 0 shifts to 0.0. The positive sign on Dec accounts for the sign convention at negative (southern) declinations. The code handles everything else.

### Feature and spot files (required)

Output from `masertrack_identify.py`, placed in the `--input-dir` directory:

    {epoch}_{mode}_features.txt     e.g. r14094b_inverse_features.txt
    {epoch}_{mode}_spots.txt        e.g. r14094b_inverse_spots.txt

Both `normal` and `inverse` modes are processed independently if present.

## Output files

### Per mode (normal and/or inverse)

| File | Description | Used by |
|------|-------------|---------|
| `*_groups.txt` | One row per matched group: velocity, grade, position, scatter, drift rate | Inspection, publication sky maps |
| `*_spots_tracked.txt` | Individual channel positions across epochs | **Primary input for `masertrack_fit.py`** |
| `*_features_tracked.txt` | Feature-level positions across epochs | Alternative input for fitting |
| `*_id_mapping.txt` | group_id ↔ feature_id per epoch | Tracing back to identify output |
| `*_diagnostic.png` | 6-panel diagnostic (sky, V, RA/Dec vs time, residuals) | Quality assessment |
| `*_dashboard.png` | 4-panel overview (grades, stats, coverage, drift) | Quick summary |
| `*_interactive.html` | Interactive plotly (6 panels + spot-level view) | Detailed inspection |

### Utility files

| File | Purpose |
|------|---------|
| `match_comparison.png` | Normal vs inverse side-by-side (when both modes present) |
| `match_corrections_template.txt` | Editable template for iterative correction |
| `match_ref_maser_table.txt` | Reference maser velocity drift and alignment corrections per epoch — already applied during alignment, this file is for inspection only |

### What goes forward to π fitting?

**For spot-level fitting (recommended, Burns et al. 2015):** Use `*_spots_tracked.txt`. Each row is one velocity channel at one epoch within a matched group. The fitter tracks the same `vlsr` channel across epochs within each `group_id`, fitting for a common π and independent μ per channel.

**For feature-level fitting:** Use `*_features_tracked.txt`. Each row is the flux-weighted feature centroid at one epoch. Simpler but less reliable when masers have complex velocity structure.

**Key columns for fitting:**

| Column | Role |
|--------|------|
| `group_id` | Which physical maser cloud |
| `dec_year` | Time coordinate for fitting |
| `vlsr` | Channel identity (track same channel across epochs) |
| `x`, `y` | Position offsets (mas) |
| `x_err`, `y_err` | Position error (mas, floor = beam/(2×SNR), min 0.001 mas) |
| `use_for_pi` | 1 = include in fit, 0 = flagged epoch or trajectory outlier |

## Corrections workflow

The pipeline supports iterative refinement:

1. **Run once** — inspect diagnostic plots and console output
2. **Edit the corrections template** (`match_corrections_template.txt`) — uncomment lines to exclude bad epochs or remove mismatched epochs from specific groups
3. **Re-run** with `--corrections my_edits.txt`
4. **Repeat** until the matching looks clean

Example corrections file:

    # Exclude epoch from parallax fitting
    EPOCH  r14307a  use_for_pi=0

    # Remove a mismatched epoch from group G6
    GROUP  G6  r15037b  EXCLUDE

## Quality grading

Groups are graded A–F based on how many unflagged epochs they appear in:

| Grade | Criterion | Parallax quality |
|-------|-----------|-----------------|
| A | All unflagged epochs | Best candidates |
| B | ≥2/3 of unflagged epochs and ≥4 epochs | Good candidates |
| C | ≥4 epochs | Usable with caution |
| D | 3 epochs | Marginal |
| E | 2 epochs | Proper motion only |
| F | 1 epoch | Not matched |

Flagged epochs (e.g., 3-station observations with systematic offsets) are excluded from the grading denominator, so a group detected in all 8 unflagged epochs out of 9 total gets Grade A.

## Interactive HTML plot

The plotly interactive HTML has 6 panels in 3 rows:

- **Row 1**: Sky map (rainbow by velocity, opacity encodes time: faint=early, solid=late) + velocity tracking (shows drift)
- **Row 2**: RA offset vs time + Dec offset vs time
- **Row 3**: Spot sky map (individual spots colored by epoch, sized by flux) + spot velocity vs epoch (shows channel persistence)

Click legend to toggle groups. "Show all"/"Hide all" buttons. Hover for epoch, position, velocity, flags, drift rate.

## Command-line options

### Input/output

    epoch_table           Epoch table file (required, positional argument)
    --input-dir DIR       Directory with feature/spot files (default: ./input)
    --outdir DIR          Output directory (default: .)
    --corrections FILE    Corrections file for manual overrides

### Beam parameters

    --beam-major F        Beam major axis in mas (default: auto-detect)
    --beam-minor F        Beam minor axis in mas (default: auto-detect)
    --chan-spacing F       Channel spacing in km/s (default: auto-detect)

### Matching parameters

    --match-radius F      Match radius × beam geo mean (default: 3.0)
    --vel-tolerance F     Velocity tolerance in channels (default: 4)
    --max-pm F            Max proper motion in mas/yr (default: 5.0)
    --min-epochs N        Min epochs for display threshold (default: 3)

### Plot control

    --no-plots            Disable all plots
    --show-plots          Open matplotlib windows (default: save PNG only)

## Using from Python

    from masertrack_match import MaserTrackMatch

    m = MaserTrackMatch("epoch_table.txt", input_dir="./input", outdir="results")
    m.run()

    # Access results
    for mode, matcher in m.matchers.items():
        print(f"{mode}: {len(matcher.groups)} groups")
        for g in matcher.groups:
            if g["grade"] in "AB":
                print(f"  G{g['group_id']} V={g['vlsr_peak']:.1f} {g['grade']}")

## Testing locally

    # 1. Prepare input
    mkdir input
    cp *_features.txt *_spots.txt input/

    # 2. Run
    python3 masertrack_match.py your_epoch_table.txt --input-dir ./input --outdir results

    # 3. Inspect
    open results/match_inverse_interactive.html
    less results/match_inverse_groups.txt

    # 4. If plots don't display:
    #    export MPLBACKEND=TkAgg   (Linux)
    #    export MPLBACKEND=MacOSX  (macOS)

## How the algorithms work

See `CODE_BREAKDOWN.md` for a detailed explanation of the matching algorithm, alignment procedure, trajectory checking, and how each parameter choice relates to the physics of maser astrometry.

---

## Credits

- **Pipeline design**: Gabor Orosz
- **Chain-linking algorithm**: Adapted from Ross Burns' Fortran code (Burns et al. 2015, MNRAS 453, 3163)
- **MFIT/SAD spot fitting and csad converter**: Hiroshi Imai (2004)
- **Sidelobe scaling theory**: Thompson, Moran & Swenson (2017); Steinberg (1972)
- **Astrometry methodology**: Reid et al. (2009, ApJ 693, 397); Hirota et al. (2020, PASJ 72, 50)
- **Inverse PR technique**: Imai et al. (2012, PASJ 64, 142); Burns et al. (2015)
- **Code implementation**: Claude (Anthropic), 2026

## License

BSD-3-Clause. See `LICENSE` file.

## Citation

If you use masertrack in your research, please cite the repository. GitHub displays a "Cite this repository" button (from the `CITATION.cff` file) with ready-to-use BibTeX and APA formats.
