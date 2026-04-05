# masertrack

**VLBI maser spot-to-feature identification, cross-epoch matching, and astrometric fitting.**

`masertrack` is a suite of three Python scripts that take the raw output of VLBI spectral-line Gaussian spot fitting (AIPS MFIT/SAD, JMFIT) and produce cleaned maser feature catalogs, cross-epoch matched feature groups, and parallax/proper motion fits.

It works with any VLBI array: VERA, VLBA, LBA, EVN, KaVA, EAVN, or any other interferometric array that produces spectral-line image cubes with fitted emission peaks.

## The three scripts

| Script | What it does | Input | Output |
|--------|-------------|-------|--------|
| `masertrack_identify.py` | Spots -> Features (per epoch) | MFIT/SAD spot table | Feature catalog + spot table |
| `masertrack_match.py` | Features -> Groups (cross-epoch) | Multiple feature catalogs | Matched feature groups |
| `masertrack_fit.py` | Groups -> Astrometry | Matched groups | Parallax & proper motion |

This README covers `masertrack_identify.py`. Documentation for `masertrack_match.py` and `masertrack_fit.py` will be added as they are developed.

## Requirements

- Python 3.8 or later
- numpy
- pandas
- matplotlib (optional, for plots)

Install dependencies:

    pip install numpy pandas matplotlib

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

### Spot table (`*_features_spots.txt`)

One row per spot with its feature assignment (`feature_id` column). This is the file used by `masertrack_match.py` and `masertrack_fit.py` for parallax fitting -- tracking individual velocity channels across epochs (see Burns et al. 2015, MNRAS 453, 3163).

### Why two output files?

Burns et al. (2015) tested three approaches to parallax fitting and found that tracking individual velocity channels ("individual fitting" and "group fitting") gave reliable parallaxes, while flux-weighted feature positions ("feature fitting") failed when masers had complex velocity structure. The spot table preserves individual channel positions for parallax fitting, while the feature catalog provides the summary properties needed for kinematic analysis and cross-epoch matching.

## Command-line options

### Modes

    --guided          Interactive step-by-step mode (recommended for first pass)
    --no-plots        Disable all plots (for batch/script use)

### Beam and array parameters

    --beam-major F    Beam major axis in mas
    --beam-minor F    Beam minor axis in mas
    --beam-pa F       Beam position angle in degrees
    --chan-spacing F   Channel spacing in km/s
    --n-stations N    Number of array stations (sets sidelobe threshold)

### Pipeline tuning

    --dedup-radius F    Dedup merge radius as multiple of beam geometric mean (default: 1.0)
    --sidelobe-flux F   Sidelobe flux ratio threshold (default: set by --n-stations)
    --link-radius F     Feature linking radius as multiple of beam geometric mean (default: 1.0)
    --max-gap F         Max velocity gap to bridge in channel spacings (default: 2.5)
    --min-dip N         Min consecutive drops to trigger peak splitting (default: 2)
    --n-brightest N     Number of brightest spots for position averaging (default: 3)
    --exclude FILE      File listing spot indices to exclude (one per line, # for comments)

### Output

    -o FILE             Output filename (default: inputname_features.txt)

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

    m = MaserTrackIdentify("input.txt", interactive=True, n_stations=4)
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

## Setting up the GitHub repository

    # 1. Create the repo on GitHub (github.com/chickin/masertrack)
    # 2. Clone and add files:
    git clone https://github.com/chickin/masertrack.git
    cd masertrack
    cp /path/to/masertrack_identify.py .
    cp /path/to/README.md .
    cp /path/to/CODE_BREAKDOWN.md .
    cp /path/to/LICENSE .
    cp /path/to/CITATION.cff .

    # 3. Commit and push:
    git add -A
    git commit -m "Initial commit: masertrack_identify.py v1.0"
    git push origin main

## Testing locally

    # 1. Make sure you have Python 3.8+ and dependencies:
    python3 --version
    pip install numpy pandas matplotlib

    # 2. Run on a test file:
    python3 masertrack_identify.py --guided your_mfit_output.txt --n-stations 4

    # 3. Check the output files:
    less your_mfit_output_features.txt
    less your_mfit_output_features_spots.txt

    # 4. If plots don't display, try setting the backend:
    #    export MPLBACKEND=TkAgg   (Linux)
    #    export MPLBACKEND=MacOSX  (macOS)

## How the algorithms work

See `CODE_BREAKDOWN.md` for a detailed explanation of each step, the physics behind each parameter choice, and how the algorithms relate to Ross Burns' original Fortran pipeline.

## Credits

- **Pipeline design**: Gabor Orosz
- **Chain-linking algorithm**: Adapted from Ross Burns' Fortran code (Imfit2Spots, Spots2Features, MultiPeakCheck, ThreeStrongest)
- **MFIT/SAD spot fitting and csad converter**: Hiroshi Imai (2004)
- **Sidelobe scaling theory**: Thompson, Moran & Swenson (2017); Steinberg (1972); Middelberg & Bach (2008)
- **Feature vs spot parallax methodology**: Burns et al. (2015, MNRAS 453, 3163); Reid et al. (2009a, ApJ 693, 397)
- **Code implementation**: Claude (Anthropic), 2026

## License

BSD-3-Clause. See `LICENSE` file.

## Citation

If you use masertrack in your research, please cite the repository. GitHub displays a "Cite this repository" button (from the `CITATION.cff` file) with ready-to-use BibTeX and APA formats.
