# masertrack — Changelog

## v1.1 (2026-04-12)

### Code
- **fit.py**: Removed dead code — `fit_iterative()` (155 lines, Imai+2013 approach that converges to same answer as group fit) and `cauchy_log_likelihood()` (17 lines, unused). Fixed `--quick` help text. Updated docstring.
- **match.py**: Expanded compressed import blocks to match identify.py style. Fixed 6 bare `except:` clauses with specific exception types. Added docstrings and inline comments to key functions. Updated docstring.
- **identify.py**: Fixed bug where `-o` output path and guided mode save did not respect `--outdir` (text files were saved to current directory while plots went to outdir). Updated docstring.

### Documentation
- New `tutorial/TUTORIAL.md` — step-by-step walkthrough with synthetic data
- New `tutorial/` — synthetic VLBI maser data (normal + inverse PR) with known parallax (π = 0.500 mas), expected outputs for comparison
- README restructured with quick-scan overview for referees, "spots → features → groups → parallax" conceptual section, tutorial link
- CODE_BREAKDOWN updated with Part 2 file structure table, removed iterative fitting references
- CHANGELOG.md added to repository

### Validation
- Pipeline tested end-to-end on synthetic tutorial data (both PR modes)
- Normal PR recovers π = 0.498 ± 0.022 mas (true: 0.500 mas)
- Inverse PR recovers π = 0.488 ± 0.028 mas (true: 0.500 mas)
- Reid+2009 Sgr B2 validation unchanged (0.0σ offset)

## v1.0 (2026-04-06)

First public release. Complete pipeline from raw VLBI spot-fitting output to trigonometric parallax measurements.

### masertrack_identify.py
Six-step pipeline for converting MFIT/SAD Gaussian spot tables into cleaned maser feature catalogs: parse input → deduplicate overlapping tile fits → flag dirty-beam sidelobes (1/N scaling, Steinberg 1972) → chain-link features across velocity channels (adapted from Burns+2015 Fortran pipeline) → split multi-peaked features → summarize with three error estimates. Produces PNG diagnostics, 9-panel summary dashboard, interactive Plotly HTML, and click-to-exclude tool.

### masertrack_match.py
Cross-epoch matching with growing-chain algorithm and velocity drift tracking. Processes both normal and inverse phase referencing independently. Inverse PR alignment via quasar-based shifts, reference spot correction, and normal PR anchor. Quality grading (A–F), trajectory smoothness checking, iterative corrections file workflow. Multi-panel diagnostic plots and interactive HTML.

### masertrack_fit.py
Parallax and proper motion fitting implementing the BeSSeL/VERA methodology. Individual channel fitting, group fitting (common π, independent μ per channel — primary result following Burns+2015), feature fitting (comparison). Error floors via bisection to χ²_red = 1 (Reid+2009). Bootstrap epoch resampling for robust uncertainties. Cauchy outlier-tolerant fitting. Chauvenet group outlier detection. Distance estimation (1/π default, optional disk/exponential priors). Publication CSV export and 4 format exporters (pmpar, BeSSeL, VERA, KaDai). Validated against Reid et al. 2009 Sgr B2 (0.0σ offset).
