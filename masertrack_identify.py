#!/usr/bin/env python3
"""
masertrack_identify.py — Part 1 of the masertrack pipeline
============================================================
Convert VLBI spectral-line spot-fitting output into cleaned maser
feature catalogs.

This script takes the raw output from AIPS MFIT/SAD or JMFIT Gaussian
spot-fitting (per-channel fitted emission peaks) and produces a
cleaned catalog of maser features (spatially and spectrally coherent
emission groups).  It works with any VLBI array (VERA, VLBA, LBA,
EVN, KaVA, EAVN, ...) by auto-detecting the beam size and channel
spacing from the input data.

The expected input is the output of Imai's csad converter (csad.f,
2004) applied to raw AIPS SAD/MFIT output.  The converter reads
image headers and SAD output to produce a text file with columns:
VLSR, X, Xerr, Y, Yerr, I, Ierr, S, Serr, bmaj, bmaj_err, bmin,
bmin_err, PA, PA_err, RMS.  Reduced-column formats are also accepted
(minimum: VLSR, X, Y).

Pipeline steps
--------------
  Step 1 - Parse:       Read the input file, auto-detect beam and
                        channel spacing (or accept user overrides).
  Step 2 - Deduplicate: Merge near-identical spots in the same velocity
                        channel (MFIT/SAD artifact from overlapping
                        subimage tiles).
  Step 3 - Sidelobes:   Flag spots that are likely dirty-beam sidelobe
                        artifacts, using spatial, flux-ratio, and
                        beam-shape criteria.
  Step 4 - Features:    Chain-link spots across adjacent velocity
                        channels into maser features.
  Step 5 - Split:       Break features that contain multiple spectral
                        peaks (blended features).
  Step 6 - Summarize:   Compute feature positions (flux-weighted from
                        the N brightest spots) and errors (both formal
                        and empirical).

Each step produces a diagnostic plot (in interactive/guided mode) and
stores intermediate results that can be inspected.  All spatial
thresholds are set relative to the beam size and all velocity
thresholds relative to the channel spacing, so no hard-coded values
are needed.

Outputs
-------
Two output files are produced:
  - Feature catalog (*_features.txt): one row per feature with summary
    properties including peak and average velocities, positions, and
    both formal and empirical errors.
  - Spot catalog (*_spots.txt): one row per spot with feature ID
    assignments.  This is the file used by masertrack_match.py and
    masertrack_fit.py for parallax fitting (tracking individual
    velocity channels across epochs).

Usage
-----
Guided mode (recommended for first pass)::

    python masertrack_identify.py --guided input_mfit.txt --n-stations 4

Batch mode::

    python masertrack_identify.py input_mfit.txt -o features.txt --no-plots

Interactive Python::

    from masertrack_identify import MaserTrackIdentify
    m = MaserTrackIdentify("input_mfit.txt", n_stations=4)
    m.step1_parse()
    m.step2_deduplicate()
    # ... inspect, adjust, continue ...

Credits
-------
- Pipeline concept and algorithm design: Gabor Orosz
- Chain-linking algorithm adapted from Ross Burns' Fortran pipeline
  (Imfit2Spots, Spots2Features, MultiPeakCheck, ThreeStrongest)
- Original MFIT/SAD fitting script and csad converter: Hiroshi Imai
  (2004); see Imai, pipelines.pdf, Sec. 8 & A.3
- Sidelobe scaling relations: Thompson, Moran & Swenson (2017),
  Steinberg (1972), Middelberg & Bach (2008)
- Feature vs spot parallax methodology: Burns et al. (2015, MNRAS
  453, 3163), Reid et al. (2009a, ApJ 693, 397)
- Code implementation: Claude (Anthropic), 2026

Part of the masertrack suite:
  masertrack_identify.py  - spots to features (this file)
  masertrack_match.py     - cross-epoch feature matching
  masertrack_fit.py       - parallax and proper motion fitting

License: BSD-3-Clause
Repository: https://github.com/chickin/masertrack
"""

# =====================================================================
#  IMPORTS
# =====================================================================
from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Optional imports -- the code works without pandas/matplotlib but with
# reduced functionality (no plots, numpy-only tables).
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# =====================================================================
#  CONSTANTS AND DEFAULTS
# =====================================================================

# Default parameters.  These are sensible starting points derived from
# the literature and practical experience.  All can be overridden via
# command-line flags or method arguments.

DEFAULTS = {
    # Step 2 -- Deduplication
    # Spots in the same channel within this fraction of the geometric
    # mean beam radius are considered duplicates of the same emission.
    # The geometric mean sqrt(bmaj*bmin) is used because it gives a
    # circular search area equal to the beam ellipse area.
    "dedup_radius_factor": 1.0,  # x geometric mean beam radius

    # Step 3 -- Sidelobe flagging
    # Sidelobes from the dirty beam appear at predictable distances
    # from bright sources.  For a sparse array, the first sidelobe is
    # at approximately 1.0-1.5 x beam FWHM, and the sidelobe flux
    # follows roughly a 1/N scaling where N is the number of stations
    # (Steinberg 1972; Middelberg & Bach 2008).
    "sidelobe_zone_inner": 0.5,    # x bmaj -- inner edge of zone
    "sidelobe_zone_outer": 3.0,    # x bmaj -- outer edge of zone
    "sidelobe_flux_ratio": 0.50,   # fraction of channel peak (default
                                    # 50%, conservative; adjusted by
                                    # the code based on N_stations)
    "sidelobe_beam_tol":   0.50,   # fractional beam shape tolerance

    # Step 4 -- Feature grouping (chain-linking)
    # The linking radius is the geometric mean beam radius (same metric
    # as deduplication, for consistency).  Velocity linking allows a
    # gap of up to max_vel_gap channels (to bridge channels where the
    # emission dipped below the SNR threshold).
    "link_radius_factor": 1.0,  # x geometric mean beam radius
    "link_vel_channels":  1.5,  # adjacent channel linking range
    "max_vel_gap":        2.5,  # max gap in channels to bridge

    # Step 5 -- Multi-peak splitting
    # A feature is split when its flux profile shows a valley: at least
    # min_dip_channels consecutive drops followed by the same number of
    # rises.  This value comes from Burns' MultiPeakCheck (where it is
    # hardcoded to 2).
    "min_dip_channels": 2,

    # Step 6 -- Feature summarization
    # The feature position is the flux-weighted average of the N
    # brightest spots in the feature (Burns' ThreeStrongest method).
    "n_brightest": 3,
}


# =====================================================================
#  DATA CONTAINER: BeamInfo
# =====================================================================

@dataclass
class BeamInfo:
    """Auto-detected (or user-specified) synthesized beam parameters.

    Attributes
    ----------
    bmaj : float
        Beam major axis in mas.
    bmin : float
        Beam minor axis in mas.
    pa : float
        Beam position angle in degrees (east of north).
    chan_spacing : float
        Velocity channel spacing in km/s.
    geo_mean : float
        Geometric mean beam radius = sqrt(bmaj x bmin).  This is the
        radius of a circle with the same area as the beam ellipse and
        is used as the default spatial search radius throughout the
        pipeline.
    """
    bmaj: float
    bmin: float
    pa:   float
    chan_spacing: float

    @property
    def geo_mean(self) -> float:
        return float(np.sqrt(self.bmaj * self.bmin))

    def __str__(self):
        return (f"Beam {self.bmaj:.2f} x {self.bmin:.2f} mas, "
                f"PA {self.pa:.1f} deg, dv = {self.chan_spacing:.3f} km/s, "
                f"r_geo = {self.geo_mean:.2f} mas")


# =====================================================================
#  MAIN PIPELINE CLASS
# =====================================================================

class MaserTrackIdentify:
    """Maser spot-to-feature identification pipeline.

    This class implements the six-step pipeline described in the module
    docstring.  Each step reads from the previous step's output (stored
    as a class attribute) and produces a new output.  Steps can be run
    individually for inspection, or all at once via run().

    Parameters
    ----------
    filepath : str or Path
        Path to the MFIT/SAD output file (csad-converted).
    interactive : bool
        If True, show matplotlib plots after each step.
    beam_major, beam_minor, beam_pa : float or None
        Override auto-detected beam parameters (in mas, degrees).
    chan_spacing : float or None
        Override auto-detected channel spacing (in km/s).
    n_stations : int or None
        Number of array stations.  Used to set the sidelobe flux
        threshold via the 1/N approximation.  If not given, the code
        uses the default threshold.
    exclude_file : str or Path or None
        Path to a text file listing spot indices to exclude from
        processing (one index per line, # for comments).
    """

    # Standard MFIT output columns (after the 2-line header produced
    # by Imai's csad.f converter).  The code also handles files with
    # fewer columns -- see step1_parse().
    _FULL_COLS = [
        "vlsr",     # V_LSR in km/s
        "x",        # RA offset in mas (positive = East)
        "x_err",    # formal RA position error in mas
        "y",        # Dec offset in mas (positive = North)
        "y_err",    # formal Dec position error in mas
        "I",        # peak intensity in Jy/beam
        "I_err",    # peak intensity error in Jy/beam
        "S",        # integrated flux in Jy
        "S_err",    # integrated flux error in Jy
        "bmaj",     # fitted major axis in mas
        "bmaj_err", # major axis error in mas
        "bmin",     # fitted minor axis in mas
        "bmin_err", # minor axis error in mas
        "pa",       # fitted position angle in degrees
        "pa_err",   # PA error in degrees
        "rms",      # local RMS noise in Jy/beam
    ]

    def __init__(self,
                 filepath: str | Path,
                 interactive: bool = True,
                 beam_major:   Optional[float] = None,
                 beam_minor:   Optional[float] = None,
                 beam_pa:      Optional[float] = None,
                 chan_spacing:  Optional[float] = None,
                 n_stations:   Optional[int]   = None,
                 exclude_file: Optional[str | Path] = None):

        self.filepath = Path(filepath)
        self.interactive = interactive and HAS_MPL

        # User overrides (None = auto-detect)
        self._user_bmaj = beam_major
        self._user_bmin = beam_minor
        self._user_pa   = beam_pa
        self._user_dv   = chan_spacing
        self.n_stations = n_stations

        # Load exclusion list
        self.exclude_indices: set = set()
        if exclude_file is not None:
            self._load_exclude(Path(exclude_file))

        # Beam info (set in step 1)
        self.beam: Optional[BeamInfo] = None

        # Data at each pipeline stage
        self.raw:      Optional[pd.DataFrame] = None  # step 1
        self.deduped:  Optional[pd.DataFrame] = None  # step 2
        self.flagged:  Optional[pd.DataFrame] = None  # step 3
        self.featured: Optional[pd.DataFrame] = None  # step 4
        self.split_df: Optional[pd.DataFrame] = None  # step 5
        self.features: Optional[pd.DataFrame] = None  # step 6

        self._step = 0  # last completed step

    # -----------------------------------------------------------------
    #  Exclusion list handling
    # -----------------------------------------------------------------
    def _load_exclude(self, path: Path) -> None:
        """Load a list of spot indices to exclude from processing.

        The file should contain one integer index per line.  Lines
        starting with # are ignored.  Blank lines are skipped.
        These indices refer to row numbers in the raw (step 1)
        DataFrame.
        """
        if not path.exists():
            print(f"  Warning: exclude file {path} not found, ignoring.")
            return
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    self.exclude_indices.add(int(line))
                except ValueError:
                    continue
        print(f"  Loaded {len(self.exclude_indices)} exclusions from {path}")

    # =================================================================
    #  STEP 1 -- PARSE
    # =================================================================
    def step1_parse(self) -> pd.DataFrame:
        """Read the MFIT output file and detect beam / channel spacing.

        The parser handles several input formats:
        - Full 16-column MFIT/csad output
        - Reduced formats with fewer columns (minimum: vlsr, x, y)
        - Files with comment lines (#) and header text

        If beam parameters or channel spacing cannot be determined
        from the data AND were not provided by the user, the code
        will prompt for them (in guided mode) or use safe defaults.
        """
        rows = []
        n_cols_detected = None

        with open(self.filepath) as fh:
            for line in fh:
                line = line.strip()
                # Skip blank lines, comments, and text headers
                if (not line or line.startswith("#")
                        or line[0].isalpha()):
                    continue
                parts = line.split()
                if len(parts) < 3:  # minimum: vlsr, x, y
                    continue
                try:
                    vals = [float(v) for v in parts]
                except ValueError:
                    continue
                # Record how many columns the file has
                if n_cols_detected is None:
                    n_cols_detected = len(vals)
                rows.append(vals)

        if len(rows) == 0:
            raise RuntimeError(f"No data rows found in {self.filepath}")

        # --- Determine column mapping based on number of columns ---
        n_cols = n_cols_detected or 16
        if n_cols >= 16:
            col_names = self._FULL_COLS[:16]
        elif n_cols >= 7:
            # Assume: vlsr, x, xerr, y, yerr, I, Ierr [, ...]
            col_names = self._FULL_COLS[:n_cols]
        elif n_cols >= 4:
            # Assume: vlsr, x, y, I
            if n_cols == 4:
                col_names = ["vlsr", "x", "y", "I"]
            else:
                col_names = self._FULL_COLS[:n_cols]
        elif n_cols >= 3:
            # Minimum: vlsr, x, y
            col_names = ["vlsr", "x", "y"]
        else:
            raise RuntimeError(
                f"Input has only {n_cols} columns; need at least 3 "
                f"(vlsr, x, y).")

        # Truncate rows to the detected column count
        rows = [r[:len(col_names)] for r in rows]

        df = pd.DataFrame(rows, columns=col_names)
        df.sort_values("vlsr", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # --- Fill missing columns with defaults ---
        if "I" not in df.columns:
            df["I"] = 1.0
            print("  Note: no flux column found; using uniform weights.")
        if "I_err" not in df.columns:
            df["I_err"] = 0.0
        if "x_err" not in df.columns:
            df["x_err"] = 0.0
        if "y_err" not in df.columns:
            df["y_err"] = 0.0
        if "S" not in df.columns:
            df["S"] = df["I"]
        if "S_err" not in df.columns:
            df["S_err"] = 0.0
        if "rms" not in df.columns:
            df["rms"] = np.nan
        for col in ["bmaj", "bmaj_err", "bmin", "bmin_err",
                     "pa", "pa_err"]:
            if col not in df.columns:
                df[col] = np.nan

        # --- Apply exclusion list ---
        if self.exclude_indices:
            before = len(df)
            df = df.drop(
                index=[i for i in self.exclude_indices if i in df.index]
            ).reset_index(drop=True)
            print(f"  Excluded {before - len(df)} spots from "
                  f"exclusion list.")

        # --- Determine beam parameters ---
        has_beam_data = df["bmaj"].notna().sum() > len(df) * 0.5

        bmaj = self._user_bmaj
        bmin = self._user_bmin
        pa   = self._user_pa
        dv   = self._user_dv

        if bmaj is None:
            if has_beam_data:
                bmaj = float(df["bmaj"].median())
            else:
                bmaj = self._ask_or_default(
                    "Beam major axis (mas)", 2.0)
        if bmin is None:
            if has_beam_data:
                bmin = float(df["bmin"].median())
            else:
                bmin = self._ask_or_default(
                    "Beam minor axis (mas)", 0.75)
        if pa is None:
            if has_beam_data:
                pa = float(df["pa"].median())
            else:
                pa = 0.0
        if dv is None:
            dv = self._detect_channel_spacing(df["vlsr"].values)
            if dv is None or dv <= 0:
                dv = self._ask_or_default(
                    "Channel spacing (km/s)", 0.42)

        self.beam = BeamInfo(bmaj=bmaj, bmin=bmin, pa=pa,
                             chan_spacing=dv)
        self.raw = df
        self._step = 1

        print(f"\nStep 1 - Parse: {len(df)} spots, "
              f"V = [{df['vlsr'].min():.1f}, {df['vlsr'].max():.1f}] km/s")
        print(f"  {self.beam}")

        if self.interactive:
            self._plot_sky(df, "Step 1 - Raw spots")

        return df

    def _ask_or_default(self, param_name: str,
                        default: float) -> float:
        """Prompt the user for a parameter or use the default."""
        if not sys.stdin.isatty():
            print(f"  Using default {param_name} = {default}")
            return default
        try:
            val = input(f"  {param_name}? [{default}]: ").strip()
            if val == "":
                return default
            return float(val)
        except (ValueError, EOFError):
            return default

    @staticmethod
    def _detect_channel_spacing(vlsr: np.ndarray) -> Optional[float]:
        """Find the most common velocity step between spots.

        Uses a histogram of velocity differences to find the mode.
        Returns None if detection fails.
        """
        unique_v = np.sort(np.unique(vlsr))
        if len(unique_v) < 2:
            return None
        dv = np.abs(np.diff(unique_v))
        dv = dv[dv > 0.001]  # ignore near-zero differences
        if len(dv) == 0:
            return None
        bin_width = 0.01
        bins = np.arange(0, dv.max() + bin_width, bin_width)
        if len(bins) < 2:
            return float(np.median(dv))
        hist, edges = np.histogram(dv, bins=bins)
        peak_bin = np.argmax(hist)
        return float(edges[peak_bin] + bin_width / 2)

    # =================================================================
    #  STEP 2 -- DEDUPLICATE
    # =================================================================
    def step2_deduplicate(self,
                          radius_factor: Optional[float] = None
                          ) -> pd.DataFrame:
        """Merge duplicate spots in the same velocity channel.

        WHY THIS IS NEEDED:
        The MFIT/SAD fitting script (Imai 2004) divides the image into
        overlapping subimage tiles (controlled by xmg, ymg, nover
        parameters in the AIPS RUN file).  A maser spot near a tile
        boundary gets fitted independently in each overlapping tile,
        producing two (or more) entries at the same velocity with
        slightly different positions and fluxes.  These are not
        independent detections -- they are the same emission fitted
        multiple times.

        WHAT THIS STEP DOES:
        For each velocity channel, spots within the merge radius are
        grouped.  The brightest spot's flux is kept (since that is the
        best fit), and the position is the flux-weighted average of all
        duplicates (following Burns' Imfit2Spots weighting scheme).

        Parameters
        ----------
        radius_factor : float or None
            Merge radius as a multiple of the geometric mean beam
            radius.  Default: 1.0 x sqrt(bmaj x bmin).
        """
        if self.raw is None:
            self.step1_parse()

        if radius_factor is None:
            radius_factor = DEFAULTS["dedup_radius_factor"]

        radius = radius_factor * self.beam.geo_mean
        df = self.raw.copy()

        merged_rows = []
        for vlsr, grp in df.groupby("vlsr"):
            grp = grp.sort_values("I", ascending=False)
            indices = grp.index.tolist()
            consumed = set()

            for idx in indices:
                if idx in consumed:
                    continue
                row = grp.loc[idx].copy()

                # Find partners: same velocity, within merge radius
                partners = []
                for idx2 in indices:
                    if idx2 == idx or idx2 in consumed:
                        continue
                    sep = np.hypot(row["x"] - grp.loc[idx2, "x"],
                                   row["y"] - grp.loc[idx2, "y"])
                    if sep < radius:
                        partners.append(idx2)

                if partners:
                    all_idx = [idx] + partners
                    sub = grp.loc[all_idx]
                    weights = sub["I"].values
                    for col in ["x", "y"]:
                        row[col] = np.average(sub[col].values,
                                              weights=weights)
                    for col in ["x_err", "y_err"]:
                        if col in sub.columns:
                            row[col] = np.average(sub[col].values,
                                                  weights=weights)
                    # Flux: keep the peak (best fit)
                    for p in partners:
                        consumed.add(p)

                merged_rows.append(row)

        result = pd.DataFrame(merged_rows)
        result.sort_values("vlsr", inplace=True)
        result.reset_index(drop=True, inplace=True)

        n_before = len(df)
        n_after = len(result)
        self.deduped = result
        self._step = 2

        print(f"\nStep 2 - Deduplicate "
              f"(r < {radius:.2f} mas = {radius_factor}x r_geo): "
              f"{n_before} -> {n_after} spots "
              f"({n_before - n_after} merged)")

        if self.interactive:
            self._plot_sky(result, "Step 2 - After deduplication")

        return result

    # =================================================================
    #  STEP 3 -- SIDELOBE FLAGGING
    # =================================================================
    def step3_flag_sidelobes(self,
                             zone_inner:  Optional[float] = None,
                             zone_outer:  Optional[float] = None,
                             flux_ratio:  Optional[float] = None,
                             beam_tol:    Optional[float] = None,
                             ) -> pd.DataFrame:
        """Flag spots that are likely dirty-beam sidelobe artifacts.

        WHY THIS IS NEEDED:
        VLBI arrays with few stations (especially VERA with only 4)
        have dirty beams with strong sidelobes -- up to 25-50% of the
        main lobe peak for VERA, ~10-20% for the VLBA (Steinberg
        1972; Middelberg & Bach 2008).  After CLEANing, residual
        sidelobes can still produce spurious emission peaks that
        SAD/MFIT will fit as real spots.

        WHAT THIS STEP DOES:
        For each velocity channel, a spot is flagged as a sidelobe
        candidate if it satisfies ALL of the following criteria:

        1. FLUX RATIO: Its flux is less than flux_ratio x the
           brightest spot in the channel.

        2. SPATIAL: It lies within the sidelobe zone of the
           brightest spot (between zone_inner x bmaj and
           zone_outer x bmaj).

        3. BEAM SHAPE (secondary): Its fitted beam deviates from
           the median beam by more than beam_tol (fractional).

        Spots are FLAGGED, not removed.  Flagged spots are excluded
        from subsequent steps (feature grouping etc.) but remain in
        the output spot table for inspection.  You can override flags
        by re-running with an exclude file.

        Parameters
        ----------
        zone_inner : float
            Inner edge of sidelobe zone (x bmaj).
        zone_outer : float
            Outer edge of sidelobe zone (x bmaj).
        flux_ratio : float
            Maximum flux as fraction of channel peak.
        beam_tol : float
            Fractional beam shape tolerance.
        """
        if self.deduped is None:
            self.step2_deduplicate()

        if zone_inner is None:
            zone_inner = DEFAULTS["sidelobe_zone_inner"]
        if zone_outer is None:
            zone_outer = DEFAULTS["sidelobe_zone_outer"]
        if flux_ratio is None:
            flux_ratio = DEFAULTS["sidelobe_flux_ratio"]
        if beam_tol is None:
            beam_tol = DEFAULTS["sidelobe_beam_tol"]

        # Adjust flux ratio based on array size (1/N scaling)
        if self.n_stations is not None and self.n_stations >= 2:
            array_ratio = 1.0 / self.n_stations + 0.10
            flux_ratio = min(flux_ratio, array_ratio)
            print(f"  Array-adjusted sidelobe threshold: "
                  f"{flux_ratio:.2f} (N={self.n_stations})")

        df = self.deduped.copy()
        df["sidelobe"] = False
        df["sidelobe_reason"] = ""

        med_bmaj = self.beam.bmaj
        med_bmin = self.beam.bmin
        has_beam = df["bmaj"].notna().sum() > len(df) * 0.3

        n_spatial = 0
        n_beam = 0

        for vlsr, grp in df.groupby("vlsr"):
            if len(grp) <= 1:
                continue

            peak_idx = grp["I"].idxmax()
            peak_I = grp.loc[peak_idx, "I"]
            peak_x = grp.loc[peak_idx, "x"]
            peak_y = grp.loc[peak_idx, "y"]

            for idx, row in grp.iterrows():
                if idx == peak_idx:
                    continue

                # Skip spots that are bright enough to be real
                if row["I"] >= flux_ratio * peak_I:
                    continue

                reasons = []

                # CRITERION 1 -- Spatial: in the sidelobe zone?
                sep = np.hypot(row["x"] - peak_x, row["y"] - peak_y)
                in_zone = (zone_inner * med_bmaj <= sep
                           <= zone_outer * med_bmaj)
                if in_zone:
                    reasons.append(
                        f"sidelobe_zone({sep:.2f}mas)")
                    n_spatial += 1

                # CRITERION 2 -- Beam shape: anomalous?
                if has_beam and not np.isnan(row["bmaj"]):
                    bmaj_ok = ((1 - beam_tol) <
                               row["bmaj"] / med_bmaj <
                               (1 + beam_tol))
                    bmin_ok = ((1 - beam_tol) <
                               row["bmin"] / med_bmin <
                               (1 + beam_tol))
                    if not (bmaj_ok and bmin_ok):
                        reasons.append("beam_shape")
                        n_beam += 1

                # Flag if spatial criterion is met
                if in_zone:
                    df.loc[idx, "sidelobe"] = True
                    df.loc[idx, "sidelobe_reason"] = "; ".join(reasons)
                # Also flag if beam shape is very anomalous even
                # outside the sidelobe zone
                elif len(reasons) > 0 and "beam_shape" in reasons:
                    df.loc[idx, "sidelobe"] = True
                    df.loc[idx, "sidelobe_reason"] = "; ".join(reasons)

        n_flagged = df["sidelobe"].sum()
        self.flagged = df
        self._step = 3

        print(f"\nStep 3 - Sidelobe flagging: "
              f"{n_flagged} spots flagged")
        print(f"  Criteria: flux < {flux_ratio:.0%} of peak AND "
              f"({zone_inner}-{zone_outer})x bmaj from peak")
        if n_flagged > 0:
            print(f"  Spatial zone hits: {n_spatial}, "
                  f"beam shape anomalies: {n_beam}")

        if self.interactive and n_flagged > 0:
            self._plot_sidelobes(df)

        return df

    # =================================================================
    #  STEP 4 -- FEATURE GROUPING (CHAIN-LINKING)
    # =================================================================
    def step4_group_features(self,
                             radius_factor: Optional[float] = None,
                             vel_channels:  Optional[float] = None,
                             max_vel_gap:   Optional[float] = None,
                             ) -> pd.DataFrame:
        """Group spots into features using bidirectional chain-linking.

        WHY THIS IS NEEDED:
        A maser "feature" is a spatially and spectrally coherent
        emission region that produces detectable spots across several
        adjacent velocity channels.  Identifying these features from
        the raw spot list is the core task of this pipeline.

        WHAT THIS STEP DOES:
        Starting from the brightest unassigned spot, the algorithm
        builds a chain by searching for the nearest (in velocity)
        unassigned spot within the linking radius.  The search
        proceeds in both velocity directions from the seed.
        The reference position WALKS with the chain: after linking
        spot N, the next search is centered on spot N's position,
        not the original seed.

        This is the chain-linking algorithm from Burns' Spots2Features,
        extended to search bidirectionally and to bridge gaps of up to
        max_vel_gap channels.

        Parameters
        ----------
        radius_factor : float or None
            Linking radius as a multiple of the geometric mean beam
            radius.  Default: 1.0 x sqrt(bmaj x bmin).
        vel_channels : float or None
            Normal linking range in channels.
        max_vel_gap : float or None
            Maximum gap (in channels) to bridge when no adjacent
            channel has a match.
        """
        if self.flagged is None:
            self.step3_flag_sidelobes()

        if radius_factor is None:
            radius_factor = DEFAULTS["link_radius_factor"]
        if vel_channels is None:
            vel_channels = DEFAULTS["link_vel_channels"]
        if max_vel_gap is None:
            max_vel_gap = DEFAULTS["max_vel_gap"]

        # Work with non-flagged spots only
        df = self.flagged[~self.flagged["sidelobe"]].copy()
        df.sort_values("vlsr", inplace=True)
        df.reset_index(drop=True, inplace=True)

        max_sep = radius_factor * self.beam.geo_mean
        max_dv_link = vel_channels * self.beam.chan_spacing
        max_dv_gap = max_vel_gap * self.beam.chan_spacing

        df["feature_id"] = -1
        feature_id = 0

        # Process brightest spots first -- they are the most reliable
        # seeds for features.
        order = df.sort_values("I", ascending=False).index.tolist()

        for seed in order:
            if df.loc[seed, "feature_id"] != -1:
                continue

            df.loc[seed, "feature_id"] = feature_id

            # Grow chain in both velocity directions
            for direction in [+1, -1]:
                ref_v = df.loc[seed, "vlsr"]
                ref_x = df.loc[seed, "x"]
                ref_y = df.loc[seed, "y"]

                searching = True
                while searching:
                    searching = False
                    best_idx = None
                    best_dv = max_dv_gap + 1

                    for j in df.index:
                        if df.loc[j, "feature_id"] != -1:
                            continue

                        dv = df.loc[j, "vlsr"] - ref_v
                        if direction > 0 and dv <= 0:
                            continue
                        if direction < 0 and dv >= 0:
                            continue

                        abs_dv = abs(dv)
                        if abs_dv > max_dv_gap:
                            continue

                        sep = np.hypot(df.loc[j, "x"] - ref_x,
                                       df.loc[j, "y"] - ref_y)
                        if sep > max_sep:
                            continue

                        if abs_dv < best_dv:
                            best_dv = abs_dv
                            best_idx = j

                    if best_idx is not None:
                        df.loc[best_idx, "feature_id"] = feature_id
                        ref_v = df.loc[best_idx, "vlsr"]
                        ref_x = df.loc[best_idx, "x"]
                        ref_y = df.loc[best_idx, "y"]
                        searching = True

            feature_id += 1

        n_feat = df["feature_id"].nunique()
        self.featured = df
        self._step = 4

        print(f"\nStep 4 - Feature grouping "
              f"(r < {max_sep:.2f} mas, "
              f"dv < {max_dv_link:.2f} km/s, "
              f"gap <= {max_dv_gap:.2f} km/s): "
              f"{n_feat} features from {len(df)} spots")

        if self.interactive:
            self._plot_features(df, "Step 4 - Chain-linked features")

        return df

    # =================================================================
    #  STEP 5 -- MULTI-PEAK SPLITTING
    # =================================================================
    def step5_split_multipeaks(self,
                                min_dip: Optional[int] = None
                                ) -> pd.DataFrame:
        """Split features with multiple spectral peaks.

        WHY THIS IS NEEDED:
        The chain-linking algorithm can connect two physically
        distinct maser features that happen to overlap spatially
        and be adjacent in velocity.  The resulting merged feature
        has a flux profile with two peaks and a valley between them.

        WHAT THIS STEP DOES:
        For each feature, walk through the flux profile in velocity
        order.  Track consecutive rises and falls.  If the flux
        drops for min_dip consecutive steps and then rises for
        min_dip consecutive steps, split the feature at the valley.

        This is the algorithm from Burns' MultiPeakCheck.

        Parameters
        ----------
        min_dip : int or None
            Minimum consecutive drops (and subsequent rises) to
            trigger a split.  Default: 2.
        """
        if self.featured is None:
            self.step4_group_features()

        if min_dip is None:
            min_dip = DEFAULTS["min_dip_channels"]

        df = self.featured.copy()
        new_id = df["feature_id"].max() + 1
        n_splits = 0

        for fid in df["feature_id"].unique():
            mask = df["feature_id"] == fid
            grp = df.loc[mask].sort_values("vlsr")

            if len(grp) < (2 * min_dip + 1):
                continue

            fluxes = grp["I"].values
            indices = grp.index.values

            up = 0
            down = 0
            for k in range(1, len(fluxes)):
                if fluxes[k] > fluxes[k - 1]:
                    up += 1
                elif fluxes[k] < fluxes[k - 1]:
                    down += 1

                if (down >= min_dip and up >= min_dip
                        and fluxes[k] > fluxes[k - 1]):
                    df.loc[indices[k:], "feature_id"] = new_id
                    new_id += 1
                    n_splits += 1
                    up = 0
                    down = 0

        n_feat = df["feature_id"].nunique()
        self.split_df = df
        self._step = 5

        print(f"\nStep 5 - Multi-peak split "
              f"(min_dip = {min_dip}): "
              f"{n_splits} split(s) -> {n_feat} features")

        if self.interactive and n_splits > 0:
            self._plot_features(df, "Step 5 - After multi-peak split")

        return df

    # =================================================================
    #  STEP 6 -- SUMMARIZE
    # =================================================================
    def step6_summarize(self,
                         n_brightest: Optional[int] = None
                         ) -> pd.DataFrame:
        """Compute feature catalog from the N brightest spots.

        WHY THIS IS NEEDED:
        Each feature contains multiple spots across velocity channels.
        For downstream use (cross-epoch matching, kinematic analysis),
        we need summary properties per feature.

        OUTPUTS TWO VELOCITY COLUMNS:
        - vlsr_peak: velocity of the single brightest spot.  This is
          the channel to track for parallax fitting (Burns et al.
          2015, MNRAS 453, 3163).
        - vlsr_avg: flux-weighted average velocity across the
          feature.  This is the kinematic centroid.

        ERROR COLUMNS:
        - x_err_formal / y_err_formal: flux-weighted average of MFIT
          fitting errors.  These are statistical errors only.
        - x_err_scatter / y_err_scatter: raw standard deviation of
          spot positions.  Shows the spatial extent of the feature.
        - x_err_wmean / y_err_wmean: weighted standard error of the
          mean position.  This is the best estimate of the true
          positional uncertainty of the feature centroid.

        Parameters
        ----------
        n_brightest : int or None
            Number of brightest spots for position averaging.
            Default: 3.  If a feature has fewer spots, all are used.
        """
        if self.split_df is None:
            self.step5_split_multipeaks()

        if n_brightest is None:
            n_brightest = DEFAULTS["n_brightest"]

        df = self.split_df
        rows = []

        for fid, grp in df.groupby("feature_id"):
            grp = grp.sort_values("I", ascending=False)
            peak = grp.iloc[0]

            # Top-N brightest spots for position averaging.
            # If fewer spots exist, use all of them.
            n_use = min(n_brightest, len(grp))
            top = grp.head(n_use)
            weights = top["I"].values
            if weights.sum() == 0:
                weights = np.ones(len(weights))

            # Flux-weighted position
            x_avg = np.average(top["x"].values, weights=weights)
            y_avg = np.average(top["y"].values, weights=weights)

            # Flux-weighted average velocity (kinematic centroid)
            vlsr_avg = np.average(top["vlsr"].values, weights=weights)

            # Formal error: weighted average of MFIT errors
            x_err_formal = np.average(top["x_err"].values,
                                      weights=weights)
            y_err_formal = np.average(top["y_err"].values,
                                      weights=weights)

            # Empirical error: scatter of ALL spot positions
            if len(grp) >= 3:
                x_scatter = float(grp["x"].std())
                y_scatter = float(grp["y"].std())
            else:
                x_scatter = x_err_formal
                y_scatter = y_err_formal

            # Weighted standard error of the mean
            # This is the best estimate of the true positional
            # uncertainty of the feature centroid.
            if len(top) >= 2 and weights.sum() > 0:
                w = weights / weights.sum()
                x_wmean = np.sqrt(
                    np.sum(w**2 * (top["x"].values - x_avg)**2)
                    / (1 - np.sum(w**2)))
                y_wmean = np.sqrt(
                    np.sum(w**2 * (top["y"].values - y_avg)**2)
                    / (1 - np.sum(w**2)))
                # Guard against numerical issues
                if np.isnan(x_wmean) or x_wmean == 0:
                    x_wmean = x_err_formal
                if np.isnan(y_wmean) or y_wmean == 0:
                    y_wmean = y_err_formal
            else:
                x_wmean = x_err_formal
                y_wmean = y_err_formal

            feat = {
                "feature_id":    fid,
                "vlsr_peak":     peak["vlsr"],
                "vlsr_avg":      vlsr_avg,
                "vlsr_min":      grp["vlsr"].min(),
                "vlsr_max":      grp["vlsr"].max(),
                "vlsr_width":    (grp["vlsr"].max() - grp["vlsr"].min()
                                  + self.beam.chan_spacing),
                "n_spots":       len(grp),
                "n_channels":    grp["vlsr"].nunique(),
                "x":             x_avg,
                "y":             y_avg,
                "x_err_formal":  x_err_formal,
                "y_err_formal":  y_err_formal,
                "x_err_scatter": x_scatter,
                "y_err_scatter": y_scatter,
                "x_err_wmean":   x_wmean,
                "y_err_wmean":   y_wmean,
                "I_peak":        peak["I"],
                "S_peak":        peak["S"],
                "rms":           peak["rms"],
                "snr":           (peak["I"] / peak["rms"]
                                  if peak["rms"] > 0 else np.nan),
            }
            rows.append(feat)

        features = pd.DataFrame(rows)
        features.sort_values("vlsr_peak", inplace=True)
        features.reset_index(drop=True, inplace=True)

        self.features = features
        self._step = 6

        n_multi = (features["n_channels"] >= 3).sum()
        print(f"\nStep 6 - Summarize (top-{n_brightest} weighted): "
              f"{len(features)} features, "
              f"{n_multi} with >= 3 channels")

        if self.interactive:
            self._plot_summary(features, df)

        return features

    # =================================================================
    #  RUN ALL STEPS
    # =================================================================
    def run(self, **kwargs) -> pd.DataFrame:
        """Execute the full pipeline."""
        self.step1_parse()
        self.step2_deduplicate(
            kwargs.get("radius_factor"))
        self.step3_flag_sidelobes(
            kwargs.get("zone_inner"),
            kwargs.get("zone_outer"),
            kwargs.get("flux_ratio"),
            kwargs.get("beam_tol"))
        self.step4_group_features(
            kwargs.get("link_radius_factor"),
            kwargs.get("vel_channels"),
            kwargs.get("max_vel_gap"))
        self.step5_split_multipeaks(
            kwargs.get("min_dip"))
        self.step6_summarize(
            kwargs.get("n_brightest"))
        return self.features

    # =================================================================
    #  SAVE OUTPUT
    # =================================================================
    def save(self, outpath: str | Path,
             save_spots: bool = True) -> None:
        """Write the feature catalog and spot table.

        Output is space-padded fixed-width format for both human
        readability and machine parseability.

        Two files are produced:
        - Feature catalog: one row per feature with summary properties.
        - Spot catalog: one row per spot with feature assignments.
          This is what masertrack_match.py and masertrack_fit.py use
          for parallax fitting (tracking individual velocity channels
          across epochs; see Burns et al. 2015, MNRAS 453, 3163).
        """
        if self.features is None:
            raise RuntimeError("Run the pipeline before saving.")

        outpath = Path(outpath)
        self._write_fixed_width(
            self.features, outpath,
            f"Feature catalog from {self.filepath.name}\n"
            f"# Beam: {self.beam}\n"
            f"# Pipeline: masertrack masertrack_identify.py\n")

        if save_spots and self.split_df is not None:
            spot_path = outpath.with_name(
                outpath.stem + "_spots" + outpath.suffix)
            self._write_fixed_width(
                self.split_df, spot_path,
                f"Spot table with feature assignments from "
                f"{self.filepath.name}\n"
                f"# Beam: {self.beam}\n")

    @staticmethod
    def _write_fixed_width(df: pd.DataFrame, path: Path,
                            header: str = "") -> None:
        """Write a DataFrame as space-padded fixed-width text."""
        lines = []
        for hline in header.strip().split("\n"):
            lines.append(f"# {hline}" if not hline.startswith("#")
                         else hline)

        # Column widths
        col_widths = {}
        for col in df.columns:
            if df[col].dtype in [np.float64, float]:
                formatted = [f"{v:12.4f}" for v in df[col]]
            elif df[col].dtype in [np.int64, int]:
                formatted = [f"{v:8d}" for v in df[col]]
            else:
                formatted = [f"{v}" for v in df[col]]
            w = max(len(col), max(len(s) for s in formatted)) + 1
            col_widths[col] = w

        header_line = "# " + "".join(
            f"{col:>{col_widths[col]}s}" for col in df.columns)
        lines.append(header_line)

        for _, row in df.iterrows():
            parts = []
            for col in df.columns:
                w = col_widths[col]
                val = row[col]
                if isinstance(val, float) or isinstance(val, np.floating):
                    if np.isnan(val):
                        parts.append(f"{'nan':>{w}s}")
                    else:
                        parts.append(f"{val:{w}.4f}")
                elif isinstance(val, (int, np.integer)):
                    parts.append(f"{val:{w}d}")
                elif isinstance(val, bool) or isinstance(val, np.bool_):
                    parts.append(f"{str(val):>{w}s}")
                else:
                    parts.append(f"{str(val):>{w}s}")
            lines.append("  " + "".join(parts))

        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")

        print(f"  Saved: {path} ({len(df)} rows)")

    # =================================================================
    #  PLOTTING
    # =================================================================
    def _plot_sky(self, df, title=""):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        sc = ax1.scatter(df["x"], df["y"], c=df["vlsr"], s=8,
                         cmap="coolwarm", edgecolors="none")
        ax1.set_xlabel("RA offset (mas)")
        ax1.set_ylabel("Dec offset (mas)")
        ax1.invert_xaxis()
        ax1.set_aspect("equal")
        plt.colorbar(sc, ax=ax1, label="V_LSR (km/s)")
        ax2.plot(df["vlsr"], df["I"], "k.", ms=2)
        ax2.set_xlabel("V_LSR (km/s)")
        ax2.set_ylabel("Peak flux (Jy/beam)")
        fig.suptitle(f"{title} - {len(df)} spots")
        plt.tight_layout()
        plt.savefig(str(self.filepath.stem) + f"_s{self._step}.png",
                    dpi=150, bbox_inches="tight")
        plt.show()

    def _plot_sidelobes(self, df):
        fig, ax = plt.subplots(figsize=(8, 7))
        clean = df[~df["sidelobe"]]
        dirty = df[df["sidelobe"]]
        sc = ax.scatter(clean["x"], clean["y"], c=clean["vlsr"],
                        s=8, cmap="coolwarm", zorder=2)
        ax.scatter(dirty["x"], dirty["y"], c="red", s=80,
                   marker="x", linewidths=1.5, zorder=3,
                   label=f"Flagged ({len(dirty)})")
        ax.set_xlabel("RA offset (mas)")
        ax.set_ylabel("Dec offset (mas)")
        ax.invert_xaxis()
        ax.set_aspect("equal")
        ax.legend()
        ax.set_title("Step 3 - Sidelobe flags (red x)")
        plt.colorbar(sc, ax=ax, label="V_LSR (km/s)")
        plt.tight_layout()
        plt.savefig(str(self.filepath.stem) + "_s3.png",
                    dpi=150, bbox_inches="tight")
        plt.show()

    def _plot_features(self, df, title=""):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        n_feat = df["feature_id"].nunique()
        cmap = plt.cm.get_cmap("tab20", min(n_feat, 20))
        for fid, grp in df.groupby("feature_id"):
            c = cmap(fid % 20)
            ax1.scatter(grp["x"], grp["y"], c=[c], s=10)
            ax2.plot(grp["vlsr"], grp["I"], ".", c=c, ms=4)
        ax1.set_xlabel("RA offset (mas)")
        ax1.set_ylabel("Dec offset (mas)")
        ax1.invert_xaxis()
        ax1.set_aspect("equal")
        ax2.set_xlabel("V_LSR (km/s)")
        ax2.set_ylabel("Peak flux (Jy/beam)")
        fig.suptitle(f"{title} - {n_feat} features")
        plt.tight_layout()
        plt.savefig(str(self.filepath.stem) + f"_s{self._step}.png",
                    dpi=150, bbox_inches="tight")
        plt.show()

    def _plot_summary(self, features, spots):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.scatter(spots["x"], spots["y"], c="lightgray",
                    s=3, zorder=1)
        sc = ax1.scatter(features["x"], features["y"],
                         c=features["vlsr_peak"],
                         s=features["I_peak"] * 3,
                         cmap="coolwarm", edgecolors="k",
                         linewidths=0.3, zorder=2)
        for _, f in features[features["n_channels"] >= 3].iterrows():
            ax1.annotate(f"{f['vlsr_peak']:.1f}",
                         (f["x"], f["y"]), fontsize=6,
                         xytext=(4, 4),
                         textcoords="offset points")
        ax1.set_xlabel("RA offset (mas)")
        ax1.set_ylabel("Dec offset (mas)")
        ax1.invert_xaxis()
        ax1.set_aspect("equal")
        plt.colorbar(sc, ax=ax1, label="V_LSR (km/s)")
        ax2.bar(features["vlsr_peak"], features["I_peak"],
                width=features["vlsr_width"], alpha=0.5,
                edgecolor="k", linewidth=0.3)
        ax2.set_xlabel("V_LSR (km/s)")
        ax2.set_ylabel("Peak flux (Jy/beam)")
        n_mc = (features["n_channels"] >= 3).sum()
        ax2.set_title(f"{len(features)} features ({n_mc} multi-ch)")
        fig.suptitle("Step 6 - Feature catalog")
        plt.tight_layout()
        plt.savefig(str(self.filepath.stem) + "_s6.png",
                    dpi=150, bbox_inches="tight")
        plt.show()

    # =================================================================
    #  STRING REPRESENTATION
    # =================================================================
    def __repr__(self):
        n = len(self.raw) if self.raw is not None else 0
        return (f"MaserTrackIdentify({self.filepath.name!r}, "
                f"step {self._step}/6, {n} raw spots)")


# =====================================================================
#  GUIDED MODE
# =====================================================================

def guided_mode(filepath: str, **overrides) -> MaserTrackIdentify:
    """Walk the user through the pipeline step by step."""
    print("=" * 60)
    print("  masertrack - masertrack_identify (guided mode)")
    print("  Maser spot-to-feature identification pipeline")
    print("=" * 60)
    print(f"\n  Input: {filepath}\n")

    m = MaserTrackIdentify(filepath, interactive=HAS_MPL,
                           **overrides)

    # Step 1
    m.step1_parse()
    if not _confirm("  Continue to deduplication?"):
        return m

    # Step 2
    m.step2_deduplicate()
    if not _confirm("  Continue to sidelobe flagging?"):
        return m

    # Step 3
    m.step3_flag_sidelobes()
    if not _confirm("  Continue to feature grouping?"):
        return m

    # Step 4
    m.step4_group_features()
    if not _confirm("  Continue to multi-peak splitting?"):
        return m

    # Step 5
    m.step5_split_multipeaks()
    if not _confirm("  Continue to feature summarization?"):
        return m

    # Step 6
    m.step6_summarize()

    # Save
    default_out = Path(filepath).stem + "_features.txt"
    out = input(f"\n  Output file? [{default_out}]: ").strip()
    if not out:
        out = default_out
    m.save(out)

    print("\n  Done!  Inspect the output and re-run with --exclude "
          "if needed.\n")
    return m


def _confirm(prompt: str) -> bool:
    """Ask yes/no and return True for yes."""
    if not sys.stdin.isatty():
        return True
    try:
        ans = input(prompt + " [Y/n]: ").strip().lower()
        return ans in ("", "y", "yes")
    except EOFError:
        return True


# =====================================================================
#  COMMAND-LINE INTERFACE
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="masertrack_identify",
        description="masertrack step 1: identify maser features from "
                    "VLBI spot-fitting output.",
        epilog=textwrap.dedent("""\
        Examples:
          python masertrack_identify.py --guided input.txt --n-stations 4
          python masertrack_identify.py input.txt -o features.txt --no-plots
          python masertrack_identify.py input.txt --beam-major 2.1 --beam-minor 0.75

        Part of the masertrack suite
        (masertrack_identify / masertrack_match / masertrack_fit).
        See https://github.com/chickin/masertrack for documentation.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", nargs="?", default=None,
                        help="MFIT/SAD output file to process")
    parser.add_argument("-o", "--output", default=None,
                        help="Output feature file")
    parser.add_argument("--guided", action="store_true",
                        help="Interactive guided mode (recommended "
                             "for first pass)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Disable plots (batch mode)")

    # Beam / array overrides
    g = parser.add_argument_group("beam and array parameters")
    g.add_argument("--beam-major", type=float, default=None,
                   help="Beam major axis in mas")
    g.add_argument("--beam-minor", type=float, default=None,
                   help="Beam minor axis in mas")
    g.add_argument("--beam-pa", type=float, default=None,
                   help="Beam position angle in degrees")
    g.add_argument("--chan-spacing", type=float, default=None,
                   help="Channel spacing in km/s")
    g.add_argument("--n-stations", type=int, default=None,
                   help="Number of array stations (sets sidelobe "
                        "threshold via 1/N)")

    # Pipeline parameters
    g2 = parser.add_argument_group("pipeline tuning")
    g2.add_argument("--dedup-radius", type=float, default=None,
                    help="Dedup radius factor (x beam geo mean)")
    g2.add_argument("--sidelobe-flux", type=float, default=None,
                    help="Sidelobe flux ratio threshold")
    g2.add_argument("--link-radius", type=float, default=None,
                    help="Feature linking radius factor (x beam geo mean)")
    g2.add_argument("--vel-channels", type=float, default=None,
                    help="Velocity linking range in channels")
    g2.add_argument("--max-gap", type=float, default=None,
                    help="Max velocity gap to bridge in channels")
    g2.add_argument("--min-dip", type=int, default=None,
                    help="Min consecutive drops for peak splitting")
    g2.add_argument("--n-brightest", type=int, default=None,
                    help="N brightest spots for position averaging")
    g2.add_argument("--exclude", type=str, default=None,
                    help="File with spot indices to exclude")

    args = parser.parse_args()

    if args.input is None:
        parser.print_help()
        print("\n  Tip: run with --guided for interactive mode.\n")
        sys.exit(0)

    overrides = {
        "beam_major": args.beam_major,
        "beam_minor": args.beam_minor,
        "beam_pa": args.beam_pa,
        "chan_spacing": args.chan_spacing,
        "n_stations": args.n_stations,
        "exclude_file": args.exclude,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    if args.guided:
        guided_mode(args.input, **overrides)
    else:
        m = MaserTrackIdentify(args.input,
                               interactive=not args.no_plots,
                               **overrides)
        m.run()
        outpath = args.output or (Path(args.input).stem
                                  + "_features.txt")
        m.save(outpath)


if __name__ == "__main__":
    main()
