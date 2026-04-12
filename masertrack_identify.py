#!/usr/bin/env python3
"""
masertrack_identify.py - Part 1 of the masertrack pipeline
============================================================
Version 1.1

Convert VLBI spectral-line spot-fitting output into cleaned maser
feature catalogs.

Changelog
---------
  v1.1 (2026-04) - Code cleanup, unified commenting style, tutorial,
    output path respects --outdir for text files.
  v1.0.0 (2026-04-06) - First public release
    - Rainbow colormap (violet=blueshifted, red=redshifted)
    - Interactive HTML plots with linked sky maps
    - Click-to-exclude with toggle, spectrum panel, criteria markers
    - Summary dashboard with 9 diagnostic panels
    - Feature IDs start from 1; integer formatting in output files
    - Three error types: formal, scatter, weighted mean (>=3 spots)
    - Individual sidelobe criteria shown as guidance markers
    - --outdir, --include, --exclude flags
  v0.1-v0.9 (2026-04) - Internal development

Input: MFIT/SAD spot tables (Imai's csad format, 2004)
Output: Feature catalog (*_features.txt) + Spot table (*_spots.txt)
Plots: PNG per step + summary dashboard + interactive HTML

Credits: Gabor Orosz (design), Ross Burns (chain-linking, see Burns+2015
  MNRAS 453, 3163), Hiroshi Imai (MFIT/csad, 2004), Claude (code, 2026)
License: BSD-3-Clause
Repository: https://github.com/chickin/masertrack
"""

# =====================================================================
#  IMPORTS -- graceful failure with install instructions
# =====================================================================
from __future__ import annotations
import argparse, os, sys, textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__version__ = "1.1"

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required.  Install:  pip3 install numpy"); sys.exit(1)
try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required.  Install:  pip3 install pandas"); sys.exit(1)
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Note: matplotlib not found; no PNG plots.  pip3 install matplotlib")
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
try:
    from adjustText import adjust_text
    HAS_ADJTEXT = True
except ImportError:
    HAS_ADJTEXT = False

# =====================================================================
#  RAINBOW COLORMAP -- red=redshifted, blue/violet=blueshifted
# =====================================================================
# We use matplotlib's 'rainbow' colormap which goes violet(0) -> red(1).
# This maps naturally to Doppler convention: most negative V_LSR
# (blueshifted) gets violet/blue, most positive (redshifted) gets red.
CMAP_NAME = "rainbow"

# Plotly equivalent colorscale for HTML plots
PLOTLY_RAINBOW = [
    [0.0, "rgb(128,0,255)"],    # violet
    [0.15, "rgb(0,0,255)"],     # blue
    [0.30, "rgb(0,180,255)"],   # cyan
    [0.45, "rgb(0,220,0)"],     # green
    [0.60, "rgb(255,255,0)"],   # yellow
    [0.75, "rgb(255,165,0)"],   # orange
    [0.90, "rgb(255,50,0)"],    # red-orange
    [1.0, "rgb(200,0,0)"],     # red
]


# =====================================================================
#  DEFAULT PARAMETERS -- all documented, all overridable
# =====================================================================
DEFAULTS = {
    # Step 2: merge radius = this factor x geometric mean beam radius
    "dedup_radius_factor": 1.0,
    # Step 3: sidelobe zone boundaries (x beam major axis)
    "sidelobe_zone_inner": 0.5,
    "sidelobe_zone_outer": 3.0,
    # Step 3: max flux ratio (fraction of channel peak) for sidelobe
    "sidelobe_flux_ratio": 0.50,
    # Step 3: beam shape tolerance (fractional deviation from median)
    "sidelobe_beam_tol": 0.50,
    # Step 4: linking radius = this factor x geometric mean beam radius
    "link_radius_factor": 1.0,
    # Step 4: normal linking range in channel spacings
    "link_vel_channels": 1.5,
    # Step 4: max gap to bridge in channel spacings
    "max_vel_gap": 2.5,
    # Step 5: min consecutive drops+rises to trigger split
    "min_dip_channels": 2,
    # Step 6: number of brightest spots for position averaging
    "n_brightest": 3,
}


# =====================================================================
#  BEAM INFO CONTAINER
# =====================================================================
@dataclass
class BeamInfo:
    """Synthesized beam parameters.

    geo_mean = sqrt(bmaj * bmin) is the radius of a circle with the
    same area as the beam ellipse.  Used as the standard spatial search
    radius throughout the pipeline.
    """
    bmaj: float        # major axis in mas
    bmin: float        # minor axis in mas
    pa: float          # position angle in degrees
    chan_spacing: float  # velocity channel spacing in km/s

    @property
    def geo_mean(self) -> float:
        return float(np.sqrt(self.bmaj * self.bmin))

    def __str__(self):
        return (f"Beam {self.bmaj:.2f} x {self.bmin:.2f} mas, PA {self.pa:.1f} deg, "
                f"dv = {self.chan_spacing:.3f} km/s, r_geo = {self.geo_mean:.2f} mas")


# =====================================================================
#  MAIN PIPELINE CLASS
# =====================================================================
class MaserTrackIdentify:
    """Maser spot-to-feature identification pipeline.

    Parameters
    ----------
    filepath : str or Path
        Path to the MFIT/SAD output file (csad-converted format).
    show_plots : bool
        If True, open matplotlib windows.  Default False (saves PNGs
        only).  On macOS M1, set MPLBACKEND=TkAgg if windows freeze.
    save_plots : bool
        If True, save PNG and HTML plots.  Default True.
    outdir : str or Path
        Directory for output files.  Default: current directory.
    beam_major, beam_minor, beam_pa : float or None
        Override auto-detected beam (mas, degrees).
    chan_spacing : float or None
        Override auto-detected channel spacing (km/s).
    n_stations : int or None
        Number of array stations.  Sets sidelobe threshold via 1/N
        scaling (Steinberg 1972).
    exclude_file : str or Path or None
        Text file of spot indices to exclude (one per line).
    include_file : str or Path or None
        Text file of spot indices to force-include (overrides flags).
    """

    # Standard 16-column csad output format (Imai 2004)
    _FULL_COLS = ["vlsr", "x", "x_err", "y", "y_err", "I", "I_err",
                  "S", "S_err", "bmaj", "bmaj_err", "bmin", "bmin_err",
                  "pa", "pa_err", "rms"]

    def __init__(self, filepath, show_plots=False, save_plots=True,
                 outdir=".", beam_major=None, beam_minor=None, beam_pa=None,
                 chan_spacing=None, n_stations=None,
                 exclude_file=None, include_file=None):
        self.filepath = Path(filepath)
        self.show_plots = show_plots and HAS_MPL
        self.save_plots = save_plots
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self._user_bmaj = beam_major
        self._user_bmin = beam_minor
        self._user_pa = beam_pa
        self._user_dv = chan_spacing
        self.n_stations = n_stations
        self.exclude_indices = self._load_idx(exclude_file, "exclude") if exclude_file else set()
        self.include_indices = self._load_idx(include_file, "include") if include_file else set()

        # Pipeline data (set by each step)
        self.beam = None
        self.raw = self.deduped = self.flagged = None
        self.featured = self.split_df = self.features = None
        self._step = 0
        self._prefix = str(self.outdir / self.filepath.stem)
        self._stats = {}

    @staticmethod
    def _load_idx(path, label):
        """Load integer indices from a text file (one per line, # = comment)."""
        p = Path(path)
        if not p.exists():
            print(f"  Warning: {label} file {p} not found."); return set()
        idx = set()
        for ln in open(p):
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                try: idx.add(int(ln))
                except ValueError: pass
        print(f"  Loaded {len(idx)} {label} indices from {p}")
        return idx

    # =================================================================
    #  STEP 1 -- PARSE INPUT FILE
    # =================================================================
    def step1_parse(self) -> pd.DataFrame:
        """Read MFIT spot file and auto-detect beam / channel spacing.

        Handles 3–16 column formats.  Missing columns get sensible
        defaults (flux=1, errors=0, beam=NaN).  Prompts for beam
        if it cannot be auto-detected.
        """
        # Read numeric lines, skipping headers and comments
        rows, nc = [], None
        for ln in open(self.filepath):
            ln = ln.strip()
            if not ln or ln.startswith("#") or ln[0].isalpha():
                continue
            parts = ln.split()
            if len(parts) < 3:
                continue
            try:
                vals = [float(v) for v in parts]
            except ValueError:
                continue
            if nc is None:
                nc = len(vals)
            rows.append(vals)
        if not rows:
            raise RuntimeError(f"No data rows found in {self.filepath}")

        # Map columns based on count
        nc = nc or 16
        if nc >= 16:
            cn = self._FULL_COLS[:16]
        elif nc >= 7:
            cn = self._FULL_COLS[:nc]
        elif nc == 4:
            cn = ["vlsr", "x", "y", "I"]
        elif nc >= 3:
            cn = ["vlsr", "x", "y"]
        else:
            raise RuntimeError(f"Need >= 3 columns, got {nc}")

        rows = [r[:len(cn)] for r in rows]
        df = pd.DataFrame(rows, columns=cn).sort_values("vlsr").reset_index(drop=True)

        # Fill missing columns with safe defaults
        for c, d in [("I", 1.0), ("I_err", 0.0), ("x_err", 0.0),
                      ("y_err", 0.0), ("S_err", 0.0), ("rms", np.nan)]:
            if c not in df.columns:
                df[c] = d
                if c == "I":
                    print("  Note: no flux column; using uniform weights.")
        if "S" not in df.columns:
            df["S"] = df["I"]
        for c in ["bmaj", "bmaj_err", "bmin", "bmin_err", "pa", "pa_err"]:
            if c not in df.columns:
                df[c] = np.nan

        # Apply exclusion list
        if self.exclude_indices:
            before = len(df)
            df = df.drop(index=[i for i in self.exclude_indices
                                if i in df.index]).reset_index(drop=True)
            print(f"  Excluded {before - len(df)} spots.")

        # Determine beam parameters (user override > data > prompt)
        has_beam = df["bmaj"].notna().sum() > len(df) * 0.5
        bmaj = self._user_bmaj or (float(df["bmaj"].median()) if has_beam
                                    else self._ask("Beam major axis (mas)", 2.0))
        bmin = self._user_bmin or (float(df["bmin"].median()) if has_beam
                                    else self._ask("Beam minor axis (mas)", 0.75))
        pa = self._user_pa or (float(df["pa"].median()) if has_beam else 0.0)
        dv = self._user_dv or self._detect_dv(df["vlsr"].values)
        if dv is None or dv <= 0:
            dv = self._ask("Channel spacing (km/s)", 0.42)

        self.beam = BeamInfo(bmaj, bmin, pa, dv)
        self.raw = df
        self._step = 1
        self._stats["raw"] = len(df)
        self._stats["vrange"] = f"[{df['vlsr'].min():.1f}, {df['vlsr'].max():.1f}]"

        print(f"\nStep 1 - Parse: {len(df)} spots, "
              f"V = {self._stats['vrange']} km/s")
        print(f"  {self.beam}")
        if self.save_plots:
            self._plot_sky(df, "Step 1 - Raw spots", "_s1")
        return df

    def _ask(self, name, default):
        """Prompt user for a value, or use default in batch mode."""
        if not sys.stdin.isatty():
            print(f"  Using default {name} = {default}")
            return default
        try:
            v = input(f"  {name}? [{default}]: ").strip()
            return float(v) if v else default
        except (ValueError, EOFError):
            return default

    @staticmethod
    def _detect_dv(vlsr):
        """Find channel spacing as the mode of velocity differences."""
        uv = np.sort(np.unique(vlsr))
        if len(uv) < 2:
            return None
        dv = np.abs(np.diff(uv))
        dv = dv[dv > 0.001]
        if len(dv) == 0:
            return None
        bw = 0.01
        bins = np.arange(0, dv.max() + bw, bw)
        if len(bins) < 2:
            return float(np.median(dv))
        h, e = np.histogram(dv, bins=bins)
        return float(e[np.argmax(h)] + bw / 2)

    # =================================================================
    #  STEP 2 -- DEDUPLICATE
    # =================================================================
    def step2_deduplicate(self, radius_factor=None) -> pd.DataFrame:
        """Merge duplicate spots from overlapping MFIT subimage tiles.

        Within each velocity channel, spots closer than the merge
        radius are grouped.  The brightest spot's flux is kept (best
        Gaussian fit), position is flux-weighted average of duplicates.
        """
        if self.raw is None:
            self.step1_parse()
        rf = radius_factor or DEFAULTS["dedup_radius_factor"]
        radius = rf * self.beam.geo_mean
        df = self.raw.copy()
        merged = []

        for _, grp in df.groupby("vlsr"):
            grp = grp.sort_values("I", ascending=False)
            idx_list = grp.index.tolist()
            used = set()
            for i in idx_list:
                if i in used:
                    continue
                row = grp.loc[i].copy()
                # Find partners within merge radius
                partners = [j for j in idx_list if j != i and j not in used
                            and np.hypot(row["x"] - grp.loc[j, "x"],
                                         row["y"] - grp.loc[j, "y"]) < radius]
                if partners:
                    sub = grp.loc[[i] + partners]
                    w = sub["I"].values
                    for c in ["x", "y", "x_err", "y_err"]:
                        row[c] = np.average(sub[c].values, weights=w)
                    used.update(partners)
                merged.append(row)

        result = pd.DataFrame(merged).sort_values("vlsr").reset_index(drop=True)
        self.deduped = result
        self._step = 2
        self._stats["deduped"] = len(result)
        self._stats["merged"] = len(df) - len(result)

        print(f"\nStep 2 - Deduplicate (r < {radius:.2f} mas): "
              f"{len(df)} -> {len(result)} ({len(df) - len(result)} merged)")
        if self.save_plots:
            self._plot_sky(result, "Step 2 - Deduplicated", "_s2")
        return result

    # =================================================================
    #  STEP 3 -- SIDELOBE FLAGGING
    # =================================================================
    def step3_flag_sidelobes(self, zone_inner=None, zone_outer=None,
                             flux_ratio=None, beam_tol=None) -> pd.DataFrame:
        """Flag spots that are likely dirty-beam sidelobe artifacts.

        A spot is flagged if ALL criteria are met:
        1. Flux < threshold x brightest spot in same channel
        2. Position within sidelobe zone (0.5-3.0 x bmaj from peak)
        OR beam shape anomalous (>50% deviation from median beam)

        Threshold is set by 1/N + 10% where N = number of stations
        (Steinberg 1972 scaling).  Flagged spots are excluded from
        feature grouping but remain in the output for inspection.
        """
        if self.deduped is None:
            self.step2_deduplicate()

        zi = zone_inner or DEFAULTS["sidelobe_zone_inner"]
        zo = zone_outer or DEFAULTS["sidelobe_zone_outer"]
        fr = flux_ratio or DEFAULTS["sidelobe_flux_ratio"]
        bt = beam_tol or DEFAULTS["sidelobe_beam_tol"]

        # Array-adjusted threshold
        if self.n_stations and self.n_stations >= 2:
            fr = min(fr, 1.0 / self.n_stations + 0.10)
            print(f"  Array-adjusted threshold: {fr:.2f} (N={self.n_stations})")

        df = self.deduped.copy()
        df["sidelobe"] = False
        df["sidelobe_reason"] = ""
        mb, mn = self.beam.bmaj, self.beam.bmin
        has_beam = df["bmaj"].notna().sum() > len(df) * 0.3

        for _, grp in df.groupby("vlsr"):
            if len(grp) <= 1:
                continue
            pi = grp["I"].idxmax()
            pI, px, py = grp.loc[pi, "I"], grp.loc[pi, "x"], grp.loc[pi, "y"]

            for idx, row in grp.iterrows():
                if idx == pi or row["I"] >= fr * pI:
                    continue
                reasons = []
                sep = np.hypot(row["x"] - px, row["y"] - py)
                if zi * mb <= sep <= zo * mb:
                    reasons.append(f"zone({sep:.2f}mas)")
                if has_beam and not np.isnan(row["bmaj"]):
                    if not ((1-bt) < row["bmaj"]/mb < (1+bt) and
                            (1-bt) < row["bmin"]/mn < (1+bt)):
                        reasons.append("beam_shape")
                if reasons:
                    df.loc[idx, "sidelobe"] = True
                    df.loc[idx, "sidelobe_reason"] = "; ".join(reasons)

        # Apply include list (force un-flag)
        if self.include_indices:
            n_ov = sum(1 for i in self.include_indices
                       if i in df.index and df.loc[i, "sidelobe"])
            for i in self.include_indices:
                if i in df.index and df.loc[i, "sidelobe"]:
                    df.loc[i, "sidelobe"] = False
                    df.loc[i, "sidelobe_reason"] = "force_included"
            if n_ov:
                print(f"  Force-included {n_ov} spots from include list.")

        nf = int(df["sidelobe"].sum())
        self.flagged = df
        self._step = 3
        self._stats["sidelobes"] = nf

        print(f"\nStep 3 - Sidelobe flagging: {nf} spots flagged")
        print(f"  Criteria: flux < {fr:.0%} of peak AND "
              f"({zi}-{zo})x bmaj from peak")

        # Print table of flagged spots for exclude-file creation
        if nf > 0:
            fd = df[df["sidelobe"]]
            print(f"\n  {'ID':>5s}  {'VLSR':>10s}  {'X':>10s}  {'Y':>10s}  "
                  f"{'Flux':>8s}  Reason")
            print(f"  {'-'*5:>5s}  {'-'*10:>10s}  {'-'*10:>10s}  {'-'*10:>10s}  "
                  f"{'-'*8:>8s}  ------")
            for i, r in fd.iterrows():
                print(f"  {i:5d}  {r['vlsr']:10.3f}  {r['x']:10.3f}  "
                      f"{r['y']:10.3f}  {r['I']:8.3f}  {r['sidelobe_reason']}")
            print(f"\n  To un-flag: add spot ID to an --include file.")

        if self.save_plots:
            self._plot_step3(df)
        return df

    # =================================================================
    #  STEP 4 -- FEATURE GROUPING (CHAIN-LINKING)
    # =================================================================
    def step4_group_features(self, radius_factor=None, vel_channels=None,
                             max_vel_gap=None) -> pd.DataFrame:
        """Group spots into features via bidirectional chain-linking.

        Starting from the brightest unassigned spot, grow a chain in
        both velocity directions.  The reference position WALKS with
        the chain (allows gradual spatial drift across velocity).

        Adapted from Burns' Spots2Features.f90 with two extensions:
        - Bidirectional search (Burns only searched forward)
        - Gap bridging up to max_vel_gap channels
        """
        if self.flagged is None:
            self.step3_flag_sidelobes()

        rf = radius_factor or DEFAULTS["link_radius_factor"]
        mg = max_vel_gap or DEFAULTS["max_vel_gap"]
        max_sep = rf * self.beam.geo_mean
        max_dv = mg * self.beam.chan_spacing

        # Work with non-flagged spots only
        df = (self.flagged[~self.flagged["sidelobe"]].copy()
              .sort_values("vlsr").reset_index(drop=True))
        df["feature_id"] = -1
        fid = 1  # Feature IDs start from 1 for user-friendly display

        # Seed from brightest spots (most reliable positions)
        for seed in df.sort_values("I", ascending=False).index:
            if df.loc[seed, "feature_id"] != -1:
                continue
            df.loc[seed, "feature_id"] = fid

            # Grow chain in both velocity directions
            for direction in [+1, -1]:
                rv = df.loc[seed, "vlsr"]
                rx = df.loc[seed, "x"]
                ry = df.loc[seed, "y"]
                while True:
                    best_idx, best_dv = None, max_dv + 1
                    for j in df.index:
                        if df.loc[j, "feature_id"] != -1:
                            continue
                        dv = df.loc[j, "vlsr"] - rv
                        if (direction > 0 and dv <= 0) or (direction < 0 and dv >= 0):
                            continue
                        adv = abs(dv)
                        if adv > max_dv:
                            continue
                        if np.hypot(df.loc[j, "x"] - rx, df.loc[j, "y"] - ry) > max_sep:
                            continue
                        if adv < best_dv:
                            best_dv, best_idx = adv, j
                    if best_idx is None:
                        break
                    df.loc[best_idx, "feature_id"] = fid
                    rv = df.loc[best_idx, "vlsr"]
                    rx = df.loc[best_idx, "x"]
                    ry = df.loc[best_idx, "y"]
            fid += 1

        nf = df["feature_id"].nunique()
        self.featured = df
        self._step = 4
        self._stats["feat_linked"] = nf
        print(f"\nStep 4 - Feature grouping (r < {max_sep:.2f} mas, "
              f"gap <= {max_dv:.2f} km/s): {nf} features from {len(df)} spots")
        if self.save_plots:
            self._plot_features(df, "Step 4 - Features", "_s4")
        return df

    # =================================================================
    #  STEP 5 -- MULTI-PEAK SPLITTING
    # =================================================================
    def step5_split_multipeaks(self, min_dip=None) -> pd.DataFrame:
        """Split features whose flux profile has multiple peaks.

        Walks through each feature's flux vs velocity.  If flux drops
        for min_dip consecutive channels then rises for min_dip
        consecutive channels, split at the valley.  From Burns'
        MultiPeakCheck.f90 (where min_dip is hardcoded to 2).
        """
        if self.featured is None:
            self.step4_group_features()
        md = min_dip or DEFAULTS["min_dip_channels"]
        df = self.featured.copy()
        nid = df["feature_id"].max() + 1
        ns = 0

        for fid in df["feature_id"].unique():
            grp = df.loc[df["feature_id"] == fid].sort_values("vlsr")
            if len(grp) < 2 * md + 1:
                continue
            fl = grp["I"].values
            ix = grp.index.values
            up, down = 0, 0
            for k in range(1, len(fl)):
                if fl[k] > fl[k-1]:
                    up += 1
                elif fl[k] < fl[k-1]:
                    down += 1
                if down >= md and up >= md and fl[k] > fl[k-1]:
                    df.loc[ix[k:], "feature_id"] = nid
                    nid += 1
                    ns += 1
                    up, down = 0, 0

        nf = df["feature_id"].nunique()
        self.split_df = df
        self._step = 5
        self._stats.update({"splits": ns, "feat_split": nf})
        print(f"\nStep 5 - Multi-peak split (min_dip={md}): "
              f"{ns} split(s) -> {nf} features")
        if self.save_plots and ns > 0:
            self._plot_features(df, "Step 5 - After split", "_s5")
        return df

    # =================================================================
    #  STEP 6 -- SUMMARIZE FEATURES
    # =================================================================
    def step6_summarize(self, n_brightest=None) -> pd.DataFrame:
        """Compute feature catalog from the N brightest spots.

        Two velocity columns:
          vlsr_peak - brightest channel (track this for parallax)
          vlsr_avg  - flux-weighted centroid (for kinematics)

        Three error columns per axis:
          formal  - weighted average of MFIT fitting errors
          scatter - std dev of all spot positions (spatial extent)
          wmean   - weighted standard error of the mean position
                    (best estimate of centroid uncertainty)
                    For features with <3 spots, wmean = formal
                    because scatter-based errors are unreliable
                    with so few points.
        """
        if self.split_df is None:
            self.step5_split_multipeaks()
        nb = n_brightest or DEFAULTS["n_brightest"]
        df = self.split_df
        rows = []

        for fid, grp in df.groupby("feature_id"):
            grp = grp.sort_values("I", ascending=False)
            pk = grp.iloc[0]  # brightest spot

            # Use top-N or all spots if fewer exist
            n_use = min(nb, len(grp))
            top = grp.head(n_use)
            w = top["I"].values
            if w.sum() == 0:
                w = np.ones(len(w))

            # Flux-weighted position and velocity
            xa = np.average(top["x"].values, weights=w)
            ya = np.average(top["y"].values, weights=w)
            va = np.average(top["vlsr"].values, weights=w)

            # Formal error: weighted average of MFIT errors
            xef = np.average(top["x_err"].values, weights=w)
            yef = np.average(top["y_err"].values, weights=w)

            # Empirical scatter: std dev of ALL spot positions
            if len(grp) >= 3:
                xsc = float(grp["x"].std())
                ysc = float(grp["y"].std())
            else:
                xsc, ysc = xef, yef

            # Weighted standard error of the mean
            # Only meaningful with >= 3 spots; otherwise use formal
            if len(top) >= 3 and w.sum() > 0:
                wn = w / w.sum()
                dn = 1 - np.sum(wn**2)
                if dn > 0:
                    xwm = np.sqrt(np.sum(wn**2 * (top["x"].values - xa)**2) / dn)
                    ywm = np.sqrt(np.sum(wn**2 * (top["y"].values - ya)**2) / dn)
                    if np.isnan(xwm) or xwm == 0:
                        xwm = xef
                    if np.isnan(ywm) or ywm == 0:
                        ywm = yef
                else:
                    xwm, ywm = xef, yef
            else:
                # < 3 spots: can't compute meaningful scatter
                xwm, ywm = xef, yef

            rows.append({
                "feature_id": fid,
                "vlsr_peak": pk["vlsr"],
                "vlsr_avg": va,
                "vlsr_min": grp["vlsr"].min(),
                "vlsr_max": grp["vlsr"].max(),
                "vlsr_width": grp["vlsr"].max() - grp["vlsr"].min() + self.beam.chan_spacing,
                "n_spots": len(grp),
                "n_channels": grp["vlsr"].nunique(),
                "x": xa, "y": ya,
                "x_err_formal": xef, "y_err_formal": yef,
                "x_err_scatter": xsc, "y_err_scatter": ysc,
                "x_err_wmean": xwm, "y_err_wmean": ywm,
                "I_peak": pk["I"], "S_peak": pk["S"],
                "rms": pk["rms"],
                "snr": pk["I"] / pk["rms"] if pk["rms"] > 0 else np.nan,
            })

        feat = pd.DataFrame(rows).sort_values("vlsr_peak").reset_index(drop=True)
        self.features = feat
        self._step = 6
        nm = int((feat["n_channels"] >= 3).sum())
        self._stats.update({"feat_final": len(feat), "multi_ch": nm})

        print(f"\nStep 6 - Summarize (top-{nb}): "
              f"{len(feat)} features, {nm} with >= 3 channels")
        if self.save_plots:
            self._plot_step6(feat, df)
            self._plot_summary()
            if HAS_PLOTLY:
                self._plot_html(feat, df)
        return feat

    # =================================================================
    #  RUN ALL / SAVE
    # =================================================================
    def run(self, **kw):
        """Execute the full pipeline with optional parameter overrides."""
        self.step1_parse()
        self.step2_deduplicate(kw.get("radius_factor"))
        self.step3_flag_sidelobes(kw.get("zone_inner"), kw.get("zone_outer"),
                                  kw.get("flux_ratio"), kw.get("beam_tol"))
        self.step4_group_features(kw.get("link_radius_factor"),
                                  kw.get("vel_channels"), kw.get("max_vel_gap"))
        self.step5_split_multipeaks(kw.get("min_dip"))
        self.step6_summarize(kw.get("n_brightest"))
        return self.features

    def save(self, outpath, save_spots=True):
        """Write feature catalog and spot table as fixed-width text."""
        if self.features is None:
            raise RuntimeError("Run pipeline before saving.")
        op = Path(outpath)
        self._write_fw(self.features, op,
                       f"Feature catalog from {self.filepath.name}\n"
                       f"# Beam: {self.beam}\n"
                       f"# Pipeline: masertrack v{__version__}\n")
        if save_spots and self.split_df is not None:
            stem = op.stem.replace("_features", "")
            sp = op.with_name(stem + "_spots" + op.suffix)
            self._write_fw(self.split_df, sp,
                           f"Spot table from {self.filepath.name}\n"
                           f"# Beam: {self.beam}\n")

    @staticmethod
    def _write_fw(df, path, header=""):
        """Write DataFrame as space-padded fixed-width text."""
        # Ensure integer columns are stored as int (not float)
        int_cols = ["feature_id", "n_spots", "n_channels"]
        for c in int_cols:
            if c in df.columns:
                df[c] = df[c].astype(int)

        lines = [f"# {h}" if not h.startswith("#") else h
                 for h in header.strip().split("\n")]
        cw = {}
        for c in df.columns:
            if c in int_cols or df[c].dtype in [np.int64, np.int32, int]:
                fm = [f"{int(v):8d}" for v in df[c]]
            elif df[c].dtype in [np.float64, float]:
                fm = [f"{v:12.4f}" for v in df[c]]
            else:
                fm = [str(v) for v in df[c]]
            cw[c] = max(len(c), max((len(s) for s in fm), default=1)) + 1
        lines.append("# " + "".join(f"{c:>{cw[c]}s}" for c in df.columns))
        for _, row in df.iterrows():
            p = []
            for c in df.columns:
                w, v = cw[c], row[c]
                if c in int_cols:
                    p.append(f"{int(v):{w}d}")
                elif isinstance(v, (float, np.floating)):
                    p.append(f"{'nan':>{w}s}" if np.isnan(v) else f"{v:{w}.4f}")
                elif isinstance(v, (int, np.integer)):
                    p.append(f"{v:{w}d}")
                elif isinstance(v, (bool, np.bool_)):
                    p.append(f"{str(v):>{w}s}")
                else:
                    p.append(f"{str(v):>{w}s}")
            lines.append("  " + "".join(p))
        open(path, "w").write("\n".join(lines) + "\n")
        print(f"  Saved: {path} ({len(df)} rows)")

    # =================================================================
    #  MATPLOTLIB PLOTS (PNG)
    # =================================================================
    def _fsz(self, flux, lo=20, hi=150):
        """Map flux values to marker sizes for visibility."""
        f = np.array(flux, dtype=float)
        if f.max() > f.min():
            n = (f - f.min()) / (f.max() - f.min())
        else:
            n = np.full_like(f, 0.5)
        return lo + n * hi

    def _savefig(self, fig, sfx):
        path = f"{self._prefix}{sfx}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        print(f"  Plot: {path}")

    def _plot_sky(self, df, title, sfx):
        """Two-panel plot: sky map (left) and spectrum (right)."""
        if not HAS_MPL:
            return
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
        sz = self._fsz(df["I"])
        sc = a1.scatter(df["x"], df["y"], c=df["vlsr"], s=sz,
                        cmap=CMAP_NAME, edgecolors="k", linewidths=0.2, alpha=0.85)
        a1.set_xlabel("RA offset (mas)"); a1.set_ylabel("Dec offset (mas)")
        a1.invert_xaxis(); a1.set_aspect("equal")
        plt.colorbar(sc, ax=a1, label="V_LSR (km/s)")
        a2.scatter(df["vlsr"], df["I"], c=df["vlsr"], s=sz,
                   cmap=CMAP_NAME, edgecolors="k", linewidths=0.2, alpha=0.85)
        a2.set_xlabel("V_LSR (km/s)"); a2.set_ylabel("Flux (Jy/beam)")
        fig.suptitle(f"{title} ({len(df)} spots)", fontsize=13)
        plt.tight_layout(); self._savefig(fig, sfx)

    def _plot_step3(self, df):
        """Sidelobe plot showing individual criteria as colored markers.

        In addition to the red X for actually flagged spots (AND logic),
        shows what each criterion would flag individually (OR logic)
        as colored rings, so the user can evaluate the thresholds:
          Blue ring:   in sidelobe spatial zone
          Green ring:  below flux ratio threshold
          Orange ring: anomalous beam shape
          Red X:       flagged (all applicable criteria met)
        """
        if not HAS_MPL:
            return
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7))
        cl = df[~df["sidelobe"]]
        dr = df[df["sidelobe"]]
        sz = self._fsz(cl["I"])
        sc = a1.scatter(cl["x"], cl["y"], c=cl["vlsr"], s=sz,
                        cmap=CMAP_NAME, edgecolors="k", linewidths=0.2,
                        alpha=0.7, zorder=2)

        # Show individual criteria as guidance markers
        # Recompute per-criterion flags for visualization
        mb = self.beam.bmaj
        mn = self.beam.bmin
        zi = DEFAULTS["sidelobe_zone_inner"]
        zo = DEFAULTS["sidelobe_zone_outer"]
        fr = DEFAULTS["sidelobe_flux_ratio"]
        bt = DEFAULTS["sidelobe_beam_tol"]
        if self.n_stations and self.n_stations >= 2:
            fr = min(fr, 1.0 / self.n_stations + 0.10)
        has_beam = df["bmaj"].notna().sum() > len(df) * 0.3

        zone_spots = []    # in spatial zone
        flux_spots = []    # below flux threshold
        beam_spots = []    # anomalous beam

        for _, grp in df.groupby("vlsr"):
            if len(grp) <= 1:
                continue
            pi = grp["I"].idxmax()
            pI, px, py = grp.loc[pi, "I"], grp.loc[pi, "x"], grp.loc[pi, "y"]
            for idx, row in grp.iterrows():
                if idx == pi:
                    continue
                sep = np.hypot(row["x"] - px, row["y"] - py)
                if zi * mb <= sep <= zo * mb:
                    zone_spots.append(idx)
                if row["I"] < fr * pI:
                    flux_spots.append(idx)
                if has_beam and not np.isnan(row["bmaj"]):
                    if not ((1-bt) < row["bmaj"]/mb < (1+bt) and
                            (1-bt) < row["bmin"]/mn < (1+bt)):
                        beam_spots.append(idx)

        # Plot individual criteria as faint rings (guidance only)
        for idxlist, color, label, marker in [
            (zone_spots, "royalblue", "Spatial zone", "o"),
            (flux_spots, "limegreen", "Below flux threshold", "s"),
            (beam_spots, "orange", "Beam shape anomaly", "D"),
        ]:
            if idxlist:
                sub = df.loc[[i for i in idxlist if i in df.index]]
                a1.scatter(sub["x"], sub["y"], facecolors="none",
                           edgecolors=color, s=180, linewidths=1.5,
                           marker=marker, alpha=0.5, zorder=2.5,
                           label=f"{label} ({len(sub)})")

        # Flagged spots (AND logic) as red X
        if len(dr) > 0:
            a1.scatter(dr["x"], dr["y"], c="red", s=150, marker="x",
                       linewidths=2.5, zorder=3, label=f"FLAGGED ({len(dr)})")
            for i, r in dr.iterrows():
                a1.annotate(str(i), (r["x"], r["y"]), fontsize=8,
                            color="red", fontweight="bold",
                            xytext=(5, 5), textcoords="offset points")

        a1.set_xlabel("RA offset (mas)"); a1.set_ylabel("Dec offset (mas)")
        a1.invert_xaxis(); a1.set_aspect("equal")
        a1.legend(fontsize=7, loc="lower left")
        plt.colorbar(sc, ax=a1, label="V_LSR (km/s)")

        # Right panel: spectrum
        a2.scatter(cl["vlsr"], cl["I"], c=cl["vlsr"], s=sz,
                   cmap=CMAP_NAME, edgecolors="k", linewidths=0.2, alpha=0.7)
        if len(dr) > 0:
            a2.scatter(dr["vlsr"], dr["I"], c="red", s=150,
                       marker="x", linewidths=2.5)
            for i, r in dr.iterrows():
                a2.annotate(str(i), (r["vlsr"], r["I"]), fontsize=8,
                            color="red", fontweight="bold",
                            xytext=(5, 5), textcoords="offset points")
        a2.set_xlabel("V_LSR (km/s)"); a2.set_ylabel("Flux (Jy/beam)")

        fig.suptitle(f"Step 3 - Sidelobes ({int(df['sidelobe'].sum())} flagged, "
                     f"colored rings = individual criteria)", fontsize=12)
        plt.tight_layout(); self._savefig(fig, "_s3")

    def _plot_features(self, df, title, sfx):
        """Feature plot with spots colored by feature, connected by lines."""
        if not HAS_MPL:
            return
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
        nf = df["feature_id"].nunique()
        cm = matplotlib.colormaps.get_cmap("tab20")
        for fid, grp in df.groupby("feature_id"):
            c = cm(fid % 20)
            sz = self._fsz(grp["I"], 10, 80)
            a1.scatter(grp["x"], grp["y"], c=[c], s=sz, edgecolors="k",
                       linewidths=0.2, alpha=0.8)
            # Connect spots within feature on sky
            gs = grp.sort_values("vlsr")
            if len(gs) > 1:
                a1.plot(gs["x"], gs["y"], "-", c=c, alpha=0.3, lw=0.8)
            # Spectrum: connected
            a2.scatter(gs["vlsr"], gs["I"], c=[c], s=sz, edgecolors="k",
                       linewidths=0.2, alpha=0.8)
            a2.plot(gs["vlsr"], gs["I"], "-", c=c, alpha=0.4, lw=1)
        a1.set_xlabel("RA offset (mas)"); a1.set_ylabel("Dec offset (mas)")
        a1.invert_xaxis(); a1.set_aspect("equal")
        a2.set_xlabel("V_LSR (km/s)"); a2.set_ylabel("Flux (Jy/beam)")
        fig.suptitle(f"{title} ({nf} features)", fontsize=13)
        plt.tight_layout(); self._savefig(fig, sfx)

    def _plot_step6(self, feat, spots):
        """Summary feature map with non-overlapping labels."""
        if not HAS_MPL:
            return
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7))
        cm = matplotlib.colormaps.get_cmap("tab20")

        # Background: spots colored by feature with connecting lines
        for fid, grp in spots.groupby("feature_id"):
            c = cm(fid % 20)
            a1.scatter(grp["x"], grp["y"], c=[c], s=6, alpha=0.25, zorder=1)
            gs = grp.sort_values("vlsr")
            if len(gs) > 1:
                a1.plot(gs["x"], gs["y"], "-", c=c, alpha=0.15, lw=0.5)

        # Foreground: feature markers
        fsz = self._fsz(feat["I_peak"], 40, 250)
        sc = a1.scatter(feat["x"], feat["y"], c=feat["vlsr_peak"], s=fsz,
                        cmap=CMAP_NAME, edgecolors="k", linewidths=0.8,
                        zorder=3, alpha=0.9)

        # Labels: top 10 brightest multi-channel features only
        # Place labels at spot positions, clipped to plot area
        multi = feat[feat["n_channels"] >= 2].nlargest(10, "I_peak")
        for _, f in multi.iterrows():
            txt = a1.text(f["x"], f["y"],
                          f" F{int(f['feature_id'])} {f['vlsr_peak']:.1f}",
                          fontsize=5.5, fontweight="bold",
                          va="bottom", ha="left", clip_on=True,
                          bbox=dict(boxstyle="round,pad=0.1", fc="white",
                                    ec="gray", alpha=0.6, lw=0.5))

        a1.set_xlabel("RA offset (mas)"); a1.set_ylabel("Dec offset (mas)")
        a1.invert_xaxis(); a1.set_aspect("equal")
        plt.colorbar(sc, ax=a1, label="V_LSR (km/s)")
        a1.set_title("Feature map (size = flux, color = velocity)")

        # Right: spectrum bars colored by feature
        for _, f in feat.iterrows():
            a2.bar(f["vlsr_peak"], f["I_peak"], width=f["vlsr_width"],
                   alpha=0.5, color=cm(int(f["feature_id"]) % 20),
                   edgecolor="k", linewidth=0.3)
        a2.set_xlabel("V_LSR (km/s)"); a2.set_ylabel("Flux (Jy/beam)")
        nm = int((feat["n_channels"] >= 3).sum())
        a2.set_title(f"{len(feat)} features ({nm} with >= 3 ch)")
        fig.suptitle("Step 6 - Feature catalog", fontsize=13)
        plt.tight_layout(); self._savefig(fig, "_s6")

    def _plot_summary(self):
        """Pipeline summary dashboard (9 panels + info text)."""
        if not HAS_MPL:
            return
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        feat = self.features
        spots = self.split_df
        cm = matplotlib.colormaps.get_cmap("tab20")

        # ---- ROW 0: Overview ----

        # (0,0) Pipeline funnel
        ax = axes[0, 0]
        vals = [self._stats.get("raw", 0), self._stats.get("deduped", 0),
                self._stats.get("deduped", 0) - self._stats.get("sidelobes", 0),
                len(spots) if spots is not None else 0]
        n_feat = self._stats.get("feat_final", "?")
        labels = ["Raw", "Deduped", "Unflagged", f"Grouped ({n_feat} feat)"]
        colors = ["#4e79a7", "#59a14f", "#edc948", "#e15759"]
        ax.barh(labels[::-1], vals[::-1], color=colors[::-1])
        for i, v in enumerate(vals[::-1]):
            ax.text(v + 1, i, str(v), va="center", fontweight="bold")
        ax.set_xlabel("Number of spots")
        ax.set_title("Spot count through pipeline")

        # (0,1) Sky map with feature colors + spot counts
        ax = axes[0, 1]
        if feat is not None and spots is not None:
            # Show spots colored by feature
            for fid, grp in spots.groupby("feature_id"):
                c = cm(fid % 20)
                ax.scatter(grp["x"], grp["y"], c=[c], s=8, alpha=0.4)
            # Feature markers
            sc = ax.scatter(feat["x"], feat["y"], c=feat["vlsr_peak"],
                            s=self._fsz(feat["I_peak"], 30, 200),
                            cmap=CMAP_NAME, edgecolors="k", linewidths=0.6,
                            zorder=3)
            ax.invert_xaxis(); ax.set_aspect("equal")
            try:
                ax.set_box_aspect(1)  # force square axes
            except AttributeError:
                pass  # matplotlib < 3.3
            plt.colorbar(sc, ax=ax, label="V_LSR (km/s)")
        ax.set_xlabel("RA (mas)"); ax.set_ylabel("Dec (mas)")
        ax.set_title("Feature positions (spots + features)")

        # (0,2) Spectrum with feature-colored bars
        ax = axes[0, 2]
        if feat is not None:
            for _, f in feat.iterrows():
                c = cm(int(f["feature_id"]) % 20)
                ax.bar(f["vlsr_peak"], f["I_peak"], width=f["vlsr_width"],
                       alpha=0.6, color=c, edgecolor="k", linewidth=0.3)
        ax.set_xlabel("V_LSR (km/s)"); ax.set_ylabel("Flux (Jy/beam)")
        ax.set_title("Feature spectrum")

        # ---- ROW 1: Error comparisons (>= 3 channels only) ----
        if feat is not None:
            m3 = feat[feat["n_channels"] >= 3]
        else:
            m3 = pd.DataFrame()

        def _error_plot(ax, xcol, ycol, xlabel, ylabel, title):
            if len(m3) > 0:
                ax.scatter(m3[xcol], m3[ycol], s=40, alpha=0.7,
                           edgecolors="k", linewidths=0.3,
                           c=m3["n_channels"], cmap="viridis")
                lm = max(m3[xcol].max(), m3[ycol].max()) * 1.3
                if lm > 0:
                    ax.plot([0, lm], [0, lm], "k--", alpha=0.3)
                    ax.set_xlim(0, lm); ax.set_ylim(0, lm)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            ax.set_title(title)

        # (1,0) Scatter vs Formal
        _error_plot(axes[1, 0], "x_err_formal", "x_err_scatter",
                    "Formal error (mas)", "Scatter (mas)",
                    "Scatter vs Formal (RA)")

        # (1,1) WMean vs Formal
        _error_plot(axes[1, 1], "x_err_formal", "x_err_wmean",
                    "Formal error (mas)", "Weighted mean error (mas)",
                    "WMean vs Formal (RA)")

        # (1,2) Scatter vs WMean
        _error_plot(axes[1, 2], "x_err_wmean", "x_err_scatter",
                    "Weighted mean error (mas)", "Scatter (mas)",
                    "Scatter vs WMean (RA)")

        # ---- ROW 2: Feature statistics ----

        # (2,0) Channels per feature
        ax = axes[2, 0]
        if feat is not None and len(feat):
            mx = int(feat["n_channels"].max())
            ax.hist(feat["n_channels"], bins=range(1, mx + 2),
                    edgecolor="k", alpha=0.7, color="#4e79a7")
        ax.set_xlabel("Channels per feature"); ax.set_ylabel("Count")
        ax.set_title("Feature size distribution")

        # (2,1) Flux distribution
        ax = axes[2, 1]
        if feat is not None and len(feat):
            ax.hist(np.log10(feat["I_peak"]), bins=20,
                    edgecolor="k", alpha=0.7, color="#59a14f")
        ax.set_xlabel("log10(Flux [Jy/beam])"); ax.set_ylabel("Count")
        ax.set_title("Flux distribution")

        # (2,2) Velocity distribution
        ax = axes[2, 2]
        if feat is not None and len(feat):
            ax.hist(feat["vlsr_peak"], bins=25,
                    edgecolor="k", alpha=0.7, color="#e15759")
        ax.set_xlabel("V_LSR (km/s)"); ax.set_ylabel("Count")
        ax.set_title("Velocity distribution")

        # Summary text at the bottom of the figure
        info = (f"masertrack v{__version__} | {self.filepath.name} | "
                f"{self.beam} | "
                f"Raw: {self._stats.get('raw','?')} | "
                f"Deduped: {self._stats.get('deduped','?')} "
                f"(-{self._stats.get('merged','?')}) | "
                f"Sidelobes: {self._stats.get('sidelobes','?')} | "
                f"Features: {self._stats.get('feat_final','?')} "
                f"({self._stats.get('multi_ch','?')} multi-ch) | "
                f"Errors: formal=MFIT, scatter=stddev, "
                f"wmean=weighted SE (>=3 spots only)")
        fig.text(0.5, 0.01, info, ha="center", fontsize=7.5,
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))

        fig.suptitle("masertrack_identify - Summary Dashboard",
                     fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0.04, 1, 0.97])
        self._savefig(fig, "_summary")

    # =================================================================
    #  PLOTLY INTERACTIVE HTML
    # =================================================================
    def _plot_html(self, feat, spots):
        """Interactive HTML plot with rainbow colors and hover info."""
        if not HAS_PLOTLY:
            return

        # Get velocity range for consistent color mapping
        vmin = min(spots["vlsr"].min(), feat["vlsr_peak"].min())
        vmax = max(spots["vlsr"].max(), feat["vlsr_peak"].max())

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Spots - Sky map (hover for info)",
                            "Spots - Spectrum",
                            "Features - Sky map (linked zoom with spots)",
                            "Features - Flux profile"),
            vertical_spacing=0.08, horizontal_spacing=0.08,
        )
        # Link sky map axes: zooming one sky map zooms the other
        # Col 1 rows 1&2 share RA (x) and Dec (y)
        fig.update_xaxes(matches="x", row=2, col=1)
        fig.update_yaxes(matches="y", row=2, col=1)

        # --- Spot panels (rows 1) ---
        for fid, grp in spots.groupby("feature_id"):
            ht = [f"<b>Spot {i}</b><br>V: {r['vlsr']:.3f} km/s<br>"
                  f"X: {r['x']:.3f} mas<br>Y: {r['y']:.3f} mas<br>"
                  f"Flux: {r['I']:.3f} Jy/beam<br>Feature: {fid}"
                  for i, r in grp.iterrows()]

            fig.add_trace(go.Scatter(
                x=grp["x"], y=grp["y"], mode="markers",
                marker=dict(size=np.clip(grp["I"] * 4 + 5, 5, 30),
                            color=grp["vlsr"], colorscale=PLOTLY_RAINBOW,
                            cmin=vmin, cmax=vmax,
                            line=dict(width=0.5, color="black")),
                text=ht, hoverinfo="text",
                name=f"F{fid}", legendgroup=f"f{fid}",
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=grp.sort_values("vlsr")["vlsr"],
                y=grp.sort_values("vlsr")["I"],
                mode="markers+lines",
                marker=dict(size=np.clip(grp["I"] * 4 + 5, 5, 25),
                            color=grp["vlsr"], colorscale=PLOTLY_RAINBOW,
                            cmin=vmin, cmax=vmax,
                            line=dict(width=0.5, color="black")),
                line=dict(width=1, dash="dot", color="gray"),
                text=ht, hoverinfo="text",
                name=f"F{fid}", legendgroup=f"f{fid}",
                showlegend=False,
            ), row=1, col=2)

        # --- Feature panels (row 2) — per-feature traces linked to legend ---
        for _, f in feat.iterrows():
            fid = int(f["feature_id"])
            vnorm = (f["vlsr_peak"] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            ci = min(int(vnorm * (len(PLOTLY_RAINBOW)-1)), len(PLOTLY_RAINBOW)-2)
            fcolor = PLOTLY_RAINBOW[ci][1]

            fht_single = (f"<b>Feature {fid}</b><br>"
                          f"V_peak: {f['vlsr_peak']:.3f} km/s<br>"
                          f"V_avg: {f['vlsr_avg']:.3f} km/s<br>"
                          f"X: {f['x']:.3f} +/- {f['x_err_wmean']:.4f} mas<br>"
                          f"Y: {f['y']:.3f} +/- {f['y_err_wmean']:.4f} mas<br>"
                          f"Flux: {f['I_peak']:.3f} Jy/beam<br>"
                          f"Channels: {int(f['n_channels'])}<br>"
                          f"SNR: {f['snr']:.1f}")

            # Feature sky map marker (linked to same legendgroup as spots)
            fig.add_trace(go.Scatter(
                x=[f["x"]], y=[f["y"]], mode="markers+text",
                marker=dict(size=max(10, f["I_peak"] * 3 + 10),
                            color=fcolor,
                            line=dict(width=1.5, color="black")),
                text=[f"{f['vlsr_peak']:.1f}"],
                textposition="top center", textfont=dict(size=8),
                hovertext=[fht_single], hoverinfo="text",
                name=f"F{fid}", legendgroup=f"f{fid}",
                showlegend=False,
            ), row=2, col=1)

            # Feature flux bar (linked to same legendgroup)
            bcolor = fcolor.replace("rgb", "rgba").replace(")", ",0.6)")
            fig.add_trace(go.Bar(
                x=[f["vlsr_peak"]], y=[f["I_peak"]], width=[f["vlsr_width"]],
                marker_color=bcolor, marker_line=dict(width=0.5, color="black"),
                hovertext=f"<b>F{fid}</b>: {f['vlsr_peak']:.1f} km/s, "
                          f"{f['I_peak']:.2f} Jy, {int(f['n_channels'])} ch",
                hoverinfo="text",
                name=f"F{fid}", legendgroup=f"f{fid}",
                showlegend=False,
            ), row=2, col=2)

        # Axis labels and formatting
        fig.update_xaxes(title_text="RA offset (mas)", autorange="reversed", row=1, col=1)
        fig.update_yaxes(title_text="Dec offset (mas)", row=1, col=1)
        fig.update_xaxes(title_text="V_LSR (km/s)", row=1, col=2)
        fig.update_yaxes(title_text="Flux (Jy/beam)", row=1, col=2)
        fig.update_xaxes(title_text="RA offset (mas)", autorange="reversed", row=2, col=1)
        fig.update_yaxes(title_text="Dec offset (mas)", row=2, col=1)
        fig.update_xaxes(title_text="V_LSR (km/s)", row=2, col=2)
        fig.update_yaxes(title_text="Flux (Jy/beam)", row=2, col=2)

        # Add explanation annotation
        fig.add_annotation(
            text=(f"masertrack v{__version__} | {self.filepath.name}<br>"
                  f"{len(feat)} features, {len(spots)} spots | "
                  f"Click legend to toggle features | Hover for details"),
            xref="paper", yref="paper", x=0.5, y=-0.06,
            showarrow=False, font=dict(size=10, color="gray"),
        )

        fig.update_layout(
            title=f"masertrack - {self.filepath.name}",
            height=950, width=1450, hovermode="closest",
        )

        hp = f"{self._prefix}_interactive.html"
        fig.write_html(hp, include_plotlyjs="cdn")
        print(f"  Interactive HTML: {hp}")

    # =================================================================
    #  CLICK-TO-EXCLUDE (matplotlib interactive)
    # =================================================================
    def interactive_exclude(self, path="exclude_list.txt"):
        """Open matplotlib window for click-to-exclude with spectrum.

        Two panels: sky map (left, clickable) and spectrum (right).
        Colored rings show individual flagging criteria as guidance:
          Blue circle:  in sidelobe spatial zone
          Green square: below flux ratio threshold
          Orange diamond: anomalous beam shape
          Red X: flagged by combined AND logic

        Click a spot on the sky map to toggle exclusion (orange star).
        Click again to un-exclude.  CLOSE THE WINDOW to continue.
        The exclude file is written when the window is closed.
        """
        if not HAS_MPL:
            print("Error: matplotlib required."); return
        if self.flagged is None:
            print("Error: run through step 3 first."); return

        df = self.flagged.copy()
        excluded = set()

        fig, (ax_sky, ax_spec) = plt.subplots(1, 2, figsize=(16, 8))
        cl = df[~df["sidelobe"]]
        dr = df[df["sidelobe"]]

        # --- Sky map (left, clickable) ---
        sz = self._fsz(cl["I"], 15, 120)
        sc = ax_sky.scatter(cl["x"], cl["y"], c=cl["vlsr"], s=sz,
                            cmap=CMAP_NAME, edgecolors="k", linewidths=0.3,
                            picker=5, zorder=2)

        # Show individual criteria as colored rings (guidance)
        mb, mn = self.beam.bmaj, self.beam.bmin
        zi, zo = DEFAULTS["sidelobe_zone_inner"], DEFAULTS["sidelobe_zone_outer"]
        fr = DEFAULTS["sidelobe_flux_ratio"]
        bt = DEFAULTS["sidelobe_beam_tol"]
        if self.n_stations and self.n_stations >= 2:
            fr = min(fr, 1.0 / self.n_stations + 0.10)
        has_bm = df["bmaj"].notna().sum() > len(df) * 0.3

        for _, grp in df.groupby("vlsr"):
            if len(grp) <= 1:
                continue
            pi = grp["I"].idxmax()
            pI, px, py = grp.loc[pi, "I"], grp.loc[pi, "x"], grp.loc[pi, "y"]
            for idx, row in grp.iterrows():
                if idx == pi:
                    continue
                sep = np.hypot(row["x"] - px, row["y"] - py)
                if zi * mb <= sep <= zo * mb:
                    ax_sky.scatter(row["x"], row["y"], facecolors="none",
                                   edgecolors="royalblue", s=200, linewidths=1,
                                   marker="o", alpha=0.4, zorder=2.3)
                if row["I"] < fr * pI:
                    ax_sky.scatter(row["x"], row["y"], facecolors="none",
                                   edgecolors="limegreen", s=220, linewidths=1,
                                   marker="s", alpha=0.4, zorder=2.3)
                if has_bm and not np.isnan(row["bmaj"]):
                    if not ((1-bt) < row["bmaj"]/mb < (1+bt) and
                            (1-bt) < row["bmin"]/mn < (1+bt)):
                        ax_sky.scatter(row["x"], row["y"], facecolors="none",
                                       edgecolors="orange", s=240, linewidths=1,
                                       marker="D", alpha=0.4, zorder=2.3)

        # Flagged spots
        if len(dr) > 0:
            ax_sky.scatter(dr["x"], dr["y"], c="red", s=130, marker="x",
                           linewidths=2, zorder=3, label=f"Auto-flagged ({len(dr)})")
            for i, r in dr.iterrows():
                ax_sky.annotate(str(i), (r["x"], r["y"]),
                                fontsize=7, color="red")

        ax_sky.set_xlabel("RA offset (mas)"); ax_sky.set_ylabel("Dec offset (mas)")
        ax_sky.invert_xaxis(); ax_sky.set_aspect("equal")
        ax_sky.legend(fontsize=7, loc="lower left")
        plt.colorbar(sc, ax=ax_sky, label="V_LSR (km/s)")

        # --- Spectrum (right) ---
        sz_spec = self._fsz(cl["I"], 15, 120)
        ax_spec.scatter(cl["vlsr"], cl["I"], c=cl["vlsr"], s=sz_spec,
                        cmap=CMAP_NAME, edgecolors="k", linewidths=0.3,
                        alpha=0.7)
        if len(dr) > 0:
            ax_spec.scatter(dr["vlsr"], dr["I"], c="red", s=130,
                            marker="x", linewidths=2)
        ax_spec.set_xlabel("V_LSR (km/s)"); ax_spec.set_ylabel("Flux (Jy/beam)")

        fig.suptitle("CLICK spots to exclude/un-exclude on sky map. "
                     "CLOSE WINDOW to continue.\n"
                     "Red X=flagged | Blue O=zone | Green sq=flux | "
                     "Orange D=beam | Orange star=your selection",
                     fontsize=9)
        plt.tight_layout()

        star_artists = {}

        def on_click(ev):
            if ev.inaxes != ax_sky:
                return
            ds = np.hypot(cl["x"].values - ev.xdata,
                          cl["y"].values - ev.ydata)
            if len(ds) == 0:
                return
            ni = np.argmin(ds)
            if ds[ni] > self.beam.geo_mean * 0.5:
                return
            si = cl.index[ni]
            s = cl.loc[si]

            if si in excluded:
                excluded.discard(si)
                if si in star_artists:
                    for a in star_artists[si]:
                        a.remove()
                    del star_artists[si]
                fig.canvas.draw_idle()
                print(f"  UN-marked {si}")
            else:
                excluded.add(si)
                # Mark on both panels
                s1 = ax_sky.scatter(s["x"], s["y"], c="orange", s=280,
                                    marker="*", zorder=4, linewidths=1,
                                    edgecolors="k")
                s2 = ax_spec.scatter(s["vlsr"], s["I"], c="orange", s=280,
                                     marker="*", zorder=4, linewidths=1,
                                     edgecolors="k")
                star_artists[si] = [s1, s2]
                fig.canvas.draw_idle()
                print(f"  Marked {si}: V={s['vlsr']:.3f}, "
                      f"X={s['x']:.3f}, Y={s['y']:.3f}, I={s['I']:.3f}")

        fig.canvas.mpl_connect("button_press_event", on_click)
        plt.show()

        if excluded:
            with open(path, "w") as f:
                f.write(f"# Exclude list from masertrack_identify\n")
                f.write(f"# Source: {self.filepath.name}\n")
                for i in sorted(excluded):
                    f.write(f"{i}\n")
            print(f"\n  Wrote {len(excluded)} spot(s) to {path}")
            print(f"  Re-run with: --exclude {path}")
        else:
            print("  No spots selected for exclusion.")

    def __repr__(self):
        n = len(self.raw) if self.raw is not None else 0
        return f"MaserTrackIdentify({self.filepath.name!r}, step {self._step}/6, {n} spots)"


# =====================================================================
#  GUIDED MODE
# =====================================================================
def guided_mode(filepath, **ov):
    """Walk the user through the pipeline step by step."""
    print("=" * 60)
    print(f"  masertrack v{__version__} - masertrack_identify (guided mode)")
    print("=" * 60)
    print(f"\n  Input: {filepath}\n")

    m = MaserTrackIdentify(filepath, save_plots=True,
                           show_plots=ov.pop("show_plots", False), **ov)

    m.step1_parse()
    if not _confirm("  Continue to deduplication?"):
        return m
    m.step2_deduplicate()
    if not _confirm("  Continue to sidelobe flagging?"):
        return m
    m.step3_flag_sidelobes()

    # Offer interactive exclusion
    if m.show_plots and HAS_MPL:
        if _confirm("  Open interactive click-to-exclude window?"):
            print("  (Close the plot window when done to continue)")
            m.interactive_exclude()

    if not _confirm("  Continue to feature grouping?"):
        return m
    m.step4_group_features()
    if not _confirm("  Continue to multi-peak splitting?"):
        return m
    m.step5_split_multipeaks()
    if not _confirm("  Continue to summarization?"):
        return m
    m.step6_summarize()

    default_out = Path(filepath).stem + "_features.txt"
    out = input(f"\n  Output file? [{default_out}]: ").strip() or default_out
    # Prepend outdir if output is just a filename
    if not Path(out).is_absolute() and str(Path(out).parent) == ".":
        out = str(m.outdir / out)
    m.save(out)

    print(f"\n  Done!  Output files in {m.outdir}:")
    print(f"  PNG plots:   *_s1.png through *_summary.png")
    if HAS_PLOTLY:
        print(f"  Interactive:  *_interactive.html (open in browser)")
    print(f"  Text output:  *_features.txt and *_spots.txt")
    print(f"  Re-run with --exclude / --include if needed.\n")
    return m


def _confirm(prompt):
    if not sys.stdin.isatty():
        return True
    try:
        return input(prompt + " [Y/n]: ").strip().lower() in ("", "y", "yes")
    except EOFError:
        return True


# =====================================================================
#  COMMAND-LINE INTERFACE
# =====================================================================
def main():
    pa = argparse.ArgumentParser(
        prog="masertrack_identify",
        description=(
            f"masertrack v{__version__}: identify maser features from "
            "VLBI spot-fitting output.\n\n"
            "Takes MFIT/SAD spot tables (csad format) and produces "
            "cleaned feature catalogs through deduplication, sidelobe "
            "flagging, chain-linking, splitting, and summarization."
        ),
        epilog=textwrap.dedent("""\
        Examples:
          python masertrack_identify.py --guided input.txt --n-stations 4
          python masertrack_identify.py input.txt -o features.txt --no-plots
          python masertrack_identify.py input.txt --n-stations 10 --outdir results/

        Output files (in --outdir or current directory):
          *_features.txt       Feature catalog (one row per feature)
          *_spots.txt          Spot table with feature ID assignments
          *_s1.png ... _s6.png Step-by-step diagnostic PNG plots
          *_summary.png        Pipeline summary dashboard
          *_interactive.html   Interactive plot (needs plotly)

        https://github.com/chickin/masertrack
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pa.add_argument("input", nargs="?", default=None,
                    help="MFIT/SAD output file (csad format)")
    pa.add_argument("-o", "--output", default=None,
                    help="Output feature catalog file (default: inputname_features.txt)")
    pa.add_argument("--guided", action="store_true",
                    help="Interactive step-by-step mode (recommended for first pass)")
    pa.add_argument("--no-plots", action="store_true",
                    help="Disable ALL plot generation (PNG and HTML)")
    pa.add_argument("--show-plots", action="store_true",
                    help="Also open matplotlib windows (default: save PNG only; "
                         "set MPLBACKEND=TkAgg on macOS if windows freeze)")
    pa.add_argument("--outdir", default=".",
                    help="Output directory for all files (default: current directory)")

    g = pa.add_argument_group("beam and array parameters")
    g.add_argument("--beam-major", type=float, default=None,
                   help="Beam major axis in mas (default: auto-detect from data)")
    g.add_argument("--beam-minor", type=float, default=None,
                   help="Beam minor axis in mas (default: auto-detect)")
    g.add_argument("--beam-pa", type=float, default=None,
                   help="Beam position angle in degrees (default: auto-detect)")
    g.add_argument("--chan-spacing", type=float, default=None,
                   help="Channel spacing in km/s (default: auto-detect)")
    g.add_argument("--n-stations", type=int, default=None,
                   help="Number of array stations; sets sidelobe threshold "
                        "via 1/N (default: use generic 50%% threshold)")

    g2 = pa.add_argument_group("pipeline tuning parameters")
    g2.add_argument("--dedup-radius", type=float, default=None,
                    help="Dedup merge radius as multiple of beam geo mean (default: 1.0)")
    g2.add_argument("--sidelobe-flux", type=float, default=None,
                    help="Sidelobe flux ratio threshold (default: 1/N + 0.10)")
    g2.add_argument("--link-radius", type=float, default=None,
                    help="Feature linking radius as multiple of beam geo mean (default: 1.0)")
    g2.add_argument("--max-gap", type=float, default=None,
                    help="Max velocity gap to bridge in channel spacings (default: 2.5)")
    g2.add_argument("--min-dip", type=int, default=None,
                    help="Min consecutive drops for peak splitting (default: 2)")
    g2.add_argument("--n-brightest", type=int, default=None,
                    help="N brightest spots for position averaging (default: 3)")
    g2.add_argument("--exclude", type=str, default=None,
                    help="File with spot indices to exclude (one per line, # = comment)")
    g2.add_argument("--include", type=str, default=None,
                    help="File with spot indices to force-include (overrides sidelobe flags)")

    args = pa.parse_args()
    if args.input is None:
        pa.print_help()
        print(f"\n  masertrack v{__version__}")
        print("  Tip: --guided for interactive mode")
        print("  Tip: pip3 install plotly for interactive HTML plots\n")
        sys.exit(0)

    # Build overrides dict
    ov = {"outdir": args.outdir}
    for attr, key in [("beam_major", "beam_major"), ("beam_minor", "beam_minor"),
                      ("beam_pa", "beam_pa"), ("chan_spacing", "chan_spacing"),
                      ("n_stations", "n_stations"), ("exclude", "exclude_file"),
                      ("include", "include_file"), ("show_plots", "show_plots")]:
        v = getattr(args, attr, None)
        if v is not None:
            ov[key] = v

    if args.guided:
        guided_mode(args.input, **ov)
    else:
        sp = ov.pop("show_plots", False)
        m = MaserTrackIdentify(args.input, save_plots=not args.no_plots,
                               show_plots=sp, **ov)
        m.run()
        out = args.output or (Path(args.input).stem + "_features.txt")
        # Prepend --outdir to output path if it's just a filename
        if not Path(out).is_absolute() and str(Path(out).parent) == ".":
            out = str(Path(args.outdir) / out)
        m.save(out)


if __name__ == "__main__":
    main()
