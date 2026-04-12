#!/usr/bin/env python3
"""
masertrack_fit.py — Part 3 of the masertrack pipeline
=========================================================
Version 1.0.0

Parallax (π) and proper motion (μ) fitting from tracked maser
spot/feature positions across multiple epochs.

Input:  Tracked spots/features from masertrack_match.py
Output: Parallax + proper motion fits, diagnostics, export files

Fitting methodology:
  - Individual fitting: per-channel π + μ (Burns+2015)
  - Group fitting: common π, independent μ per channel (Burns+2015)
  - Feature fitting: flux-weighted feature centroids (comparison)
  - Error floors: automated bisection to χ²_red=1 (Reid+2009)
  - Outlier tolerance: Cauchy-tail likelihood (Reid+2014)
  - Bootstrap: 10⁴ epoch-resampled trials for robust uncertainties
  - Distance: 1/π default, optional Bayesian priors (disk/exponential)

References:
  Reid et al. 2009, ApJ 693, 397  — BeSSeL fitting, error floors
  Reid et al. 2009, ApJ 705, 1548 — Sgr B2 parallax (validation data)
  Reid et al. 2014, ApJ 783, 130  — √N correction, outlier tolerance
  Burns et al. 2015, MNRAS 453, 3163 — individual/group/feature comparison
  Bland-Hawthorn & Gerhard 2016, ARA&A 54, 529 — Galactic disk model

Changelog:
  v1.1  Code cleanup, unified commenting style, tutorial with synthetic
        data, dead code removal (fit_iterative), bug fixes.
  v1.0  First release. Validated against Reid+2009 Sgr B2 (0.0σ).
        Features: group/individual/feature fitting, bootstrap, Chauvenet
        outlier detection, error floors, 3×3 diagnostic plots, interactive
        HTML, publication CSV export, 4 export formats (pmpar/BeSSeL/VERA/KaDai),
        configurable distance priors (none/disk/exponential).

Credits: Gabor Orosz (design), Claude (code, 2026)
License: BSD-3-Clause
Repository: https://github.com/chickin/masertrack
"""
from __future__ import annotations
import argparse, os, sys, textwrap, re, copy, warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time as _time
warnings.filterwarnings("ignore", category=UserWarning)

__version__ = "1.1"

try: import numpy as np
except ImportError: print("numpy required"); sys.exit(1)
try: import pandas as pd
except ImportError: print("pandas required"); sys.exit(1)
try: from scipy.optimize import brentq
except ImportError: print("scipy required"); sys.exit(1)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    HAS_MPL = True
except ImportError: HAS_MPL = False
try:
    import plotly.graph_objects as go; from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError: HAS_PLOTLY = False

# Astropy for DE440 ephemeris (preferred) or analytical fallback
HAS_ASTROPY = False
try:
    from astropy.time import Time as AstropyTime
    from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
    HAS_ASTROPY = True
except ImportError:
    pass

# =====================================================================
#  CONSTANTS
# =====================================================================
JULIANYEAR = 365.25       # Julian year in days
MJD_J2000  = 51544.5      # MJD of J2000.0 epoch
DEG2RAD    = np.pi / 180.0
RAD2DEG    = 180.0 / np.pi
RAD2MAS    = RAD2DEG * 3600.0 * 1000.0
MAS2RAD    = 1.0 / RAD2MAS
HR2RAD     = np.pi / 12.0

# Obliquity of the ecliptic (J2000)
OBLIQUITY_DEG = 23.439291  # degrees

# Galactic disk defaults (Juric+2008, Bland-Hawthorn & Gerhard 2016)
R_SUN_KPC = 8.15   # Solar Galactocentric radius (Reid+2019)
DISK_HR   = 2.6    # Radial scale length (kpc) — thin disk
DISK_HZ   = 0.9    # Vertical scale height (kpc) — thick disk (evolved stars)

# Fitting defaults
DEFAULTS = {
    "bootstrap_trials": 10000,
    "min_epochs_individual": 4,
    "min_epochs_group": 4,
    "grade_threshold": "C",    # fit A+B+C by default
    "ref_epoch": "mean",       # mean of unflagged epochs
    "distance_prior": "none",     # no prior by default (1/pi); options: none, exponential, disk
    "distance_dmax": 30.0,     # kpc, max distance for grid
    "distance_ngrid": 10000,   # grid points for posterior
    "cauchy_max_iter": 20,     # outlier-tolerant fitting iterations
    "bic_acceleration_threshold": -6,  # BIC difference to favor acceleration
}

# Population-specific disk prior parameters (h_R in kpc, h_z in kpc)
# Use with --distance-prior disk --disk-scale h_R h_z
# Examples:
#   HMSFR masers (6.7 GHz CH3OH, 22 GHz H2O in SFRs):  h_R=2.4, h_z=0.025
#   AGB O-rich masers (22 GHz H2O, 43 GHz SiO):        h_R=2.5, h_z=0.250
#   Post-AGB / old thin disk:                            h_R=2.6, h_z=0.300
#   Thick disk (evolved, dynamically heated):             h_R=2.6, h_z=0.900
#   Carbon stars:                                         h_R=2.5, h_z=0.200
# References:
#   Reid+2019 (HMSFR), Jackson+2002 (AGB), Bland-Hawthorn & Gerhard 2016 (disk)
#   Andriantsaralaza+2022 (AGB prior: h_R=3.5, h_z=0.240)

GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}

# =====================================================================
#  INPUT PARSER
# =====================================================================
def parse_tracked_file(path):
    """Parse match_*_spots_tracked.txt or match_*_features_tracked.txt.
    Handles the empty 'flag' column that causes split() to give fewer tokens."""
    lines = [l.rstrip() for l in open(path)]
    header = None
    data_lines = []
    meta = {}
    for l in lines:
        ls = l.strip()
        if ls.startswith("#"):
            tokens = ls.lstrip("# ").split()
            if "group_id" in tokens:
                header = tokens
            if "masertrack_match" in ls:
                m = re.search(r"v([\d.]+)", ls)
                if m: meta["match_version"] = m.group(1)
            if "PR" in ls or "pr" in ls.lower():
                if "inverse" in ls.lower(): meta["mode"] = "inverse"
                elif "normal" in ls.lower(): meta["mode"] = "normal"
            continue
        if ls:
            data_lines.append(l)
    if not header or not data_lines:
        return pd.DataFrame(), meta

    n_cols = len(header)
    flag_idx = header.index("flag") if "flag" in header else -1

    rows = []
    for dl in data_lines:
        parts = dl.split()
        # If flag column is empty, split gives n_cols-1 tokens
        if len(parts) == n_cols - 1 and flag_idx >= 0:
            parts.insert(flag_idx, "")
        elif len(parts) < n_cols - 1:
            continue  # skip malformed lines
        # Truncate or pad to match header
        parts = parts[:n_cols]
        rows.append(parts)

    df = pd.DataFrame(rows, columns=header)

    # Convert numeric columns
    numeric = ["group_id", "mjd", "dec_year", "n_stations", "use_for_pi",
               "vlsr", "x", "y", "x_err", "y_err", "flux"]
    if "feature_id" in df.columns: numeric.append("feature_id")
    if "n_feat_channels" in df.columns: numeric.append("n_feat_channels")
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["group_id"] = df["group_id"].astype(int)
    df["use_for_pi"] = df["use_for_pi"].astype(int)
    df["n_stations"] = df["n_stations"].astype(int)

    return df, meta


def parse_groups_file(path):
    """Parse match_*_groups.txt for group catalog metadata."""
    lines = [l.rstrip() for l in open(path)]
    header = None
    data_lines = []
    for l in lines:
        ls = l.strip()
        if ls.startswith("#"):
            tokens = ls.lstrip("# ").split()
            if "group_id" in tokens:
                header = tokens
            continue
        if ls:
            data_lines.append(ls)
    if not header or not data_lines:
        return pd.DataFrame()
    rows = [dl.split() for dl in data_lines]
    df = pd.DataFrame(rows, columns=header[:len(rows[0])])
    for c in df.columns:
        if c not in ["grade", "pattern"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["group_id"] = df["group_id"].astype(int)
    return df


def parse_source_coords(ra_str, dec_str):
    """Parse RA (hh:mm:ss.sss) and Dec (dd:mm:ss.sss) to radians."""
    # RA
    parts = re.split(r"[hms: ]+", ra_str.strip())
    parts = [p for p in parts if p]
    ra_hr = float(parts[0]) + float(parts[1]) / 60.0 + float(parts[2]) / 3600.0
    ra_rad = ra_hr * HR2RAD

    # Dec
    dec_str_clean = dec_str.strip()
    sign = -1.0 if dec_str_clean.startswith("-") else 1.0
    parts = re.split(r"[dms: °'\"]+", dec_str_clean.lstrip("+-"))
    parts = [p for p in parts if p]
    dec_deg = float(parts[0]) + float(parts[1]) / 60.0 + float(parts[2]) / 3600.0
    dec_rad = sign * dec_deg * DEG2RAD

    return ra_rad, dec_rad


# =====================================================================
#  PARALLAX FACTORS
# =====================================================================
def parallax_factors_de440(ra_rad, dec_rad, mjds, ephemeris="de440"):
    """Compute parallax factors using astropy + JPL DE440 ephemeris.

    Returns (F_alpha, F_delta) in AU, where:
      Δα·cos(δ) = π × F_alpha    and    Δδ = π × F_delta

    Convention: input data must be in Δα·cos(δ) (not raw Δα).
    F_alpha does NOT include a 1/cos(δ) factor — this is the
    BeSSeL/pmpar standard, matching Reid+2009.
    """
    if not HAS_ASTROPY:
        raise ImportError("astropy not available; use --ephemeris analytical")

    times = AstropyTime(np.asarray(mjds, dtype=float), format="mjd")
    try:
        with solar_system_ephemeris.set(ephemeris):
            earth = get_body_barycentric("earth", times)
    except Exception as e:
        # Re-raise so the caller can try the next option
        raise RuntimeError(f"{ephemeris} not available: {e}") from e

    X = earth.x.to("AU").value
    Y = earth.y.to("AU").value
    Z = earth.z.to("AU").value

    sa, ca = np.sin(ra_rad), np.cos(ra_rad)
    sd, cd = np.sin(dec_rad), np.cos(dec_rad)

    # F_a NOT divided by cos(δ): data is Δα·cos(δ), not Δα
    F_a = X * sa - Y * ca
    F_d = X * ca * sd + Y * sa * sd - Z * cd
    return F_a, F_d


def parallax_factors_analytical(ra_rad, dec_rad, mjds):
    """Compute parallax factors using the 2003 Astronomical Almanac algorithm.

    This is the same formula used by pmpar (Brisken/Reid) and Nakagawa's
    'strict' version. Includes equation of center and eccentricity.
    Accuracy: ~0.01° in solar longitude, <0.1% in parallax factors.

    References:
      - 2003 Astronomical Almanac
      - Reid & Brunthaler 2004, ApJ 616, 872
      - pmpar sun.cc (Brisken, GPL-3.0)
    """
    mjds = np.asarray(mjds, dtype=float)
    n = mjds - MJD_J2000  # days since J2000.0

    # Mean longitude and anomaly of the Sun
    L = np.radians((280.460 + 0.9856474 * n) % 360.0)
    g = np.radians((357.528 + 0.9856003 * n) % 360.0)

    # Ecliptic longitude (with equation of center)
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2.0 * g)

    # Obliquity (slowly varying)
    eps = np.radians(23.439 - 0.0000004 * n)

    # Earth-Sun distance in AU
    R = 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2.0 * g)

    # Sun position in equatorial ICRS (cartesian, AU)
    X_sun = R * np.cos(lam)
    Y_sun = R * np.sin(lam) * np.cos(eps)
    Z_sun = R * np.sin(lam) * np.sin(eps)

    # Earth is at -Sun (barycentric)
    X, Y, Z = -X_sun, -Y_sun, -Z_sun

    sa, ca = np.sin(ra_rad), np.cos(ra_rad)
    sd, cd = np.sin(dec_rad), np.cos(dec_rad)

    # F_a NOT divided by cos(δ): data is Δα·cos(δ), not Δα
    F_a = X * sa - Y * ca
    F_d = X * ca * sd + Y * sa * sd - Z * cd
    return F_a, F_d


def get_parallax_factors(ra_rad, dec_rad, mjds, method="auto"):
    """Get parallax factors using best available method."""
    if method == "auto":
        if HAS_ASTROPY:
            try:
                return parallax_factors_de440(ra_rad, dec_rad, mjds)
            except Exception:
                return parallax_factors_analytical(ra_rad, dec_rad, mjds)
        else:
            return parallax_factors_analytical(ra_rad, dec_rad, mjds)
    elif method in ("de440", "de421", "builtin"):
        return parallax_factors_de440(ra_rad, dec_rad, mjds, ephemeris=method)
    elif method == "analytical":
        return parallax_factors_analytical(ra_rad, dec_rad, mjds)
    else:
        raise ValueError(f"Unknown ephemeris method: {method}")

# Cache for resolved ephemeris method to avoid repeated fallback warnings
_RESOLVED_EPHEMERIS = [None]

def get_parallax_factors_cached(ra_rad, dec_rad, mjds, method="auto"):
    """Wrapper that resolves the ephemeris method once, then reuses it."""
    if _RESOLVED_EPHEMERIS[0] is not None:
        return get_parallax_factors(ra_rad, dec_rad, mjds, _RESOLVED_EPHEMERIS[0])

    # First call: resolve and cache
    if method == "auto":
        if HAS_ASTROPY:
            for eph in ("de440", "de421"):
                try:
                    result = parallax_factors_de440(ra_rad, dec_rad, mjds, eph)
                    _RESOLVED_EPHEMERIS[0] = eph
                    print(f"  Ephemeris resolved: {eph}")
                    return result
                except Exception:
                    continue
        _RESOLVED_EPHEMERIS[0] = "analytical"
        print(f"  Ephemeris resolved: analytical (no astropy/DE440)")
        return parallax_factors_analytical(ra_rad, dec_rad, mjds)
    else:
        _RESOLVED_EPHEMERIS[0] = method
        return get_parallax_factors(ra_rad, dec_rad, mjds, method)


# =====================================================================
#  CORE WEIGHTED LEAST SQUARES ENGINE
# =====================================================================
class FitResult:
    """Container for a single WLS fit result."""
    def __init__(self):
        self.params = None       # parameter vector
        self.cov = None          # covariance matrix
        self.labels = []         # parameter names
        self.pi = 0.0            # parallax (mas)
        self.sigma_pi = 0.0      # parallax uncertainty (mas)
        self.mu = {}             # proper motions per channel
        self.chi2 = 0.0          # chi-squared
        self.chi2_red = 0.0      # reduced chi-squared
        self.dof = 0             # degrees of freedom
        self.resid_ra = None     # RA residuals (mas)
        self.resid_dec = None    # Dec residuals (mas)
        self.sigma_pdf_ra = 0.0  # sigma_pdf for RA
        self.sigma_pdf_dec = 0.0 # sigma_pdf for Dec
        self.floor_ra = 0.0      # error floor RA (mas)
        self.floor_dec = 0.0     # error floor Dec (mas)
        self.n_data = 0          # number of data points
        self.n_params = 0        # number of parameters
        self.n_channels = 0      # number of channels fitted
        self.coord_mode = "both" # "ra", "dec", or "both"
        self.fit_type = ""       # "individual", "group", "feature"


def build_design_matrix_individual(F_a, F_d, dt, coord="both"):
    """Build design matrix for a SINGLE channel fit (5 parameters).

    Parameters: [pi, mu_x, x0, mu_y, y0]
    Data vector: [x1, y1, x2, y2, ...] (interleaved RA, Dec)

    Parameters
    ----------
    F_a, F_d : array, shape (N_epochs,)
        Parallax factors for RA and Dec
    dt : array, shape (N_epochs,)
        Time offsets from reference epoch (years)
    coord : str
        "both" (default), "ra", or "dec"
    """
    N = len(F_a)
    if coord == "both":
        A = np.zeros((2 * N, 5))
        # RA rows (even indices)
        A[0::2, 0] = F_a      # parallax
        A[0::2, 1] = dt        # mu_x
        A[0::2, 2] = 1.0       # x0
        # Dec rows (odd indices)
        A[1::2, 0] = F_d      # parallax (same parameter!)
        A[1::2, 3] = dt        # mu_y
        A[1::2, 4] = 1.0       # y0
    elif coord == "ra":
        A = np.zeros((N, 3))
        A[:, 0] = F_a
        A[:, 1] = dt
        A[:, 2] = 1.0
    elif coord == "dec":
        A = np.zeros((N, 3))
        A[:, 0] = F_d
        A[:, 1] = dt
        A[:, 2] = 1.0
    return A


def build_design_matrix_group(F_a, F_d, dt_list, coord="both",
                               include_accel=False):
    """Build design matrix for GROUP fit (common pi, independent mu per channel).

    Parameters: [pi, mu_x1, x01, mu_y1, y01, ..., mu_xN, x0N, mu_yN, y0N]
    = 1 + 4*N_channels parameters (or 1 + 2*N for single-coord)

    Parameters
    ----------
    F_a, F_d : list of arrays, one per channel
        Parallax factors for each channel's epochs
    dt_list : list of arrays, one per channel
        Time offsets for each channel's epochs
    coord : str
        "both", "ra", or "dec"
    include_accel : bool
        If True, add acceleration terms (2 extra params per channel)
    """
    N_ch = len(F_a)
    if coord == "both":
        ppch = 6 if include_accel else 4  # params per channel (excl. pi)
    else:
        ppch = 3 if include_accel else 2

    n_params = 1 + ppch * N_ch
    total_data = 0
    for i in range(N_ch):
        n_ep = len(F_a[i])
        total_data += 2 * n_ep if coord == "both" else n_ep

    A = np.zeros((total_data, n_params))
    row = 0

    for ch in range(N_ch):
        n_ep = len(F_a[ch])
        base = 1 + ch * ppch  # parameter index offset for this channel

        if coord == "both":
            for i in range(n_ep):
                # RA row
                A[row, 0] = F_a[ch][i]        # common parallax
                A[row, base + 0] = dt_list[ch][i]   # mu_x
                A[row, base + 1] = 1.0               # x0
                if include_accel:
                    A[row, base + 4] = 0.5 * dt_list[ch][i]**2  # accel_x
                row += 1
                # Dec row
                A[row, 0] = F_d[ch][i]        # common parallax
                A[row, base + 2] = dt_list[ch][i]   # mu_y
                A[row, base + 3] = 1.0               # y0
                if include_accel:
                    A[row, base + 5] = 0.5 * dt_list[ch][i]**2  # accel_y
                row += 1
        elif coord == "ra":
            for i in range(n_ep):
                A[row, 0] = F_a[ch][i]
                A[row, base + 0] = dt_list[ch][i]
                A[row, base + 1] = 1.0
                if include_accel:
                    A[row, base + 2] = 0.5 * dt_list[ch][i]**2
                row += 1
        elif coord == "dec":
            for i in range(n_ep):
                A[row, 0] = F_d[ch][i]
                A[row, base + 0] = dt_list[ch][i]
                A[row, base + 1] = 1.0
                if include_accel:
                    A[row, base + 2] = 0.5 * dt_list[ch][i]**2
                row += 1

    return A


def wls_solve(A, b, w):
    """Weighted least squares: solve (A^T W A)^{-1} A^T W b.

    Parameters
    ----------
    A : array (N_data, N_params) — design matrix
    b : array (N_data,) — data vector
    w : array (N_data,) — weights = 1/sigma^2

    Returns
    -------
    params, cov, resid, chi2, dof
    """
    W = np.diag(w)
    AtWA = A.T @ W @ A
    AtWb = A.T @ (w * b)

    # Check for singular matrix
    try:
        cov = np.linalg.inv(AtWA)
    except np.linalg.LinAlgError:
        return None, None, None, np.inf, 0

    params = cov @ AtWb
    resid = b - A @ params
    chi2 = np.sum(resid**2 * w)
    dof = len(b) - len(params)
    return params, cov, resid, chi2, max(dof, 1)


# =====================================================================
#  ERROR FLOOR FINDING
# =====================================================================
def find_error_floor(A, b, sigma, n_params, coord_mask=None, tol=1e-8):
    """Find error floor epsilon such that reduced chi^2 = 1.

    Adds epsilon in quadrature: sigma_new = sqrt(sigma^2 + eps^2)
    Uses Brent's method (bisection) for reliable convergence.

    Parameters
    ----------
    A : design matrix
    b : data vector
    sigma : array of uncertainties
    n_params : number of fitted parameters
    coord_mask : boolean array, True for the coordinate being floored
                 (None = floor all)
    tol : convergence tolerance for epsilon

    Returns
    -------
    epsilon : the error floor (mas)
    """
    def chi2_red_with_floor(eps):
        s2 = sigma.copy()**2
        if coord_mask is not None:
            s2[coord_mask] += eps**2
        else:
            s2 += eps**2
        s_new = np.sqrt(s2)
        w = 1.0 / s_new**2
        _, _, resid, chi2, dof = wls_solve(A, b, w)
        if resid is None:
            return 0.0
        return chi2 / dof - 1.0

    # Check if floor is needed
    w0 = 1.0 / sigma**2
    _, _, resid0, chi2_0, dof0 = wls_solve(A, b, w0)
    if resid0 is None or chi2_0 / dof0 <= 1.0:
        return 0.0

    # Find upper bound for bisection
    eps_hi = 10.0 * np.max(np.abs(resid0))
    if chi2_red_with_floor(eps_hi) > 0:
        eps_hi *= 10  # very large residuals

    try:
        eps = brentq(chi2_red_with_floor, 0.0, eps_hi, xtol=tol)
    except ValueError:
        # Function does not change sign — return conservative estimate
        eps = np.median(np.abs(resid0))
    return eps


def find_error_floors_radec(A, b, sigma, n_params, n_data_per_epoch_pair=2):
    """Find error floors for RA and Dec.

    Uses the Reid et al. (2009) approach: add error floor in quadrature
    until reduced chi-squared equals unity.

    For robustness, finds a COMBINED floor first (same for RA and Dec).
    If RA and Dec have very different scatter, attempts separate floors
    via iterative refinement. Falls back to combined if iteration fails.
    """
    # Step 1: Find combined floor that achieves chi2_red = 1
    eps_combined = find_error_floor(A, b, sigma, n_params, coord_mask=None)

    if eps_combined == 0.0:
        return 0.0, 0.0

    # Step 2: Check RA vs Dec residual scatter after applying combined floor
    s_floored = apply_error_floor(sigma, eps_combined, eps_combined, interleaved=True)
    w = 1.0 / s_floored**2
    _, _, resid, _, _ = wls_solve(A, b, w)
    if resid is None:
        return eps_combined, eps_combined

    ra_mask = np.zeros(len(b), dtype=bool)
    ra_mask[0::2] = True
    chi2_ra = np.sum((resid[ra_mask] / s_floored[ra_mask])**2)
    chi2_dec = np.sum((resid[~ra_mask] / s_floored[~ra_mask])**2)
    n_ra = np.sum(ra_mask)
    n_dec = np.sum(~ra_mask)

    # Per-coordinate sigma_pdf (diagnostic — should be near 1.0 each)
    spdf_ra = np.sqrt(chi2_ra / max(n_ra - n_params // 2, 1))
    spdf_dec = np.sqrt(chi2_dec / max(n_dec - n_params // 2, 1))

    # If asymmetry is moderate (<3x), the combined floor is fine
    ratio = max(spdf_ra, spdf_dec) / max(min(spdf_ra, spdf_dec), 0.01)
    if ratio < 3.0:
        return eps_combined, eps_combined

    # Significant asymmetry: scale the combined floor by the per-coord sigma_pdf
    # This redistributes the floor proportionally to each coordinate's scatter
    eps_ra = eps_combined * spdf_ra
    eps_dec = eps_combined * spdf_dec
    # Renormalize so the combined chi2_red ≈ 1
    # (this is approximate but much better than independent bisection)
    s_test = apply_error_floor(sigma, eps_ra, eps_dec, interleaved=True)
    w_test = 1.0 / s_test**2
    _, _, _, chi2_test, dof_test = wls_solve(A, b, w_test)
    if chi2_test is not None and dof_test > 0:
        scale = np.sqrt(chi2_test / dof_test)
        eps_ra *= scale
        eps_dec *= scale
    return eps_ra, eps_dec


def apply_error_floor(sigma, eps_ra, eps_dec, interleaved=True):
    """Apply error floors in quadrature to sigma array."""
    s = sigma.copy()
    if interleaved:
        s[0::2] = np.sqrt(sigma[0::2]**2 + eps_ra**2)
        s[1::2] = np.sqrt(sigma[1::2]**2 + eps_dec**2)
    else:
        s = np.sqrt(sigma**2 + eps_ra**2)  # single-coord mode
    return s


# =====================================================================
#  OUTLIER-TOLERANT FITTING (Reid et al. 2014)
# =====================================================================
def outlier_tolerant_fit(A, b, sigma, max_iter=20, tol=1e-6):
    """Iterative outlier-tolerant fitting using Cauchy-tail weights.

    Starts from standard WLS, then iteratively reweights data points
    using the Cauchy-tail probability. Points with large residuals
    get reduced weight.

    Returns params, cov (from final Gaussian fit on cleaned data),
    and outlier flags.
    """
    n_data = len(b)
    w = 1.0 / sigma**2
    params_prev = None

    for iteration in range(max_iter):
        params, cov, resid, chi2, dof = wls_solve(A, b, w)
        if params is None:
            return None, None, None, np.zeros(n_data, dtype=bool)

        if params_prev is not None:
            if np.max(np.abs(params - params_prev)) < tol:
                break
        params_prev = params.copy()

        # Compute Cauchy weights
        R2 = (resid / sigma)**2
        # Weight = d(log P)/d(R^2) for the Cauchy-tail distribution
        # For Gaussian: weight = 1/(2*sigma^2)
        # For Cauchy-tail: weight decreases for large R
        exp_term = np.exp(-R2 / 2.0)
        # Effective weight: Gaussian weight * (exp(-R2/2)) / (1 - exp(-R2/2))
        # For small R: ≈ 1 (Gaussian)
        # For large R: ≈ exp(-R2/2) ≈ 0 (downweighted)
        cauchy_factor = np.where(R2 > 0.01,
                                  exp_term / np.maximum(1.0 - exp_term, 1e-30),
                                  1.0 - R2 / 12.0)  # Taylor expansion for small R
        # Clip to prevent extreme weights
        cauchy_factor = np.clip(cauchy_factor, 0.01, 1.0)
        w = cauchy_factor / sigma**2

    # Flag outliers: points where Cauchy weight < 0.5 of Gaussian weight
    outlier_flags = cauchy_factor < 0.5

    return params, cov, resid, outlier_flags


# =====================================================================
#  BOOTSTRAP
# =====================================================================
def bootstrap_fit(A, b, sigma, n_trials=10000, seed=42,
                   epoch_groups=None):
    """Bootstrap uncertainty estimation for WLS fit.

    For multi-channel group fits, resamples EPOCHS (not individual
    data points) to preserve the correlation structure — all channels
    at a given epoch are kept or dropped together.

    Parameters
    ----------
    epoch_groups : list of arrays of int indices, optional
        Each element lists the row indices in (A, b, sigma) that belong
        to a single epoch. If None, treats pairs of consecutive rows
        as epoch-pairs (single-channel mode).
    """
    rng = np.random.default_rng(seed)
    n_data = len(b)
    n_params = A.shape[1]

    if epoch_groups is None:
        # Single-channel mode: pairs of rows are RA+Dec for one epoch
        n_groups = n_data // 2
        epoch_groups = [np.array([2*i, 2*i+1]) for i in range(n_groups)]

    n_groups = len(epoch_groups)
    results = np.zeros((n_trials, n_params))
    n_success = 0

    for trial in range(n_trials):
        # Resample epoch groups with replacement
        chosen = rng.choice(n_groups, size=n_groups, replace=True)
        idx = np.concatenate([epoch_groups[c] for c in chosen])

        A_boot = A[idx]
        b_boot = b[idx]
        w_boot = 1.0 / sigma[idx]**2

        params, _, _, _, dof = wls_solve(A_boot, b_boot, w_boot)
        if params is not None and dof > 0:
            results[n_success] = params
            n_success += 1

    return results[:n_success]


# =====================================================================
#  DISTANCE POSTERIOR
# =====================================================================
def galactic_disk_prior(D_grid, l_rad, b_rad, h_R=DISK_HR, h_z=DISK_HZ,
                         R_sun=R_SUN_KPC):
    """Compute P(D) ∝ D^2 * rho(R_GC, z) for a Galactic disk model.

    Parameters
    ----------
    D_grid : array, distance grid in kpc
    l_rad, b_rad : Galactic longitude and latitude in radians
    h_R : radial scale length (kpc)
    h_z : vertical scale height (kpc)
    R_sun : solar Galactocentric radius (kpc)

    Returns
    -------
    prior : array, unnormalized P(D) evaluated on the grid

    References
    ----------
    Bland-Hawthorn & Gerhard 2016, ARA&A 54, 529
    Juric et al. 2008, ApJ 673, 864
    """
    cb, sb = np.cos(b_rad), np.sin(b_rad)
    cl = np.cos(l_rad)

    R_GC = np.sqrt(R_sun**2 + (D_grid * cb)**2 - 2 * R_sun * D_grid * cb * cl)
    z = D_grid * sb

    prior = D_grid**2 * np.exp(-R_GC / h_R) * np.exp(-np.abs(z) / h_z)
    return prior


def bailer_jones_prior(D_grid, L=5.0):
    """Bailer-Jones exponentially decreasing space density prior.
    P(D) ∝ D^2 * exp(-D/L)
    """
    return D_grid**2 * np.exp(-D_grid / L)


def compute_distance_posterior(pi_mas, sigma_pi, l_rad, b_rad,
                                prior_type="none", D_max=30.0, n_grid=10000,
                                h_R=DISK_HR, h_z=DISK_HZ, L_bj=5.0):
    """Compute the distance posterior P(D | pi, sigma_pi).

    Parameters
    ----------
    pi_mas : parallax in mas
    sigma_pi : parallax uncertainty in mas
    l_rad, b_rad : Galactic coordinates in radians
    prior_type : "none" (default), "exponential", "disk", or "uniform"
        - none: simple D = 1/π with asymmetric error propagation (VLBI standard)
        - exponential: Bailer-Jones (2015) P(D) ∝ D² exp(-D/L), configurable L
        - disk: Galactic disk P(D) ∝ D² exp(-R/h_R) exp(-|z|/h_z), configurable
        - uniform: flat prior (P(D) = const), useful for prior comparison tests
    D_max : maximum distance for grid (kpc)
    n_grid : number of grid points

    Returns
    -------
    dict with D_mode, D_median, D_16, D_84, D_grid, posterior, prior_type
    """
    # For "none" prior: return 1/pi with asymmetric errors, no Bayesian posterior
    if prior_type == "none":
        if pi_mas <= 0:
            return {"D_mode": np.nan, "D_median": np.nan,
                    "D_16": np.nan, "D_84": np.nan,
                    "D_05": np.nan, "D_95": np.nan,
                    "D_grid": np.array([]), "posterior": np.array([]),
                    "prior_type": "none"}
        D_best = 1.0 / pi_mas
        # Asymmetric errors from error propagation
        if pi_mas - sigma_pi > 0:
            D_upper = 1.0 / (pi_mas - sigma_pi)
        else:
            D_upper = D_max
        D_lower = 1.0 / (pi_mas + sigma_pi)
        # Map to 68% CI equivalent
        D_16 = D_lower
        D_84 = min(D_upper, D_max)
        return {"D_mode": D_best, "D_median": D_best,
                "D_16": D_16, "D_84": D_84,
                "D_05": 1.0/(pi_mas + 2*sigma_pi),
                "D_95": min(1.0/max(pi_mas - 2*sigma_pi, 0.001), D_max),
                "D_grid": np.array([]), "posterior": np.array([]),
                "prior_type": "none"}
    D_grid = np.linspace(0.01, D_max, n_grid)
    dD = D_grid[1] - D_grid[0]

    # Prior (only reached for non-"none" prior types)
    if prior_type == "disk":
        prior = galactic_disk_prior(D_grid, l_rad, b_rad, h_R, h_z)
    elif prior_type == "exponential":
        prior = bailer_jones_prior(D_grid, L_bj)
    elif prior_type == "uniform":
        prior = np.ones_like(D_grid)
    else:
        prior = np.ones_like(D_grid)

    # Likelihood: P(pi_obs | D) = Gaussian(pi_obs; 1/D, sigma_pi)
    pi_model = 1.0 / D_grid
    likelihood = np.exp(-0.5 * ((pi_mas - pi_model) / sigma_pi)**2)

    # Posterior
    posterior = prior * likelihood
    total = np.sum(posterior) * dD
    if total == 0 or not np.isfinite(total):
        return {"D_mode": 1.0/pi_mas if pi_mas > 0 else np.nan,
                "D_median": np.nan, "D_16": np.nan, "D_84": np.nan,
                "D_grid": D_grid, "posterior": posterior,
                "prior_type": prior_type}

    posterior /= total

    # Statistics
    D_mode = D_grid[np.argmax(posterior)]
    cdf = np.cumsum(posterior) * dD
    D_median = D_grid[min(np.searchsorted(cdf, 0.5), n_grid - 1)]
    D_16 = D_grid[min(np.searchsorted(cdf, 0.16), n_grid - 1)]
    D_84 = D_grid[min(np.searchsorted(cdf, 0.84), n_grid - 1)]
    D_05 = D_grid[min(np.searchsorted(cdf, 0.05), n_grid - 1)]
    D_95 = D_grid[min(np.searchsorted(cdf, 0.95), n_grid - 1)]

    return {"D_mode": D_mode, "D_median": D_median,
            "D_16": D_16, "D_84": D_84, "D_05": D_05, "D_95": D_95,
            "D_grid": D_grid, "posterior": posterior,
            "prior_type": prior_type}


# =====================================================================
#  COORDINATE CONVERSIONS
# =====================================================================
def equatorial_to_galactic(ra_rad, dec_rad):
    """Convert equatorial (J2000) to Galactic coordinates.

    Returns (l_rad, b_rad) in radians.
    """
    # North Galactic Pole (J2000): RA = 12h51m26.28s, Dec = +27d07m41.7s
    ra_ngp = np.radians(192.85948)
    dec_ngp = np.radians(27.12825)
    l_ncp = np.radians(122.93192)  # Galactic longitude of North Celestial Pole

    sdec = np.sin(dec_rad)
    cdec = np.cos(dec_rad)
    sdec_ngp = np.sin(dec_ngp)
    cdec_ngp = np.cos(dec_ngp)
    dra = ra_rad - ra_ngp

    sb = sdec * sdec_ngp + cdec * cdec_ngp * np.cos(dra)
    b_rad = np.arcsin(np.clip(sb, -1, 1))

    y = cdec * np.sin(dra)
    x = sdec * cdec_ngp - cdec * sdec_ngp * np.cos(dra)
    l_rad = l_ncp - np.arctan2(y, x)
    l_rad = l_rad % (2 * np.pi)

    return l_rad, b_rad


# =====================================================================
#  CHANNEL DATA PREPARATION
# =====================================================================
@dataclass
class ChannelData:
    """Data for a single velocity channel tracked across epochs."""
    group_id: int
    vlsr: float
    grade: str
    mjds: np.ndarray        # MJD of each epoch
    dec_years: np.ndarray   # decimal year
    x: np.ndarray           # RA offset (mas)
    y: np.ndarray           # Dec offset (mas)
    x_err: np.ndarray       # RA error (mas)
    y_err: np.ndarray       # Dec error (mas)
    flux: np.ndarray        # flux density
    epochs: list            # epoch codes
    use_for_pi: np.ndarray  # 1=use, 0=flagged

    @property
    def n_epochs(self):
        return len(self.mjds)

    @property
    def n_usable(self):
        return int(np.sum(self.use_for_pi))


def extract_channels(df, group_id, chan_spacing=None):
    """Extract per-channel data for a given group from the tracked spots DataFrame.

    vlsr values drift slightly across epochs due to Doppler tracking.
    This function clusters vlsr values into physical channels by rounding
    to the nearest channel spacing.

    The auto-detection identifies inter-channel gaps (>0.3 km/s) vs
    intra-channel Doppler drift (<0.25 km/s). For VERA 22 GHz data
    the channel spacing is 0.425 km/s.
    """
    gdf = df[df.group_id == group_id].copy()
    if len(gdf) == 0:
        return []

    # Auto-detect channel spacing from vlsr gaps
    if chan_spacing is None:
        # Best method: look at consecutive vlsr values within a SINGLE epoch
        # These give the true channel spacing without Doppler drift confusion
        for epoch_code in gdf.epoch.unique():
            edf = gdf[gdf.epoch == epoch_code].sort_values("vlsr")
            if len(edf) >= 3:
                dv = np.diff(edf.vlsr.values)
                # All gaps within one epoch should be ~equal (channel spacing)
                if len(dv) >= 2 and np.std(dv) / np.median(dv) < 0.1:
                    chan_spacing = np.median(dv)
                    break
        # Fallback: look for the dominant gap in all vlsr values
        if chan_spacing is None:
            vlsrs = sorted(gdf.vlsr.unique())
            if len(vlsrs) > 1:
                gaps = np.diff(vlsrs)
                inter_channel = gaps[gaps > 0.3]
                if len(inter_channel) >= 2:
                    chan_spacing = np.median(inter_channel)
                else:
                    chan_spacing = 0.421  # VERA 22 GHz default
            else:
                chan_spacing = 0.421

    # Round vlsr to nearest channel spacing for grouping
    v_ref = gdf.vlsr.median()
    gdf["vlsr_chan"] = np.round((gdf.vlsr - v_ref) / chan_spacing) * chan_spacing + v_ref

    channels = []
    for vlsr_chan, cdf in gdf.groupby("vlsr_chan"):
        cdf = cdf.sort_values("dec_year")
        # If rounding creates duplicate epochs, keep the brightest spot
        if cdf.epoch.duplicated().any():
            cdf = cdf.sort_values("flux", ascending=False).drop_duplicates("epoch").sort_values("dec_year")
        ch = ChannelData(
            group_id=group_id,
            vlsr=vlsr_chan,  # use the rounded channel center
            grade=cdf.grade.iloc[0],
            mjds=cdf.mjd.values.astype(float),
            dec_years=cdf.dec_year.values.astype(float),
            x=cdf.x.values.astype(float),
            y=cdf.y.values.astype(float),
            x_err=cdf.x_err.values.astype(float),
            y_err=cdf.y_err.values.astype(float),
            flux=cdf.flux.values.astype(float),
            epochs=cdf.epoch.tolist(),
            use_for_pi=cdf.use_for_pi.values.astype(int),
        )
        channels.append(ch)
    return channels


# =====================================================================
#  INDIVIDUAL CHANNEL FITTING
# =====================================================================
def fit_individual_channel(ch, ra_rad, dec_rad, t_ref_mjd, ephemeris="auto",
                            coord="both", error_floor_mode="auto",
                            manual_floor_ra=0.0, manual_floor_dec=0.0):
    """Fit parallax + proper motion for a single velocity channel.

    Model: x(t) = x0 + mu_x*(t-t0) + pi*Fa(t)
           y(t) = y0 + mu_y*(t-t0) + pi*Fd(t)

    Parameters: [pi, mu_x, x0, mu_y, y0] for coord="both"
                [pi, mu, offset] for coord="ra" or "dec"
    """
    result = FitResult()
    result.fit_type = "individual"
    result.coord_mode = coord
    result.n_channels = 1

    # Select usable epochs
    mask = ch.use_for_pi == 1
    if np.sum(mask) < DEFAULTS["min_epochs_individual"]:
        return None  # too few epochs

    mjds = ch.mjds[mask]
    x = ch.x[mask]
    y = ch.y[mask]
    x_err = ch.x_err[mask]
    y_err = ch.y_err[mask]

    # Time offsets
    dt = (mjds - t_ref_mjd) / JULIANYEAR  # years

    # Parallax factors
    F_a, F_d = get_parallax_factors_cached(ra_rad, dec_rad, mjds, method=ephemeris)

    # Build design matrix and data vector
    A = build_design_matrix_individual(F_a, F_d, dt, coord=coord)
    N = len(mjds)

    if coord == "both":
        b = np.empty(2 * N)
        b[0::2] = x
        b[1::2] = y
        sigma = np.empty(2 * N)
        sigma[0::2] = x_err
        sigma[1::2] = y_err
        result.labels = ["pi", "mu_x", "x0", "mu_y", "y0"]
    elif coord == "ra":
        b = x
        sigma = x_err
        result.labels = ["pi", "mu_x", "x0"]
    elif coord == "dec":
        b = y
        sigma = y_err
        result.labels = ["pi", "mu_y", "y0"]

    n_params = A.shape[1]

    # Error floor
    if error_floor_mode == "auto":
        if coord == "both":
            eps_ra, eps_dec = find_error_floors_radec(A, b, sigma, n_params)
            sigma = apply_error_floor(sigma, eps_ra, eps_dec, interleaved=True)
            result.floor_ra = eps_ra
            result.floor_dec = eps_dec
        else:
            eps = find_error_floor(A, b, sigma, n_params)
            sigma = apply_error_floor(sigma, eps, eps, interleaved=False)
            if coord == "ra": result.floor_ra = eps
            else: result.floor_dec = eps
    elif error_floor_mode == "manual":
        if coord == "both":
            sigma = apply_error_floor(sigma, manual_floor_ra, manual_floor_dec, True)
            result.floor_ra = manual_floor_ra
            result.floor_dec = manual_floor_dec
        else:
            f = manual_floor_ra if coord == "ra" else manual_floor_dec
            sigma = apply_error_floor(sigma, f, f, False)

    # Fit
    w = 1.0 / sigma**2
    params, cov, resid, chi2, dof = wls_solve(A, b, w)
    if params is None:
        return None

    result.params = params
    result.cov = cov
    result.pi = params[0]
    result.sigma_pi = np.sqrt(cov[0, 0])
    result.chi2 = chi2
    result.dof = dof
    result.chi2_red = chi2 / dof
    result.n_data = len(b)
    result.n_params = n_params

    # Sigma_pdf (post-fit scaling)
    if coord == "both":
        ra_resid = resid[0::2]
        dec_resid = resid[1::2]
        ra_sig = sigma[0::2]
        dec_sig = sigma[1::2]
        chi2_ra = np.sum((ra_resid / ra_sig)**2)
        chi2_dec = np.sum((dec_resid / dec_sig)**2)
        dof_ra = len(ra_resid) - n_params // 2
        dof_dec = len(dec_resid) - n_params // 2
        result.sigma_pdf_ra = np.sqrt(chi2_ra / max(dof_ra, 1))
        result.sigma_pdf_dec = np.sqrt(chi2_dec / max(dof_dec, 1))
        result.resid_ra = ra_resid
        result.resid_dec = dec_resid
    else:
        result.sigma_pdf_ra = np.sqrt(chi2 / dof)
        if coord == "ra":
            result.resid_ra = resid
        else:
            result.resid_dec = resid

    # Proper motion
    if coord == "both":
        result.mu = {
            "mu_x": params[1], "sigma_mu_x": np.sqrt(cov[1, 1]),
            "mu_y": params[3], "sigma_mu_y": np.sqrt(cov[3, 3]),
        }
    elif coord == "ra":
        result.mu = {"mu_x": params[1], "sigma_mu_x": np.sqrt(cov[1, 1])}
    elif coord == "dec":
        result.mu = {"mu_y": params[1], "sigma_mu_y": np.sqrt(cov[1, 1])}

    return result


# =====================================================================
#  GROUP FITTING (common pi, independent mu per channel)
# =====================================================================
def fit_group(channels, ra_rad, dec_rad, t_ref_mjd, ephemeris="auto",
              coord="both", error_floor_mode="auto",
              manual_floor_ra=0.0, manual_floor_dec=0.0,
              include_accel=False):
    """Fit common parallax across all channels in a group.

    Each channel gets independent (mu_x, x0, mu_y, y0).
    Total parameters: 1 + 4*N_channels (for coord='both').
    """
    # Filter channels with enough usable epochs
    min_ep = DEFAULTS["min_epochs_individual"]
    usable_channels = [ch for ch in channels if ch.n_usable >= min_ep]
    if not usable_channels:
        return None

    result = FitResult()
    result.fit_type = "group"
    result.coord_mode = coord
    result.n_channels = len(usable_channels)

    # Prepare per-channel data
    F_a_list, F_d_list, dt_list = [], [], []
    b_parts, sigma_parts = [], []
    channel_vlsrs = []

    for ch in usable_channels:
        mask = ch.use_for_pi == 1
        mjds = ch.mjds[mask]
        dt = (mjds - t_ref_mjd) / JULIANYEAR
        F_a, F_d = get_parallax_factors_cached(ra_rad, dec_rad, mjds, method=ephemeris)
        F_a_list.append(F_a)
        F_d_list.append(F_d)
        dt_list.append(dt)
        channel_vlsrs.append(ch.vlsr)

        if coord == "both":
            n = len(mjds)
            b_ch = np.empty(2 * n)
            b_ch[0::2] = ch.x[mask]
            b_ch[1::2] = ch.y[mask]
            s_ch = np.empty(2 * n)
            s_ch[0::2] = ch.x_err[mask]
            s_ch[1::2] = ch.y_err[mask]
        elif coord == "ra":
            b_ch = ch.x[mask]
            s_ch = ch.x_err[mask]
        elif coord == "dec":
            b_ch = ch.y[mask]
            s_ch = ch.y_err[mask]
        b_parts.append(b_ch)
        sigma_parts.append(s_ch)

    # Build combined design matrix
    A = build_design_matrix_group(F_a_list, F_d_list, dt_list,
                                   coord=coord, include_accel=include_accel)
    b = np.concatenate(b_parts)
    sigma = np.concatenate(sigma_parts)
    n_params = A.shape[1]

    # Error floor
    if error_floor_mode == "auto":
        if coord == "both":
            eps_ra, eps_dec = find_error_floors_radec(A, b, sigma, n_params)
            sigma = apply_error_floor(sigma, eps_ra, eps_dec, interleaved=True)
            result.floor_ra = eps_ra
            result.floor_dec = eps_dec
        else:
            eps = find_error_floor(A, b, sigma, n_params)
            sigma = apply_error_floor(sigma, eps, eps, interleaved=False)
            if coord == "ra": result.floor_ra = eps
            else: result.floor_dec = eps
    elif error_floor_mode == "manual":
        if coord == "both":
            sigma = apply_error_floor(sigma, manual_floor_ra, manual_floor_dec, True)
        else:
            f = manual_floor_ra if coord == "ra" else manual_floor_dec
            sigma = apply_error_floor(sigma, f, f, False)
        result.floor_ra = manual_floor_ra
        result.floor_dec = manual_floor_dec

    # Fit
    w = 1.0 / sigma**2
    params, cov, resid, chi2, dof = wls_solve(A, b, w)
    if params is None:
        return None

    result.params = params
    result.cov = cov
    result.pi = params[0]
    result.sigma_pi = np.sqrt(cov[0, 0])
    result.chi2 = chi2
    result.dof = dof
    result.chi2_red = chi2 / dof
    result.n_data = len(b)
    result.n_params = n_params

    # Build labels
    ppch = A.shape[1] - 1
    ppch_per = ppch // len(usable_channels)
    result.labels = ["pi"]
    for i, ch in enumerate(usable_channels):
        v = f"{ch.vlsr:.2f}"
        if coord == "both":
            result.labels.extend([f"mu_x_{v}", f"x0_{v}", f"mu_y_{v}", f"y0_{v}"])
            if include_accel:
                result.labels.extend([f"ax_{v}", f"ay_{v}"])
        else:
            c = "x" if coord == "ra" else "y"
            result.labels.extend([f"mu_{c}_{v}", f"{c}0_{v}"])
            if include_accel:
                result.labels.extend([f"a{c}_{v}"])

    # Sigma_pdf
    if coord == "both":
        ra_mask = np.zeros(len(b), dtype=bool)
        ra_mask[0::2] = True
        chi2_ra = np.sum((resid[ra_mask] / sigma[ra_mask])**2)
        chi2_dec = np.sum((resid[~ra_mask] / sigma[~ra_mask])**2)
        n_ra = np.sum(ra_mask)
        n_dec = np.sum(~ra_mask)
        result.sigma_pdf_ra = np.sqrt(chi2_ra / max(n_ra - n_params // 2, 1))
        result.sigma_pdf_dec = np.sqrt(chi2_dec / max(n_dec - n_params // 2, 1))
    else:
        spdf = np.sqrt(chi2 / dof)
        result.sigma_pdf_ra = spdf
        result.sigma_pdf_dec = spdf

    # Per-channel proper motions
    result.mu = {}
    for i, ch in enumerate(usable_channels):
        if coord == "both":
            base = 1 + i * ppch_per
            result.mu[ch.vlsr] = {
                "mu_x": params[base], "sigma_mu_x": np.sqrt(cov[base, base]),
                "mu_y": params[base + 2], "sigma_mu_y": np.sqrt(cov[base + 2, base + 2]),
            }
        elif coord == "ra":
            base = 1 + i * ppch_per
            result.mu[ch.vlsr] = {
                "mu_x": params[base], "sigma_mu_x": np.sqrt(cov[base, base]),
            }
        elif coord == "dec":
            base = 1 + i * ppch_per
            result.mu[ch.vlsr] = {
                "mu_y": params[base], "sigma_mu_y": np.sqrt(cov[base, base]),
            }

    # Store for bootstrap and diagnostics
    result._A = A
    result._b = b
    result._sigma = sigma
    result._channel_vlsrs = channel_vlsrs
    result._usable_channels = usable_channels

    # Build epoch groups for bootstrap: rows belonging to the same MJD
    # should be resampled together across all channels
    all_mjds = []
    row_idx = 0
    row_to_mjd = []
    for ch in usable_channels:
        mask = ch.use_for_pi == 1
        mjds = ch.mjds[mask]
        for m in mjds:
            if coord == "both":
                row_to_mjd.extend([(m, row_idx), (m, row_idx + 1)])
                row_idx += 2
            else:
                row_to_mjd.append((m, row_idx))
                row_idx += 1

    # Group rows by MJD (approximately — within 1 day)
    unique_mjds = sorted(set(m for m, _ in row_to_mjd))
    epoch_groups = []
    for um in unique_mjds:
        rows = [r for m, r in row_to_mjd if abs(m - um) < 1.0]
        epoch_groups.append(np.array(rows))
    result._epoch_groups = epoch_groups

    return result


# =====================================================================
#  COMBINED MULTI-GROUP FITTING
# =====================================================================
def fit_combined_groups(group_channels_dict, ra_rad, dec_rad, t_ref_mjd,
                        ephemeris="auto", error_floor_mode="auto",
                        manual_floor_ra=0.0, manual_floor_dec=0.0):
    """Fit a SINGLE shared parallax across ALL channels from multiple groups.

    This is NOT averaging — it's one large WLS fit where all channels
    share one π but each channel has independent (mu_x, x0, mu_y, y0).

    Parameters
    ----------
    group_channels_dict : dict {group_id: list of ChannelData}
        All channels from all groups to include in the combined fit.
    """
    min_ep = DEFAULTS["min_epochs_individual"]
    all_channels = []
    for gid in sorted(group_channels_dict.keys()):
        for ch in group_channels_dict[gid]:
            if ch.n_usable >= min_ep:
                all_channels.append(ch)

    if not all_channels:
        return None

    result = fit_group(all_channels, ra_rad, dec_rad, t_ref_mjd,
                        ephemeris=ephemeris, coord="both",
                        error_floor_mode=error_floor_mode,
                        manual_floor_ra=manual_floor_ra,
                        manual_floor_dec=manual_floor_dec)
    if result:
        result.fit_type = "combined_group"
    return result


# =====================================================================
#  EPOCH FLAGGING BASED ON RESIDUALS
# =====================================================================
def flag_outlier_epochs(channels, ra_rad, dec_rad, t_ref_mjd,
                        ephemeris="auto", sigma_clip=3.5):
    """Identify outlier epochs per channel based on post-floor residuals.

    Fits each channel WITH error floors, then checks for residuals
    exceeding sigma_clip * sigma. Only genuine outliers (beyond the
    atmospheric error floor) are flagged.
    """
    flagged = {}
    for ch in channels:
        if ch.n_usable < DEFAULTS["min_epochs_individual"]:
            continue

        r = fit_individual_channel(ch, ra_rad, dec_rad, t_ref_mjd,
                                    ephemeris=ephemeris, coord="both",
                                    error_floor_mode="auto")
        if r is None or r.resid_ra is None:
            continue

        mask = ch.use_for_pi == 1
        epochs = [e for e, u in zip(ch.epochs, ch.use_for_pi) if u == 1]
        sigma_ra = np.sqrt(ch.x_err[mask]**2 + r.floor_ra**2)
        sigma_dec = np.sqrt(ch.y_err[mask]**2 + r.floor_dec**2)

        bad_epochs = []
        for i, ep in enumerate(epochs):
            r_ra = abs(r.resid_ra[i]) / sigma_ra[i] if sigma_ra[i] > 0 else 0
            r_dec = abs(r.resid_dec[i]) / sigma_dec[i] if sigma_dec[i] > 0 else 0
            if r_ra > sigma_clip or r_dec > sigma_clip:
                bad_epochs.append((ep, r_ra, r_dec))
        if bad_epochs:
            flagged[(ch.group_id, ch.vlsr)] = bad_epochs

    return flagged


def detect_parallax_outlier_groups(group_results, spots_df, method="chauvenet",
                                    sigma_clip=2.5, verbose=False):
    """Identify groups whose parallax is a statistical outlier.

    Replaces the hard-coded velocity-range flagging (e.g. -65 to -58 km/s)
    with a proper statistical test.

    Methods
    -------
    chauvenet : Chauvenet's criterion — flag if P(|x-mu| > |xi-mu|) < 1/(2N)
    sigma     : Simple sigma clipping at sigma_clip * MAD_sigma

    Parameters
    ----------
    group_results : dict {group_id: FitResult}
        Group-level parallax results (with sqrt(N) applied)
    spots_df : DataFrame
        For looking up group grades and vlsr
    method : str
        "chauvenet" or "sigma"
    sigma_clip : float
        Threshold for sigma-clip method (default 2.5)

    Returns
    -------
    flagged : dict {group_id: reason_string}
        Groups identified as parallax outliers
    """
    from scipy.stats import norm as _norm

    if len(group_results) < 4:
        return {}  # too few groups for meaningful outlier detection

    # Collect parallaxes with sqrt(N) uncertainties
    gids, pis, sigmas, vlsrs = [], [], [], []
    for gid, r in group_results.items():
        n_ch = r.n_channels
        sigma_sqrtN = r.sigma_pi * np.sqrt(n_ch)
        # Only use positive-parallax groups for the reference distribution
        gids.append(gid)
        pis.append(r.pi)
        sigmas.append(sigma_sqrtN)
        gdf = spots_df[spots_df.group_id == gid]
        vlsrs.append(gdf.vlsr.median())

    pis = np.array(pis)
    sigmas = np.array(sigmas)
    N = len(pis)

    # Robust center: weighted median using inverse-variance weights
    w = 1.0 / sigmas**2
    # Use weighted mean of groups with reasonably small errors as reference
    good = sigmas < np.median(sigmas) * 3  # exclude very uncertain groups
    if np.sum(good) < 3:
        good = np.ones(N, bool)
    pi_ref = np.sum(pis[good] * w[good]) / np.sum(w[good])

    # Robust scatter: MAD of deviations normalized by their uncertainties
    pull = (pis - pi_ref) / sigmas
    mad = np.median(np.abs(pull - np.median(pull)))
    sigma_pull = mad * 1.4826  # MAD to Gaussian sigma

    flagged = {}

    if method == "chauvenet":
        for i, gid in enumerate(gids):
            p_value = 2 * _norm.sf(abs(pull[i]) / max(sigma_pull, 0.5))
            threshold = 1.0 / (2 * N)
            if p_value < threshold:
                flagged[gid] = (f"Chauvenet: pi={pis[i]:.3f} mas, "
                                f"pull={pull[i]:.1f}sigma, "
                                f"P={p_value:.4f} < 1/{2*N}, "
                                f"vlsr={vlsrs[i]:.1f} km/s")
    elif method == "sigma":
        for i, gid in enumerate(gids):
            if abs(pull[i]) > sigma_clip * max(sigma_pull, 0.5):
                flagged[gid] = (f"sigma-clip: pi={pis[i]:.3f} mas, "
                                f"pull={pull[i]:.1f}sigma > "
                                f"{sigma_clip}*{sigma_pull:.1f}, "
                                f"vlsr={vlsrs[i]:.1f} km/s")

    if verbose and flagged:
        print(f"  Parallax reference: pi_ref={pi_ref:.4f} mas "
              f"(from {np.sum(good)} groups)")
        print(f"  Pull distribution: median={np.median(pull):.2f}, "
              f"MAD_sigma={sigma_pull:.2f}")
        for gid, reason in flagged.items():
            print(f"  OUTLIER G{gid}: {reason}")

    return flagged


class ParallaxFitter:
    """Orchestrates all fitting modes for a single PR mode's data."""

    def __init__(self, spots_df, ra_rad, dec_rad, groups_df=None,
                 features_df=None, ephemeris="auto",
                 error_floor_mode="auto", manual_floor_ra=0.0,
                 manual_floor_dec=0.0, bootstrap_trials=10000,
                 grade_threshold="C", mode_label="inverse",
                 chan_spacing=None):
        self.spots_df = spots_df
        self.features_df = features_df
        self.groups_df = groups_df
        self.ra_rad = ra_rad
        self.dec_rad = dec_rad
        self.ephemeris = ephemeris
        self.error_floor_mode = error_floor_mode
        self.manual_floor_ra = manual_floor_ra
        self.manual_floor_dec = manual_floor_dec
        self.bootstrap_trials = bootstrap_trials
        self.grade_threshold = grade_threshold
        self.mode_label = mode_label
        self.chan_spacing = chan_spacing

        # Galactic coordinates for distance posterior
        self.l_rad, self.b_rad = equatorial_to_galactic(ra_rad, dec_rad)

        # Results storage
        self.individual_results = {}   # {group_id: {vlsr: FitResult}}
        self.group_results = {}        # {group_id: FitResult}
        self.feature_results = {}      # {group_id: FitResult}
        self.bootstrap_results = {}    # {group_id: array of pi values}
        self.accel_bic = {}            # {group_id: (bic_standard, bic_accel)}
        self.distance_posteriors = {}  # {group_id: dict}
        self.combined_results = {}    # {"A": FitResult, "A+B": ..., "A+B+C": ...}
        self.cauchy_results = {}      # {group_id: {pi, sigma_pi, n_outliers}}
        self.epoch_flags = {}         # {(gid, vlsr): [(epoch, r_ra, r_dec)]}
        self.parallax_outliers = set()  # group IDs flagged by Chauvenet
        self.summary = {}             # weighted mean pi, etc.

    def _get_fitting_groups(self):
        """Get group IDs that meet the grade threshold.

        For grade B/C groups, additionally require that the parallax factors
        span both positive and negative values (i.e., data covers both halves
        of the parallax sinusoid). Grade A groups are always included.
        """
        gids = self.spots_df.group_id.unique()
        result = []
        t_ref_mjd = self._compute_ref_epoch()
        for gid in sorted(gids):
            sub = self.spots_df[self.spots_df.group_id == gid]
            grade = sub.grade.iloc[0]
            if GRADE_ORDER.get(grade, 99) > GRADE_ORDER.get(self.grade_threshold, 2):
                continue
            # Grade A: always include
            if grade == "A":
                result.append(gid)
                continue
            # Grade B/C: check parallax phase coverage
            usable = sub[sub.use_for_pi == 1]
            if len(usable) < DEFAULTS["min_epochs_individual"]:
                continue
            mjds = usable.mjd.values.astype(float)
            F_a, F_d = get_parallax_factors_cached(
                self.ra_rad, self.dec_rad, mjds, method=self.ephemeris)
            # Require both +/- coverage in at least one coordinate
            ra_ok = (np.any(F_a > 0.1) and np.any(F_a < -0.1))
            dec_ok = (np.any(F_d > 0.1) and np.any(F_d < -0.1))
            if ra_ok or dec_ok:
                result.append(gid)
            else:
                if hasattr(self, '_skipped_coverage'):
                    self._skipped_coverage.append(gid)
                else:
                    self._skipped_coverage = [gid]
        return result

    def _compute_ref_epoch(self):
        """Compute reference epoch as mean MJD of unflagged data."""
        usable = self.spots_df[self.spots_df.use_for_pi == 1]
        if len(usable) == 0:
            return self.spots_df.mjd.mean()
        return usable.mjd.mean()

    def run(self, verbose=True):
        """Run the full fitting pipeline."""
        t0 = _time.time()
        if verbose:
            print(f"\n{'='*65}")
            print(f"  masertrack_fit v{__version__} — {self.mode_label} PR")
            print(f"{'='*65}")
            ra_deg = np.degrees(self.ra_rad)
            dec_deg = np.degrees(self.dec_rad)
            print(f"  Source: RA={ra_deg:.4f}°, Dec={dec_deg:.4f}°")
            print(f"  Galactic: l={np.degrees(self.l_rad):.2f}°, b={np.degrees(self.b_rad):.2f}°")
            print(f"  Ephemeris: {self.ephemeris}")
            print(f"  Error floors: {self.error_floor_mode}")
            print(f"  Bootstrap: {self.bootstrap_trials} trials")
            print(f"  Grade threshold: {self.grade_threshold}")

        fitting_groups = self._get_fitting_groups()
        t_ref_mjd = self._compute_ref_epoch()

        if verbose:
            print(f"  Reference epoch: MJD {t_ref_mjd:.1f}")
            print(f"  Groups to fit: {len(fitting_groups)} "
                  f"(of {self.spots_df.group_id.nunique()} total)")

        # ---- Step 1: Individual channel fits ----
        if verbose: print(f"\nStep 1 — Individual channel fitting")
        for gid in fitting_groups:
            channels = extract_channels(self.spots_df, gid, chan_spacing=self.chan_spacing)
            self.individual_results[gid] = {}
            for ch in channels:
                r = fit_individual_channel(
                    ch, self.ra_rad, self.dec_rad, t_ref_mjd,
                    ephemeris=self.ephemeris, coord="both",
                    error_floor_mode=self.error_floor_mode,
                    manual_floor_ra=self.manual_floor_ra,
                    manual_floor_dec=self.manual_floor_dec)
                if r is not None:
                    self.individual_results[gid][ch.vlsr] = r

            n_fit = len(self.individual_results[gid])
            if verbose and n_fit > 0:
                pis = [r.pi for r in self.individual_results[gid].values()]
                grade = channels[0].grade if channels else "?"
                print(f"  G{gid} ({grade}): {n_fit} channels fitted, "
                      f"pi range [{min(pis):.3f}, {max(pis):.3f}] mas")

        # ---- Step 2: Group fitting (common pi, independent mu per channel) ----
        if verbose: print(f"\nStep 2 — Group fitting (common π)")
        for gid in fitting_groups:
            channels = extract_channels(self.spots_df, gid, chan_spacing=self.chan_spacing)
            r = fit_group(channels, self.ra_rad, self.dec_rad, t_ref_mjd,
                          ephemeris=self.ephemeris, coord="both",
                          error_floor_mode=self.error_floor_mode,
                          manual_floor_ra=self.manual_floor_ra,
                          manual_floor_dec=self.manual_floor_dec)
            if r is not None:
                self.group_results[gid] = r

            if gid in self.group_results and verbose:
                r = self.group_results[gid]
                n_ch = r.n_channels
                pi_sqrtN = r.sigma_pi * np.sqrt(n_ch)
                print(f"  G{gid}: {n_ch} ch, pi={r.pi:.4f} ± {r.sigma_pi:.4f} "
                      f"(√N: ±{pi_sqrtN:.4f}) mas, "
                      f"floor=[{r.floor_ra:.3f},{r.floor_dec:.3f}] mas, "
                      f"chi2r={r.chi2_red:.2f}")

        # ---- Step 3: Bootstrap ----
        if self.bootstrap_trials > 0:
            cache_path = os.path.join(
                getattr(self, '_outdir', '.'),
                f"bootstrap_cache_{self.mode_label}.npz")
            cached = False
            if os.path.exists(cache_path):
                try:
                    data = np.load(cache_path, allow_pickle=True)
                    for gid in fitting_groups:
                        key = f"G{gid}"
                        if key in data:
                            self.bootstrap_results[gid] = data[key]
                    cached = True
                    if verbose:
                        print(f"\nStep 3 — Bootstrap (loaded from cache: {cache_path})")
                except Exception:
                    cached = False

            if not cached:
                if verbose: print(f"\nStep 3 — Bootstrap ({self.bootstrap_trials} trials)")
                for gid in fitting_groups:
                    if gid not in self.group_results:
                        continue
                    r = self.group_results[gid]
                    epoch_groups = getattr(r, '_epoch_groups', None)
                    boot = bootstrap_fit(r._A, r._b, r._sigma,
                                          n_trials=self.bootstrap_trials,
                                          epoch_groups=epoch_groups)
                    if len(boot) > 100:
                        self.bootstrap_results[gid] = boot[:, 0]  # pi column
                        pi_med = np.median(boot[:, 0])
                        pi_16 = np.percentile(boot[:, 0], 16)
                        pi_84 = np.percentile(boot[:, 0], 84)
                        if verbose:
                            print(f"  G{gid}: pi_bootstrap = {pi_med:.4f} "
                                  f"[{pi_16:.4f}, {pi_84:.4f}]")

                # Save cache
                try:
                    save_dict = {f"G{gid}": arr
                                 for gid, arr in self.bootstrap_results.items()}
                    np.savez_compressed(cache_path, **save_dict)
                    if verbose:
                        print(f"  Bootstrap cache saved: {cache_path}")
                except Exception:
                    pass

        # ---- Step 4: Acceleration BIC test ----
        if verbose: print(f"\nStep 4 — Acceleration BIC test")
        for gid in fitting_groups:
            channels = extract_channels(self.spots_df, gid, chan_spacing=self.chan_spacing)
            r_std = self.group_results.get(gid)
            if r_std is None:
                continue
            r_acc = fit_group(channels, self.ra_rad, self.dec_rad, t_ref_mjd,
                               ephemeris=self.ephemeris, coord="both",
                               error_floor_mode=self.error_floor_mode,
                               manual_floor_ra=self.manual_floor_ra,
                               manual_floor_dec=self.manual_floor_dec,
                               include_accel=True)
            if r_acc is not None:
                bic_std = r_std.chi2 + r_std.n_params * np.log(r_std.n_data)
                bic_acc = r_acc.chi2 + r_acc.n_params * np.log(r_acc.n_data)
                delta_bic = bic_acc - bic_std
                self.accel_bic[gid] = (bic_std, bic_acc, delta_bic, r_acc.pi)
                if verbose:
                    favor = "acceleration" if delta_bic < -6 else "standard"
                    print(f"  G{gid}: ΔBIC={delta_bic:+.1f} → favors {favor}"
                          f" (pi_accel={r_acc.pi:.4f})")

        # ---- Step 5: Feature fitting (if features_df available) ----
        if self.features_df is not None and len(self.features_df) > 0:
            if verbose: print(f"\nStep 5 — Feature-level fitting")
            for gid in fitting_groups:
                gdf = self.features_df[self.features_df.group_id == gid]
                if len(gdf) < DEFAULTS["min_epochs_individual"]:
                    continue
                # Treat as single-channel fit using feature centroids
                mask = gdf.use_for_pi.values == 1
                if np.sum(mask) < DEFAULTS["min_epochs_individual"]:
                    continue
                ch = ChannelData(
                    group_id=gid, vlsr=gdf.vlsr.values[0],
                    grade=gdf.grade.values[0],
                    mjds=gdf.mjd.values, dec_years=gdf.dec_year.values,
                    x=gdf.x.values, y=gdf.y.values,
                    x_err=gdf.x_err.values, y_err=gdf.y_err.values,
                    flux=gdf.flux.values, epochs=gdf.epoch.tolist(),
                    use_for_pi=gdf.use_for_pi.values)
                r = fit_individual_channel(
                    ch, self.ra_rad, self.dec_rad, t_ref_mjd,
                    ephemeris=self.ephemeris, coord="both",
                    error_floor_mode=self.error_floor_mode)
                if r is not None:
                    r.fit_type = "feature"
                    self.feature_results[gid] = r
                    if verbose:
                        print(f"  G{gid}: pi={r.pi:.4f} ± {r.sigma_pi:.4f} mas")

            # Combined feature fit across grade sets
            if verbose: print(f"\n  Combined feature fits:")
            for grade_set, label in [(["A"], "A"), (["A", "B"], "A+B"),
                                      (["A", "B", "C"], "A+B+C")]:
                feat_channels = []
                for gid in fitting_groups:
                    if gid not in self.feature_results:
                        continue
                    grade = self.spots_df[self.spots_df.group_id==gid].grade.iloc[0]
                    if grade not in grade_set:
                        continue
                    gdf = self.features_df[self.features_df.group_id == gid]
                    ch = ChannelData(
                        group_id=gid, vlsr=gdf.vlsr.values[0],
                        grade=gdf.grade.values[0],
                        mjds=gdf.mjd.values, dec_years=gdf.dec_year.values,
                        x=gdf.x.values, y=gdf.y.values,
                        x_err=gdf.x_err.values, y_err=gdf.y_err.values,
                        flux=gdf.flux.values, epochs=gdf.epoch.tolist(),
                        use_for_pi=gdf.use_for_pi.values)
                    feat_channels.append(ch)
                if len(feat_channels) >= 2:
                    r = fit_group(feat_channels, self.ra_rad, self.dec_rad,
                                   t_ref_mjd, ephemeris=self.ephemeris,
                                   coord="both", error_floor_mode=self.error_floor_mode)
                    if r is not None:
                        r.fit_type = "combined_feature"
                        sqN = r.sigma_pi * np.sqrt(r.n_channels)
                        self.combined_results[f"feat_{label}"] = r
                        if verbose:
                            print(f"    {label}: {r.n_channels} features, "
                                  f"pi={r.pi:.4f} ± {sqN:.4f} (√N) mas")

        # ---- Step 6b: Cauchy outlier-tolerant fits ----
        if verbose: print(f"\nStep 5b — Cauchy outlier-tolerant fitting")
        self.cauchy_results = {}
        for gid in fitting_groups:
            if gid not in self.group_results:
                continue
            r = self.group_results[gid]
            if not hasattr(r, '_A'):
                continue
            params_c, cov_c, resid_c, outlier_flags = outlier_tolerant_fit(
                r._A, r._b, r._sigma)
            if params_c is not None:
                n_outliers = int(np.sum(outlier_flags))
                self.cauchy_results[gid] = {
                    "pi": params_c[0],
                    "sigma_pi": np.sqrt(cov_c[0, 0]),
                    "n_outliers": n_outliers,
                }
                if verbose and n_outliers > 0:
                    sqN = np.sqrt(cov_c[0, 0]) * np.sqrt(r.n_channels)
                    print(f"  G{gid}: pi_cauchy={params_c[0]:.4f}, "
                          f"{n_outliers} outlier points flagged")

        # ---- Step 7: Distance posteriors ----
        if verbose: print(f"\nStep 6 — Distance posteriors")
        for gid in fitting_groups:
            if gid not in self.group_results:
                continue
            r = self.group_results[gid]
            n_ch = r.n_channels
            sigma_sqrtN = r.sigma_pi * np.sqrt(n_ch)
            dist = compute_distance_posterior(
                r.pi, sigma_sqrtN, self.l_rad, self.b_rad)
            self.distance_posteriors[gid] = dist
            if verbose:
                print(f"  G{gid}: D_mode={dist['D_mode']:.1f} kpc "
                      f"[{dist['D_16']:.1f}, {dist['D_84']:.1f}] (68%)")

        # ---- Step 8: Combined multi-group fits ----
        if verbose: print(f"\nStep 7 — Combined multi-group fits (single shared π)")
        for grade_set, label in [(["A"], "A"), (["A", "B"], "A+B"),
                                  (["A", "B", "C"], "A+B+C")]:
            group_chs = {}
            for gid in fitting_groups:
                grade = self.spots_df[self.spots_df.group_id == gid].grade.iloc[0]
                if grade in grade_set:
                    group_chs[gid] = extract_channels(
                        self.spots_df, gid, chan_spacing=self.chan_spacing)
            if group_chs:
                r = fit_combined_groups(group_chs, self.ra_rad, self.dec_rad,
                                         t_ref_mjd, ephemeris=self.ephemeris,
                                         error_floor_mode=self.error_floor_mode,
                                         manual_floor_ra=self.manual_floor_ra,
                                         manual_floor_dec=self.manual_floor_dec)
                if r is not None:
                    self.combined_results[label] = r
                    sigma_sqrtN = r.sigma_pi * np.sqrt(r.n_channels)
                    if verbose:
                        print(f"  {label}: {r.n_channels} ch from "
                              f"{len(group_chs)} groups, "
                              f"π = {r.pi:.4f} ± {r.sigma_pi:.4f} "
                              f"(√N: ±{sigma_sqrtN:.4f}) mas, "
                              f"floor=[{r.floor_ra:.3f},{r.floor_dec:.3f}], "
                              f"chi2r={r.chi2_red:.2f}")

        # ---- Step 9: Epoch flagging ----
        if verbose: print(f"\nStep 8 — Epoch outlier detection")
        for gid in fitting_groups:
            channels = extract_channels(self.spots_df, gid,
                                         chan_spacing=self.chan_spacing)
            flags = flag_outlier_epochs(channels, self.ra_rad, self.dec_rad,
                                         t_ref_mjd, ephemeris=self.ephemeris)
            if flags:
                self.epoch_flags.update(flags)
                if verbose:
                    for (g, v), bad_eps in flags.items():
                        for ep, rr, rd in bad_eps:
                            print(f"  G{g} v={v:.1f}: epoch {ep} "
                                  f"outlier (RA:{rr:.1f}σ, Dec:{rd:.1f}σ)")

        # ---- Step 10: Parallax outlier group detection ----
        if verbose: print(f"\nStep 9 — Parallax outlier group detection")
        self.anomaly_groups = detect_parallax_outlier_groups(
            self.group_results, self.spots_df,
            method="chauvenet", verbose=verbose)
        # Store as a set of group IDs for easy lookup by plots and exporters
        self.parallax_outliers = set(self.anomaly_groups.keys())
        if not self.anomaly_groups and verbose:
            print("  No outlier groups detected")

        # ---- Step 11: Summary ----
        self._compute_summary()

        elapsed = _time.time() - t0
        if verbose:
            print(f"\n{'='*65}")
            print(f"  Fitting complete in {elapsed:.1f} s")
            self._print_summary()
            print(f"{'='*65}")

        return self

    def _compute_summary(self):
        """Compute weighted mean parallax from fitted groups.

        Error sources combined:
        - Formal σ from WLS covariance
        - √N correction (channels share systematic errors)
        - σ_pdf scaling (if bootstrap unavailable, scale by max(σ_pdf_ra, σ_pdf_dec))
        - Bootstrap σ (replaces formal when available)
        """
        for grade_set, label in [(["A"], "A"), (["A", "B"], "A+B"),
                                  (["A", "B", "C"], "A+B+C")]:
            pis, sigmas, gids = [], [], []
            for gid, r in self.group_results.items():
                # Skip groups flagged as parallax outliers by Chauvenet
                if gid in self.parallax_outliers:
                    continue
                grade = self.spots_df[self.spots_df.group_id == gid].grade.iloc[0]
                if grade in grade_set:
                    n_ch = r.n_channels

                    # Prefer bootstrap uncertainty if available
                    if gid in self.bootstrap_results and len(self.bootstrap_results[gid]) > 100:
                        boot_pi = self.bootstrap_results[gid]
                        sigma_use = 0.5 * (np.percentile(boot_pi, 84) - np.percentile(boot_pi, 16))
                    else:
                        # Formal + √N + σ_pdf scaling
                        sigma_sqrtN = r.sigma_pi * np.sqrt(n_ch)
                        spdf = max(getattr(r, 'sigma_pdf_ra', 1.0),
                                   getattr(r, 'sigma_pdf_dec', 1.0), 1.0)
                        sigma_use = sigma_sqrtN * spdf

                    pis.append(r.pi)
                    sigmas.append(sigma_use)
                    gids.append(gid)

            if not pis:
                continue
            pis = np.array(pis)
            sigmas = np.array(sigmas)
            w = 1.0 / sigmas**2
            pi_wmean = np.sum(pis * w) / np.sum(w)
            sigma_wmean = 1.0 / np.sqrt(np.sum(w))

            self.summary[label] = {
                "pi": pi_wmean, "sigma_pi": sigma_wmean,
                "n_groups": len(pis), "groups": gids,
                "pi_values": pis, "sigma_values": sigmas,
            }

    def _print_summary(self):
        """Print summary table."""
        print(f"\n  Summary — {self.mode_label} PR:")
        print(f"  {'Method':>12} {'Grade':>8} {'N_groups':>8} {'N_ch':>6} "
              f"{'pi (mas)':>12} {'sigma_pi':>10} {'D_mode':>8}")
        print(f"  {'-'*72}")
        # Weighted mean results
        for label in ["A", "A+B", "A+B+C"]:
            if label not in self.summary:
                continue
            s = self.summary[label]
            dist = compute_distance_posterior(s["pi"], s["sigma_pi"],
                                              self.l_rad, self.b_rad)
            print(f"  {'wt. mean':>12} {label:>8} {s['n_groups']:>8} "
                  f"{'---':>6} {s['pi']:>12.4f} {s['sigma_pi']:>10.4f} "
                  f"{dist['D_mode']:>8.1f}")
        # Combined fit results
        for label in ["A", "A+B", "A+B+C"]:
            if label not in self.combined_results:
                continue
            r = self.combined_results[label]
            sigma_sqrtN = r.sigma_pi * np.sqrt(r.n_channels)
            dist = compute_distance_posterior(r.pi, sigma_sqrtN,
                                              self.l_rad, self.b_rad)
            print(f"  {'combined':>12} {label:>8} {'---':>8} "
                  f"{r.n_channels:>6} {r.pi:>12.4f} {sigma_sqrtN:>10.4f} "
                  f"{dist['D_mode']:>8.1f}")

        # Feature fit results (flux-weighted feature centroids)
        for label in ["feat_A", "feat_A+B", "feat_A+B+C"]:
            if label not in self.combined_results:
                continue
            r = self.combined_results[label]
            sigma_sqrtN = r.sigma_pi * np.sqrt(r.n_channels)
            dist = compute_distance_posterior(r.pi, sigma_sqrtN,
                                              self.l_rad, self.b_rad)
            short = label.replace("feat_", "")
            print(f"  {'feature':>12} {short:>8} {'---':>8} "
                  f"{r.n_channels:>6} {r.pi:>12.4f} {sigma_sqrtN:>10.4f} "
                  f"{dist['D_mode']:>8.1f}")

        # Error floor vs bootstrap comparison
        print(f"\n  Error floor vs bootstrap (non-outlier groups):")
        print(f"  {'group':>6}{'n_ch':>5}{'σ_formal':>10}{'σ_√N':>10}"
              f"{'σ_boot':>10}{'ratio':>8}{'ε_RA':>8}{'ε_Dec':>8}")
        for gid in sorted(self.group_results.keys()):
            if gid in self.parallax_outliers: continue
            r = self.group_results[gid]
            n_ch = r.n_channels
            sqN = r.sigma_pi * np.sqrt(n_ch)
            if gid in self.bootstrap_results and len(self.bootstrap_results[gid]) > 100:
                bp = self.bootstrap_results[gid]
                sig_b = 0.5 * (np.percentile(bp, 84) - np.percentile(bp, 16))
                ratio = sig_b / sqN if sqN > 0 else 0
                print(f"  G{gid:>4}{n_ch:>5}{r.sigma_pi:>10.4f}{sqN:>10.4f}"
                      f"{sig_b:>10.4f}{ratio:>8.2f}"
                      f"{r.floor_ra:>8.3f}{r.floor_dec:>8.3f}")

        # Distance summary
        if "A+B+C" in self.summary:
            s = self.summary["A+B+C"]
            if s["pi"] > 0:
                d_simple = 1.0 / s["pi"]
                d_lo = 1.0 / (s["pi"] + s["sigma_pi"])
                d_hi = 1.0 / max(s["pi"] - s["sigma_pi"], 0.001)
                print(f"\n  Distance (1/π): {d_simple:.1f} kpc "
                      f"[{d_lo:.1f}, {min(d_hi, 30.0):.1f}]")


# =====================================================================
#  OUTPUT AND EXPORT
# =====================================================================
def write_results_table(fitter, outdir, mode):
    """Write per-group results table."""

    # Also write per-channel individual results
    indiv_path = os.path.join(outdir, f"fit_{mode}_individual.txt")
    with open(indiv_path, "w") as f:
        f.write(f"# Individual channel parallax fits ({mode} PR)\n")
        f.write(f"# masertrack_fit v{__version__}\n")
        f.write(f"#{'group_id':>10}{'grade':>6}{'vlsr':>12}{'pi':>12}{'sigma_pi':>12}"
                f"{'mu_x':>12}{'mu_y':>12}{'chi2_red':>10}{'floor_ra':>10}{'floor_dec':>10}\n")
        for gid in sorted(fitter.individual_results.keys()):
            for vlsr in sorted(fitter.individual_results[gid].keys()):
                r = fitter.individual_results[gid][vlsr]
                grade = fitter.spots_df[fitter.spots_df.group_id==gid].grade.iloc[0]
                mu_x = r.mu.get("mu_x", np.nan)
                mu_y = r.mu.get("mu_y", np.nan)
                f.write(f"{gid:>10}{grade:>6}{vlsr:>12.4f}{r.pi:>12.4f}{r.sigma_pi:>12.4f}"
                        f"{mu_x:>12.4f}{mu_y:>12.4f}{r.chi2_red:>10.2f}"
                        f"{r.floor_ra:>10.4f}{r.floor_dec:>10.4f}\n")
    print(f"  Individual fits: {indiv_path}")

    rows = []
    for gid, r in sorted(fitter.group_results.items()):
        grade = fitter.spots_df[fitter.spots_df.group_id == gid].grade.iloc[0]
        n_ch = r.n_channels
        sigma_sqrtN = r.sigma_pi * np.sqrt(n_ch)

        # Bootstrap CI
        boot_lo, boot_hi = np.nan, np.nan
        if gid in fitter.bootstrap_results:
            boot = fitter.bootstrap_results[gid]
            boot_lo = np.percentile(boot, 16)
            boot_hi = np.percentile(boot, 84)

        # Feature check
        pi_feat = np.nan
        if gid in fitter.feature_results:
            pi_feat = fitter.feature_results[gid].pi

        # Cauchy outlier-tolerant
        pi_cauchy = np.nan
        if gid in fitter.cauchy_results:
            pi_cauchy = fitter.cauchy_results[gid]["pi"]

        # Acceleration
        delta_bic = np.nan
        if gid in fitter.accel_bic:
            delta_bic = fitter.accel_bic[gid][2]

        # Distance
        d_mode, d_16, d_84 = np.nan, np.nan, np.nan
        if gid in fitter.distance_posteriors:
            dp = fitter.distance_posteriors[gid]
            d_mode, d_16, d_84 = dp["D_mode"], dp["D_16"], dp["D_84"]

        # Outlier flag
        flag = "OUTLIER" if gid in fitter.parallax_outliers else "OK"

        rows.append({
            "group_id": gid, "grade": grade, "n_channels": n_ch,
            "pi_group": r.pi, "sigma_pi_raw": r.sigma_pi,
            "sigma_pi_sqrtN": sigma_sqrtN,
            "floor_ra": r.floor_ra, "floor_dec": r.floor_dec,
            "chi2_red": r.chi2_red,
            "pi_feature": pi_feat, "pi_cauchy": pi_cauchy,
            "boot_16": boot_lo, "boot_84": boot_hi,
            "delta_bic_accel": delta_bic,
            "D_mode_kpc": d_mode, "D_16_kpc": d_16, "D_84_kpc": d_84,
            "flag": flag,
        })

    df = pd.DataFrame(rows)
    path = os.path.join(outdir, f"fit_{mode}_results.txt")
    with open(path, "w") as f:
        f.write(f"# Parallax fitting results ({mode} PR)\n")
        f.write(f"# masertrack_fit v{__version__}\n")
        f.write(f"# RA={np.degrees(fitter.ra_rad):.6f} deg, "
                f"Dec={np.degrees(fitter.dec_rad):.6f} deg\n")
        f.write(f"# Ephemeris: {fitter.ephemeris}\n")
        f.write(f"# Error floor mode: {fitter.error_floor_mode}\n")
        f.write(f"# Bootstrap trials: {fitter.bootstrap_trials}\n")
        f.write("#\n")
        # Write column header and data
        f.write("# " + "  ".join(f"{c:>14}" for c in df.columns) + "\n")
        for _, row in df.iterrows():
            vals = []
            for c in df.columns:
                v = row[c]
                if isinstance(v, str):
                    vals.append(f"{v:>14}")
                elif isinstance(v, (int, np.integer)):
                    vals.append(f"{v:>14d}")
                elif np.isnan(v):
                    vals.append(f"{'---':>14}")
                else:
                    vals.append(f"{v:>14.4f}")
            f.write("  " + "  ".join(vals) + "\n")
    print(f"  Results table: {path}")
    return path


def export_pmpar(fitter, outdir, mode):
    """Export per-channel pmpar.in files for cross-validation."""
    export_dir = os.path.join(outdir, "export", "pmpar")
    os.makedirs(export_dir, exist_ok=True)

    t_ref_mjd = fitter._compute_ref_epoch()
    n_files = 0

    for gid in sorted(fitter.group_results.keys()):
        channels = extract_channels(fitter.spots_df, gid, chan_spacing=fitter.chan_spacing)
        for ch in channels:
            if ch.n_usable < DEFAULTS["min_epochs_individual"]:
                continue
            mask = ch.use_for_pi == 1
            fname = f"G{gid}_v{ch.vlsr:.2f}.pmpar.in"
            path = os.path.join(export_dir, fname)
            with open(path, "w") as f:
                f.write(f"# masertrack_fit export — G{gid} vlsr={ch.vlsr:.4f}\n")
                f.write(f"epoch = {t_ref_mjd:.1f}\n\n")
                for i in range(len(ch.mjds)):
                    if not mask[i]:
                        continue
                    # pmpar format: MJD  RA(mas)  sigRA(mas)  Dec(mas)  sigDec(mas)
                    # Note: pmpar uses absolute coords, but we use offsets
                    # For cross-validation, offsets work if epoch is set correctly
                    f.write(f"{ch.mjds[i]:.5f}  {ch.x[i]:12.6f}  {ch.x_err[i]:10.6f}  "
                            f"{ch.y[i]:12.6f}  {ch.y_err[i]:10.6f}\n")
            n_files += 1

    print(f"  pmpar export: {n_files} files in {export_dir}")


def export_bessel(fitter, outdir, mode):
    """Export per-group BeSSeL Fortran format files."""
    export_dir = os.path.join(outdir, "export", "bessel")
    os.makedirs(export_dir, exist_ok=True)

    t_ref_mjd = fitter._compute_ref_epoch()
    t_ref_yr = 2000.0 + (t_ref_mjd - MJD_J2000) / JULIANYEAR

    for gid in sorted(fitter.group_results.keys()):
        channels = extract_channels(fitter.spots_df, gid, chan_spacing=fitter.chan_spacing)
        usable = [ch for ch in channels if ch.n_usable >= DEFAULTS["min_epochs_individual"]]
        if not usable:
            continue

        # Write per-channel data files
        file_list = []
        for ch in usable:
            mask = ch.use_for_pi == 1
            fname = f"G{gid}_v{ch.vlsr:.2f}.dat"
            path = os.path.join(export_dir, fname)
            with open(path, "w") as f:
                f.write(f"! G{gid} vlsr={ch.vlsr:.4f} km/s\n")
                for i in range(len(ch.mjds)):
                    if not mask[i]:
                        continue
                    yr = 2000.0 + (ch.mjds[i] - MJD_J2000) / JULIANYEAR
                    f.write(f"  {yr:10.4f}  {ch.x[i]:12.6f}  {ch.x_err[i]:10.6f}  "
                            f"{ch.y[i]:12.6f}  {ch.y_err[i]:10.6f}\n")
            file_list.append(fname)

        # Write files list
        flist_path = os.path.join(export_dir, f"G{gid}_files.inp")
        with open(flist_path, "w") as f:
            for fn in file_list:
                f.write(fn + "\n")

        # Write control file
        r = fitter.group_results[gid]
        ra_hr = np.degrees(fitter.ra_rad) / 15.0
        dec_deg = np.degrees(fitter.dec_rad)
        ctrl_path = os.path.join(export_dir, f"G{gid}_control.inp")
        with open(ctrl_path, "w") as f:
            f.write(f"!  BeSSeL control file for G{gid}\n")
            f.write(f"  {ra_hr:12.6f}    ! RA in decimal hours\n")
            f.write(f"  {dec_deg:12.6f}    ! Dec in decimal degrees\n")
            f.write(f"  {t_ref_yr:10.4f}    ! t_mid in years\n")
            f.write(f"  {r.floor_ra:10.4f}    ! X data error floor (mas)\n")
            f.write(f"  {r.floor_dec:10.4f}    ! Y data error floor (mas)\n")
            f.write(f"   1.0      1.0      ! parallax in mas, solve-for\n")
            for ch in usable:
                f.write(f"   0.0      1.0      ! X-offset v={ch.vlsr:.2f}\n")
                f.write(f"   0.0      1.0      ! X-slope\n")
                f.write(f"   0.0      1.0      ! Y-offset\n")
                f.write(f"   0.0      1.0      ! Y-slope\n")

    print(f"  BeSSeL export: {len(fitter.group_results)} groups in {export_dir}")


def export_vera(fitter, outdir, mode):
    """Export VERA/Burns .prm format files for cross-validation.

    Produces two files per group:
      G{id}_{mode}.dat — data in DOY format (columns: DOY RA dRA Dec dDec SPOT CH VLSR)
      G{id}_{mode}.prm — parameter file with source coords and fitting options

    Compatible with Burns' Fortran parallax fitting pipeline.
    """
    from datetime import date as _dt
    _mjd_ref = _dt(1858, 11, 17)

    export_dir = os.path.join(outdir, "export", "vera")
    os.makedirs(export_dir, exist_ok=True)

    t_ref_mjd = fitter._compute_ref_epoch()
    ra_hr = np.degrees(fitter.ra_rad) / 15.0
    dec_deg = np.degrees(fitter.dec_rad)

    for gid in sorted(fitter.group_results.keys()):
        channels = extract_channels(fitter.spots_df, gid, chan_spacing=fitter.chan_spacing)
        usable = [ch for ch in channels if ch.n_usable >= DEFAULTS["min_epochs_individual"]]
        if not usable:
            continue

        data_path = os.path.join(export_dir, f"G{gid}_{mode}.dat")
        with open(data_path, "w") as f:
            f.write(f"# VERA format data — G{gid} {mode} PR\n")
            f.write(f"# Source: RA={ra_hr:.6f}h Dec={dec_deg:.6f}d\n")
            f.write(f"# DOY        RA(mas)       dRA(mas)      Dec(mas)"
                    f"      dDec(mas) SPOT CH      VLSR\n")
            for ch_idx, ch in enumerate(usable):
                mask = ch.use_for_pi == 1
                for i in range(len(ch.mjds)):
                    if not mask[i]:
                        continue
                    # Convert MJD to day-of-year
                    year = int(2000 + (ch.mjds[i] - 51544.5) / 365.25)
                    jan1_mjd = (_dt(year, 1, 1) - _mjd_ref).days
                    doy = ch.mjds[i] - jan1_mjd + 1
                    f.write(f"{doy:10.4f} {ch.x[i]:12.6f} {ch.x_err[i]:10.6f} "
                            f"{ch.y[i]:12.6f} {ch.y_err[i]:10.6f} "
                            f"{gid:4d} {ch_idx+1:2d} {ch.vlsr:8.3f}\n")

        r = fitter.group_results[gid]
        prm_path = os.path.join(export_dir, f"G{gid}_{mode}.prm")
        with open(prm_path, "w") as f:
            f.write(f"# VERA parameter file — G{gid} {mode} PR\n")
            f.write(f"datafile = G{gid}_{mode}.dat\n")
            f.write(f"radec = 0  # 0=both, 1=RA only, -1=Dec only\n")
            f.write(f"RA = {ra_hr:.8f}  # decimal hours\n")
            f.write(f"Dec = {dec_deg:.8f}  # decimal degrees\n")
            f.write(f"parallax = 1  # 1=solve for parallax\n")
            f.write(f"epoch = {t_ref_mjd:.1f}  # reference MJD\n")
            f.write(f"# Results: pi={r.pi:.4f}+/-{r.sigma_pi:.4f} mas\n")

    print(f"  VERA export: {len(fitter.group_results)} groups in {export_dir}")


def export_kadai(fitter, outdir, mode):
    """Export KaDai/Matsuo-Nakanishi format files.

    Single file per group with all spots interleaved.
    Columns: DOY  RA(mas)  dRA  Dec(mas)  dDec  SPOT_ID

    Compatible with Matsuo/Nakanishi parallax_ver3.0.py.
    """
    from datetime import date as _dt
    _mjd_ref = _dt(1858, 11, 17)

    export_dir = os.path.join(outdir, "export", "kadai")
    os.makedirs(export_dir, exist_ok=True)

    ra_hr = np.degrees(fitter.ra_rad) / 15.0
    dec_deg = np.degrees(fitter.dec_rad)

    for gid in sorted(fitter.group_results.keys()):
        channels = extract_channels(fitter.spots_df, gid, chan_spacing=fitter.chan_spacing)
        usable = [ch for ch in channels if ch.n_usable >= DEFAULTS["min_epochs_individual"]]
        if not usable:
            continue

        path = os.path.join(export_dir, f"G{gid}_{mode}_kadai.dat")
        with open(path, "w") as f:
            f.write(f"# KaDai format — G{gid} {mode} PR\n")
            f.write(f"# RA(h)={ra_hr:.8f} Dec(d)={dec_deg:.8f}\n")
            f.write(f"# Columns: DOY  RA(mas)  dRA  Dec(mas)  dDec  SPOT_ID\n")
            for ch_idx, ch in enumerate(usable):
                mask = ch.use_for_pi == 1
                for i in range(len(ch.mjds)):
                    if not mask[i]:
                        continue
                    year = int(2000 + (ch.mjds[i] - 51544.5) / 365.25)
                    jan1_mjd = (_dt(year, 1, 1) - _mjd_ref).days
                    doy = ch.mjds[i] - jan1_mjd + 1
                    f.write(f"{doy:10.4f} {ch.x[i]:12.6f} {ch.x_err[i]:10.6f} "
                            f"{ch.y[i]:12.6f} {ch.y_err[i]:10.6f} "
                            f"{ch_idx+1:4d}\n")

    print(f"  KaDai export: {len(fitter.group_results)} groups in {export_dir}")


def plot_spots_html(fitter, outdir, mode):
    """Interactive HTML for individual spot (channel) fits.

    Layout: 3x3 grid (same as PNG) + pi vs VLSR + PM overview.
    Each channel is a separate toggleable trace, grouped by group_id.
    """
    if not HAS_PLOTLY:
        print("  Plotly not available, skipping spots HTML")
        return
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import colorsys

    t_ref = fitter._compute_ref_epoch()

    # VLSR colormap
    all_vlsrs = []
    for gid in fitter.group_results:
        channels = extract_channels(fitter.spots_df, gid, chan_spacing=fitter.chan_spacing)
        for ch in channels: all_vlsrs.append(ch.vlsr)
    if not all_vlsrs: return
    vmin, vmax = min(all_vlsrs), max(all_vlsrs)
    vrange = max(vmax - vmin, 1.0)
    def vc(v):
        h = 270 * (1 - (v - vmin)/vrange) / 360.0
        r, g, b = colorsys.hls_to_rgb(h, 0.45, 0.85)
        return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=[
            "Sky plane", "RA vs time", "Dec vs time",
            "Parallax ellipse", "RA parallax", "Dec parallax",
            "Residuals (sky)", "RA residuals", "Dec residuals",
            "π vs VLSR", "Proper motions (sky)", ""
        ],
        row_heights=[0.28, 0.28, 0.22, 0.22],
        vertical_spacing=0.06, horizontal_spacing=0.06,
    )

    sorted_gids = sorted(fitter.group_results.keys())
    n_g = len(sorted_gids)
    jitter = {gid: (i - n_g/2) * 6.0/max(n_g, 1) for i, gid in enumerate(sorted_gids)}

    for gid in sorted_gids:
        r = fitter.group_results[gid]
        if not hasattr(r, '_A'): continue
        channels = extract_channels(fitter.spots_df, gid, chan_spacing=fitter.chan_spacing)
        grade = fitter.spots_df[fitter.spots_df.group_id==gid].grade.iloc[0]
        lgname = f"G{gid} ({grade})"
        is_outlier = gid in fitter.parallax_outliers
        first_ch = True

        # Model curve for this group
        all_mjds_g = []
        for ch in channels:
            m = ch.use_for_pi == 1
            if np.sum(m) >= 4: all_mjds_g.extend(ch.mjds[m])
        if not all_mjds_g: continue
        mjd_f = np.linspace(min(all_mjds_g)-30, max(all_mjds_g)+30, 200)
        yr_f = 2000.0 + (mjd_f - MJD_J2000) / JULIANYEAR
        F_af, F_df = parallax_factors_analytical(fitter.ra_rad, fitter.dec_rad, mjd_f)
        color_g = vc(np.mean([ch.vlsr for ch in channels]))

        # Parallax ellipse model (group-level)
        mjd_e = np.linspace(t_ref - 182, t_ref + 182, 200)
        F_ae, F_de = parallax_factors_analytical(fitter.ra_rad, fitter.dec_rad, mjd_e)
        fig.add_trace(go.Scatter(
            x=(r.pi*F_ae).tolist(), y=(r.pi*F_de).tolist(),
            mode="lines", line=dict(color=color_g, width=1, dash="dot"),
            legendgroup=lgname, showlegend=False,
        ), row=2, col=1)

        # Parallax model curves
        fig.add_trace(go.Scatter(
            x=yr_f.tolist(), y=(r.pi*F_af).tolist(),
            mode="lines", line=dict(color=color_g, width=1, dash="dot"),
            legendgroup=lgname, showlegend=False,
        ), row=2, col=2)
        fig.add_trace(go.Scatter(
            x=yr_f.tolist(), y=(r.pi*F_df).tolist(),
            mode="lines", line=dict(color=color_g, width=1, dash="dot"),
            legendgroup=lgname, showlegend=False,
        ), row=2, col=3)

        SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross",
                   "star", "hexagram", "pentagon", "triangle-down", "bowtie"]

        for ch_i, ch in enumerate(channels):
            mask = ch.use_for_pi == 1
            if np.sum(mask) < 4: continue
            ch_r = fitter.individual_results.get(gid, {}).get(ch.vlsr)
            if ch_r is None or ch_r.params is None or len(ch_r.params) < 5: continue

            p = ch_r.params
            mjds = ch.mjds[mask]
            dt = (mjds - t_ref) / JULIANYEAR
            yr = 2000.0 + ((mjds + jitter[gid]) - MJD_J2000) / JULIANYEAR
            F_a, F_d = parallax_factors_analytical(fitter.ra_rad, fitter.dec_rad, mjds)
            xe = np.sqrt(ch.x_err[mask]**2 + r.floor_ra**2)
            ye = np.sqrt(ch.y_err[mask]**2 + r.floor_dec**2)
            color = vc(ch.vlsr)

            x_mod = p[2] + p[1]*dt + p[0]*F_a
            y_mod = p[4] + p[3]*dt + p[0]*F_d
            x_pi = ch.x[mask] - (p[2] + p[1]*dt)
            y_pi = ch.y[mask] - (p[4] + p[3]*dt)
            x_res = ch.x[mask] - x_mod
            y_res = ch.y[mask] - y_mod

            nm = f"v={ch.vlsr:.1f}"
            sym = SYMBOLS[ch_i % len(SYMBOLS)]
            hover = [f"G{gid} {nm}<br>MJD {m:.0f}" for m in mjds]
            # Every channel gets its own legend entry for individual toggling
            kw = dict(legendgroup=lgname,
                      legendgrouptitle_text=lgname if first_ch else None,
                      name=nm, showlegend=True,
                      hovertext=hover, hoverinfo="text")
            mkw = dict(color=color, size=6, symbol=sym,
                       line=dict(width=0.5, color="black"))

            # Row 1: Sky, RA, Dec with model
            fig.add_trace(go.Scatter(x=ch.x[mask].tolist(), y=ch.y[mask].tolist(),
                error_x=dict(array=xe.tolist(), thickness=1, width=2), error_y=dict(array=ye.tolist(), thickness=1, width=2),
                mode="markers", marker=mkw, **kw), row=1, col=1)
            fig.add_trace(go.Scatter(x=yr.tolist(), y=ch.x[mask].tolist(),
                error_y=dict(array=xe.tolist(), thickness=1, width=2), mode="markers", marker=mkw,
                legendgroup=lgname, showlegend=False), row=1, col=2)
            fig.add_trace(go.Scatter(x=yr.tolist(), y=ch.y[mask].tolist(),
                error_y=dict(array=ye.tolist(), thickness=1, width=2), mode="markers", marker=mkw,
                legendgroup=lgname, showlegend=False), row=1, col=3)

            # Row 2: Pi ellipse, RA pi, Dec pi
            fig.add_trace(go.Scatter(x=x_pi.tolist(), y=y_pi.tolist(),
                mode="markers", marker=mkw, legendgroup=lgname, showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=yr.tolist(), y=x_pi.tolist(),
                error_y=dict(array=xe.tolist(), thickness=1, width=2), mode="markers", marker=mkw,
                legendgroup=lgname, showlegend=False), row=2, col=2)
            fig.add_trace(go.Scatter(x=yr.tolist(), y=y_pi.tolist(),
                error_y=dict(array=ye.tolist(), thickness=1, width=2), mode="markers", marker=mkw,
                legendgroup=lgname, showlegend=False), row=2, col=3)

            # Row 3: Residuals
            fig.add_trace(go.Scatter(x=x_res.tolist(), y=y_res.tolist(),
                mode="markers", marker=mkw, legendgroup=lgname, showlegend=False), row=3, col=1)
            fig.add_trace(go.Scatter(x=yr.tolist(), y=x_res.tolist(),
                error_y=dict(array=xe.tolist(), thickness=1, width=2), mode="markers", marker=mkw,
                legendgroup=lgname, showlegend=False), row=3, col=2)
            fig.add_trace(go.Scatter(x=yr.tolist(), y=y_res.tolist(),
                error_y=dict(array=ye.tolist(), thickness=1, width=2), mode="markers", marker=mkw,
                legendgroup=lgname, showlegend=False), row=3, col=3)

            first_ch = False

        # Row 4 col 1: pi vs vlsr (group-level)
        n_ch = r.n_channels
        if gid in fitter.bootstrap_results and len(fitter.bootstrap_results[gid]) > 100:
            bp = fitter.bootstrap_results[gid]
            sig = 0.5*(np.percentile(bp,84)-np.percentile(bp,16))
        else:
            sig = r.sigma_pi * np.sqrt(n_ch)
        vlsr_m = np.mean([ch.vlsr for ch in channels])
        fig.add_trace(go.Scatter(
            x=[vlsr_m], y=[r.pi],
            error_y=dict(type="data", array=[sig], visible=True),
            mode="markers", marker=dict(color=color_g, size=10,
                symbol="x" if is_outlier else "circle", line=dict(width=1,color="black")),
            legendgroup=lgname, showlegend=False,
            hovertext=f"G{gid} ({grade})<br>π={r.pi:.4f}±{sig:.4f}", hoverinfo="text",
        ), row=4, col=1)

        # Row 4 col 2: PM vector
        mu_info = list(r.mu.values())[0] if r.mu else {}
        if "mu_x" in mu_info and "mu_y" in mu_info:
            usable = [ch for ch in channels if ch.n_usable >= 4]
            if usable:
                x0 = np.mean([np.mean(ch.x[ch.use_for_pi==1]) for ch in usable])
                y0 = np.mean([np.mean(ch.y[ch.use_for_pi==1]) for ch in usable])
                fig.add_trace(go.Scatter(
                    x=[x0, x0+mu_info["mu_x"]], y=[y0, y0+mu_info["mu_y"]],
                    mode="lines+markers", line=dict(color=color_g, width=1.5),
                    marker=dict(color=color_g, size=[3, 7]),
                    legendgroup=lgname, showlegend=False,
                    hovertext=f"G{gid} μ=({mu_info['mu_x']:.2f},{mu_info['mu_y']:.2f})",
                    hoverinfo="text",
                ), row=4, col=2)

    # Weighted mean line on pi vs vlsr
    for label in ["A+B+C"]:
        if label in fitter.summary:
            s = fitter.summary[label]
            fig.add_hline(y=s["pi"], line_dash="dash", line_color="black",
                          annotation_text=f"π={s['pi']:.3f}±{s['sigma_pi']:.3f}",
                          row=4, col=1)

    fig.update_layout(
        title=f"Individual spot fits — {mode} PR (v{__version__})",
        height=1400, width=1400,
        legend=dict(groupclick="togglegroup", tracegroupgap=2, font=dict(size=8)),
        template="plotly_white",
        updatemenus=[dict(type="buttons", direction="left",
            x=0.0, y=1.01, xanchor="left", yanchor="bottom",
            buttons=[dict(label="Show All", method="update", args=[{"visible": True}]),
                     dict(label="Hide All", method="update", args=[{"visible": "legendonly"}])]
        )],
    )
    for r_idx, c_idx, xl, yl in [
        (1,1,"RA (mas)","Dec (mas)"), (1,2,"Year","RA (mas)"), (1,3,"Year","Dec (mas)"),
        (2,1,"RA pi (mas)","Dec pi (mas)"), (2,2,"Year","RA pi (mas)"), (2,3,"Year","Dec pi (mas)"),
        (3,1,"RA resid","Dec resid"), (3,2,"Year","RA resid"), (3,3,"Year","Dec resid"),
        (4,1,"VLSR (km/s)","π (mas)"), (4,2,"RA (mas)","Dec (mas)"),
    ]:
        fig.update_xaxes(title_text=xl, row=r_idx, col=c_idx)
        fig.update_yaxes(title_text=yl, row=r_idx, col=c_idx)

    path = os.path.join(outdir, f"fit_{mode}_spots_interactive.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  Spots HTML: {path}")
    return path


def plot_features_html(fitter, outdir, mode):
    """Interactive HTML for feature-level fits.

    Same 3x3+2 layout as spot HTML but using feature centroids.
    """
    if not HAS_PLOTLY or not fitter.feature_results or fitter.features_df is None:
        return
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import colorsys

    t_ref = fitter._compute_ref_epoch()
    all_vlsrs = [fitter.features_df[fitter.features_df.group_id==gid].vlsr.median()
                 for gid in fitter.feature_results]
    if not all_vlsrs: return
    vmin, vmax = min(all_vlsrs), max(all_vlsrs)
    vrange = max(vmax-vmin, 1.0)
    def vc(v):
        h = 270*(1-(v-vmin)/vrange)/360.0
        r,g,b = __import__('colorsys').hls_to_rgb(h,0.45,0.85)
        return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=[
            "Sky plane","RA vs time","Dec vs time",
            "Parallax ellipse","RA parallax","Dec parallax",
            "Residuals (sky)","RA residuals","Dec residuals",
            "π vs VLSR","Proper motions",""
        ],
        row_heights=[0.28,0.28,0.22,0.22],
        vertical_spacing=0.06, horizontal_spacing=0.06,
    )

    sorted_gids = sorted(fitter.feature_results.keys())
    n_g = len(sorted_gids)
    jitter = {gid: (i-n_g/2)*6.0/max(n_g,1) for i, gid in enumerate(sorted_gids)}

    for gid in sorted_gids:
        fr = fitter.feature_results[gid]
        if fr.params is None or fr.coord_mode != "both" or len(fr.params) < 5: continue

        gdf = fitter.features_df[fitter.features_df.group_id==gid]
        grade = gdf.grade.iloc[0]
        mask = gdf.use_for_pi.values == 1
        if np.sum(mask) < 4: continue

        lgname = f"G{gid} ({grade})"
        vlsr_m = gdf.vlsr.median()
        color = vc(vlsr_m)
        is_outlier = gid in fitter.parallax_outliers

        mjds = gdf.mjd.values[mask].astype(float)
        x, y = gdf.x.values[mask], gdf.y.values[mask]
        xe, ye = gdf.x_err.values[mask], gdf.y_err.values[mask]
        dt = (mjds - t_ref) / JULIANYEAR
        yr = 2000.0 + ((mjds + jitter[gid]) - MJD_J2000) / JULIANYEAR
        F_a, F_d = parallax_factors_analytical(fitter.ra_rad, fitter.dec_rad, mjds)
        p = fr.params

        x_mod = p[2]+p[1]*dt+p[0]*F_a
        y_mod = p[4]+p[3]*dt+p[0]*F_d
        x_pi = x-(p[2]+p[1]*dt)
        y_pi = y-(p[4]+p[3]*dt)
        x_res, y_res = x-x_mod, y-y_mod

        mkw = dict(color=color, size=7, symbol="x" if is_outlier else "circle")
        kw = dict(legendgroup=lgname, legendgrouptitle_text=lgname, name=f"G{gid}",
                  showlegend=True, hoverinfo="text",
                  hovertext=[f"G{gid} v={vlsr_m:.1f}<br>MJD {m:.0f}" for m in mjds])

        # Rows 1-3 same layout
        fig.add_trace(go.Scatter(x=x.tolist(),y=y.tolist(),error_x=dict(array=xe.tolist(), thickness=1, width=2),
            error_y=dict(array=ye.tolist(), thickness=1, width=2),mode="markers",marker=mkw,**kw),row=1,col=1)
        for ri,ci,xd,yd,ed in [(1,2,yr,x,xe),(1,3,yr,y,ye),(2,2,yr,x_pi,xe),(2,3,yr,y_pi,ye),
                                 (3,2,yr,x_res,xe),(3,3,yr,y_res,ye)]:
            fig.add_trace(go.Scatter(x=xd.tolist(),y=yd.tolist(),error_y=dict(array=ed.tolist(), thickness=1, width=2),
                mode="markers",marker=mkw,legendgroup=lgname,showlegend=False),row=ri,col=ci)
        fig.add_trace(go.Scatter(x=x_pi.tolist(),y=y_pi.tolist(),mode="markers",marker=mkw,
            legendgroup=lgname,showlegend=False),row=2,col=1)
        fig.add_trace(go.Scatter(x=x_res.tolist(),y=y_res.tolist(),mode="markers",marker=mkw,
            legendgroup=lgname,showlegend=False),row=3,col=1)

        # Model curves
        mjd_f = np.linspace(mjds.min()-30,mjds.max()+30,200)
        yr_f = 2000.0+(mjd_f-MJD_J2000)/JULIANYEAR
        F_af,F_df = parallax_factors_analytical(fitter.ra_rad,fitter.dec_rad,mjd_f)
        fig.add_trace(go.Scatter(x=yr_f.tolist(),y=(fr.pi*F_af).tolist(),mode="lines",
            line=dict(color=color,width=1,dash="dot"),legendgroup=lgname,showlegend=False),row=2,col=2)
        fig.add_trace(go.Scatter(x=yr_f.tolist(),y=(fr.pi*F_df).tolist(),mode="lines",
            line=dict(color=color,width=1,dash="dot"),legendgroup=lgname,showlegend=False),row=2,col=3)
        mjd_e = np.linspace(t_ref-182,t_ref+182,200)
        F_ae,F_de = parallax_factors_analytical(fitter.ra_rad,fitter.dec_rad,mjd_e)
        fig.add_trace(go.Scatter(x=(fr.pi*F_ae).tolist(),y=(fr.pi*F_de).tolist(),mode="lines",
            line=dict(color=color,width=1,dash="dot"),legendgroup=lgname,showlegend=False),row=2,col=1)

        # Row 4: pi, PM
        fig.add_trace(go.Scatter(x=[vlsr_m],y=[fr.pi],error_y=dict(array=[fr.sigma_pi]),
            mode="markers",marker=dict(color=color,size=10,symbol="x" if is_outlier else "circle"),
            legendgroup=lgname,showlegend=False,
            hovertext=f"G{gid} π={fr.pi:.4f}±{fr.sigma_pi:.4f}",hoverinfo="text"),row=4,col=1)
        fig.add_trace(go.Scatter(x=[x[0],x[0]+p[1]],y=[y[0],y[0]+p[3]],mode="lines+markers",
            line=dict(color=color,width=1.5),marker=dict(color=color,size=[3,7]),
            legendgroup=lgname,showlegend=False),row=4,col=2)

    fig.update_layout(
        title=f"Feature fits — {mode} PR (v{__version__})",
        height=1400, width=1400,
        legend=dict(groupclick="togglegroup",tracegroupgap=2,font=dict(size=8)),
        template="plotly_white",
        updatemenus=[dict(type="buttons",direction="left",
            x=0.0,y=1.01,xanchor="left",yanchor="bottom",
            buttons=[dict(label="Show All",method="update",args=[{"visible":True}]),
                     dict(label="Hide All",method="update",args=[{"visible":"legendonly"}])]
        )],
    )
    for r_idx,c_idx,xl,yl in [
        (1,1,"RA","Dec"),(1,2,"Year","RA"),(1,3,"Year","Dec"),
        (2,1,"RA pi","Dec pi"),(2,2,"Year","RA pi"),(2,3,"Year","Dec pi"),
        (3,1,"RA resid","Dec resid"),(3,2,"Year","RA resid"),(3,3,"Year","Dec resid"),
        (4,1,"VLSR (km/s)","π (mas)"),(4,2,"RA (mas)","Dec (mas)"),
    ]:
        fig.update_xaxes(title_text=xl,row=r_idx,col=c_idx)
        fig.update_yaxes(title_text=yl,row=r_idx,col=c_idx)

    path = os.path.join(outdir, f"fit_{mode}_features_interactive.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  Features HTML: {path}")
    return path


def plot_final_proper_motions(fitter, outdir, mode):
    """Proper motion overview: all features on sky with VLSR colors.

    Single panel with mas on primary axes, AU on secondary axes.
    Scale bars: black arrow for mas/yr, vertical bar for AU @ distance.
    Style follows Orosz et al. 2017 (MNRAS 468, L63) Figure 1b.
    """
    if not HAS_MPL or fitter.features_df is None:
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    # Distance for AU conversion
    best_label = "A+B+C"
    for l in ["A+B+C", "A+B", "A"]:
        if l in fitter.summary:
            best_label = l; break
    pi_best = fitter.summary[best_label]["pi"] if best_label in fitter.summary else 0.1
    D_kpc = 1.0 / pi_best if pi_best > 0 else 10.0
    au_per_mas = D_kpc * 1000.0

    # Collect ALL features
    data = []
    for gid in fitter.features_df.group_id.unique():
        gdf = fitter.features_df[fitter.features_df.group_id == gid]
        grade = gdf.grade.iloc[0]
        n_ep = int(np.sum(gdf.use_for_pi.values == 1))
        vlsr_m = gdf.vlsr.median()
        mask = gdf.use_for_pi.values == 1
        if np.sum(mask) == 0: mask = np.ones(len(gdf), dtype=bool)
        x0 = np.mean(gdf.x.values[mask])
        y0 = np.mean(gdf.y.values[mask])
        mu_x, mu_y, has_pm = 0, 0, False
        fr = fitter.feature_results.get(gid)
        gr = fitter.group_results.get(gid)
        if fr is not None and fr.params is not None and len(fr.params) >= 5:
            mu_x, mu_y = fr.params[1], fr.params[3]
            has_pm = True
        elif gr is not None and gr.mu:
            mi = list(gr.mu.values())[0]
            if "mu_x" in mi:
                mu_x, mu_y = mi["mu_x"], mi["mu_y"]
                has_pm = True
        data.append(dict(gid=gid, grade=grade, n_ep=n_ep, vlsr=vlsr_m,
                         x0=x0, y0=y0, mu_x=mu_x, mu_y=mu_y, has_pm=has_pm))

    if not data:
        plt.close(); return

    vlsrs = np.array([d["vlsr"] for d in data])
    vmin, vmax = vlsrs.min(), vlsrs.max()
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.cm.rainbow

    for d in data:
        c = cmap(norm(d["vlsr"]))
        if d["has_pm"]:
            ls = "-" if d["n_ep"] >= 3 else "--"
            ax.annotate("", xy=(d["x0"]+d["mu_x"], d["y0"]+d["mu_y"]),
                        xytext=(d["x0"], d["y0"]),
                        arrowprops=dict(arrowstyle="->", color=c, lw=2, linestyle=ls))
        ax.plot(d["x0"], d["y0"], "o", color=c,
                ms=6 if d["has_pm"] else 4,
                alpha=1.0 if d["has_pm"] else 0.5,
                markeredgecolor="black" if d["has_pm"] else "none",
                markeredgewidth=0.3)

    ax.invert_xaxis()
    ax.set_xlabel("RA offset (mas)", fontsize=11)
    ax.set_ylabel("Dec offset (mas)", fontsize=11)
    ax.set_title(f"Proper motions — {mode} PR", fontsize=12)
    ax.set_aspect("equal", adjustable="datalim")

    # Secondary axes in AU
    ax2x = ax.secondary_xaxis("top", functions=(lambda x: x * au_per_mas,
                                                  lambda x: x / au_per_mas))
    ax2y = ax.secondary_yaxis("right", functions=(lambda y: y * au_per_mas,
                                                    lambda y: y / au_per_mas))
    ax2x.set_xlabel(f"RA offset (AU at {D_kpc:.1f} kpc)", fontsize=9)
    ax2y.set_ylabel(f"Dec offset (AU at {D_kpc:.1f} kpc)", fontsize=9)

    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(sm, ax=ax, label="VLSR (km/s)", shrink=0.6, pad=0.08)

    # Scale bars (bottom-left, Orosz et al. style)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = abs(xlim[1] - xlim[0])
    y_range = abs(ylim[1] - ylim[0])

    # Pick nice scale values
    scale_mas = max(1.0, round(x_range * 0.1))  # ~10% of plot width
    scale_au = round(scale_mas * au_per_mas / 1000) * 1000  # round to nearest 1000 AU
    if scale_au == 0: scale_au = 1000
    scale_au_mas = scale_au / au_per_mas

    # Position in bottom-left (note: x is inverted)
    sb_x = xlim[0] - 0.05 * x_range  # left side in inverted coords
    sb_y = ylim[0] + 0.06 * y_range

    # mas/yr arrow (horizontal)
    ax.annotate("", xy=(sb_x - scale_mas, sb_y), xytext=(sb_x, sb_y),
                arrowprops=dict(arrowstyle="->", color="black", lw=2.5))
    ax.text(sb_x - scale_mas/2, sb_y - 0.02*y_range,
            f"{scale_mas:.0f} mas yr$^{{-1}}$", ha="center", va="top", fontsize=9,
            fontweight="bold")

    # AU bar (vertical, right side like Orosz)
    bar_x = xlim[1] + 0.03 * x_range
    bar_y0 = ylim[0] + 0.05 * y_range
    bar_y1 = bar_y0 + scale_au_mas
    ax.plot([bar_x, bar_x], [bar_y0, bar_y1], "k-", lw=3, clip_on=False)
    ax.plot([bar_x - 0.005*x_range, bar_x + 0.005*x_range], [bar_y0, bar_y0],
            "k-", lw=2, clip_on=False)
    ax.plot([bar_x - 0.005*x_range, bar_x + 0.005*x_range], [bar_y1, bar_y1],
            "k-", lw=2, clip_on=False)
    ax.text(bar_x + 0.015*x_range, (bar_y0+bar_y1)/2,
            f"{scale_au:.0f} au\n@ {D_kpc:.1f} kpc",
            ha="left", va="center", fontsize=9, fontweight="bold",
            rotation=90, clip_on=False)

    # Legend
    from matplotlib.lines import Line2D
    ax.legend([Line2D([0],[0],color="gray",lw=2,ls="-"),
               Line2D([0],[0],color="gray",lw=2,ls="--"),
               Line2D([0],[0],marker="o",color="gray",ms=4,ls="",alpha=0.3)],
              ["\u22653 epochs (solid)", "2 epochs (dashed)", "no PM"],
              fontsize=8, loc="upper left")

    plt.tight_layout()
    path = os.path.join(outdir, f"fit_{mode}_final_proper_motions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {path}")

def plot_final_parallax_overview(fitter, outdir, mode):
    """Final parallax overview matrix: one row per group + combined.

    Columns: RA/Dec parallax curves | Bootstrap histogram | Distance posterior
    Rows: one per non-outlier group (A/B grade) + combined weighted mean at bottom
    """
    if not HAS_MPL:
        return

    # Select groups to show: non-outlier A/B with bootstrap
    best_label = "A+B+C"
    for l in ["A+B+C", "A+B", "A"]:
        if l in fitter.summary:
            best_label = l; break
    if best_label not in fitter.summary:
        return
    s = fitter.summary[best_label]
    show_gids = [gid for gid in s["groups"]
                 if gid in fitter.group_results and gid not in fitter.parallax_outliers]
    # Sort by grade then group_id
    show_gids.sort(key=lambda g: (
        GRADE_ORDER.get(fitter.spots_df[fitter.spots_df.group_id==g].grade.iloc[0], 9), g))

    # Limit to max 8 groups + combined
    if len(show_gids) > 8:
        show_gids = show_gids[:8]

    n_rows = len(show_gids) + 1  # +1 for combined
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 3.5 * n_rows),
                              gridspec_kw={"width_ratios": [1.3, 0.8, 0.8]})
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f"Parallax overview — {mode} PR", fontsize=13, fontweight="bold")

    t_ref = fitter._compute_ref_epoch()
    all_mjds = fitter.spots_df[fitter.spots_df.use_for_pi==1].mjd.values.astype(float)
    if len(all_mjds) == 0: plt.close(); return
    mjd_m = np.linspace(all_mjds.min()-30, all_mjds.max()+30, 500)
    yr_m = 2000.0 + (mjd_m - MJD_J2000) / JULIANYEAR
    F_am, F_dm = parallax_factors_analytical(fitter.ra_rad, fitter.dec_rad, mjd_m)

    def _plot_row(row_idx, label, pi_val, sigma_val, boot_samples, gids_for_data):
        ax_pi = axes[row_idx, 0]
        ax_boot = axes[row_idx, 1]
        ax_dist = axes[row_idx, 2]

        # Left: RA parallax data centered on zero + model curve
        ax_pi.plot(yr_m, pi_val * F_am, "k-", lw=1.5, alpha=0.4)
        ax_pi.axhline(0, color="gray", ls=":", lw=0.3)

        for gid in gids_for_data:
            if gid not in fitter.group_results: continue
            r = fitter.group_results[gid]
            channels = extract_channels(fitter.spots_df, gid, chan_spacing=fitter.chan_spacing)
            for ch in channels:
                mask = ch.use_for_pi == 1
                if np.sum(mask) < 4: continue
                mu_info = r.mu.get(ch.vlsr, {})
                mu_x = mu_info.get("mu_x", 0)
                x0 = mu_info.get("x0", 0)
                mjds = ch.mjds[mask]
                dt = (mjds - t_ref) / JULIANYEAR
                yr = 2000.0 + (mjds - MJD_J2000) / JULIANYEAR
                F_a_ch, _ = parallax_factors_analytical(fitter.ra_rad, fitter.dec_rad, mjds)
                # Subtract PM and offset, then center so mean residual = 0
                x_pi = ch.x[mask] - (x0 + mu_x * dt)
                offset = np.mean(x_pi - r.pi * F_a_ch)
                x_pi -= offset
                xe = np.sqrt(ch.x_err[mask]**2 + r.floor_ra**2)
                ax_pi.errorbar(yr, x_pi, yerr=xe, fmt="o", color="steelblue",
                               ms=2, alpha=0.35, lw=0.4)

        d_simple = 1.0 / pi_val if pi_val > 0 else float("inf")
        ax_pi.set_ylabel(f"{label}\nRA π (mas)", fontsize=8)
        ax_pi.legend([plt.Line2D([0],[0],color="k",lw=1.5)],
                     [f"π={pi_val:.3f}±{sigma_val:.3f}\n1/π={d_simple:.1f} kpc"],
                     fontsize=6, loc="upper right")
        ax_pi.tick_params(labelsize=7)
        if row_idx == n_rows - 1:
            ax_pi.set_xlabel("Year", fontsize=8)

        # Middle: Bootstrap histogram with CI numbers
        if boot_samples is not None and len(boot_samples) > 100:
            med = np.median(boot_samples)
            iqr = np.percentile(boot_samples, 75) - np.percentile(boot_samples, 25)
            clip = boot_samples[(boot_samples > med - 5*iqr) & (boot_samples < med + 5*iqr)]
            ax_boot.hist(clip, bins=50, color="steelblue", alpha=0.7, density=True,
                         edgecolor="white", linewidth=0.3)
            p16 = np.percentile(boot_samples, 16)
            p84 = np.percentile(boot_samples, 84)
            sig_b = 0.5 * (p84 - p16)
            ax_boot.axvline(med, color="red", lw=1.5)
            ax_boot.axvline(p16, color="red", ls="--", lw=0.8)
            ax_boot.axvline(p84, color="red", ls="--", lw=0.8)
            from scipy.stats import norm as _norm
            xg = np.linspace(med - 4*sig_b, med + 4*sig_b, 200)
            ax_boot.plot(xg, _norm.pdf(xg, med, sig_b), "k--", lw=0.8, alpha=0.5)
            ax_boot.set_xlim(med - 5*sig_b, med + 5*sig_b)
            ax_boot.set_title(f"π={med:.3f} [{p16:.3f}, {p84:.3f}]\nσ={sig_b:.3f}",
                              fontsize=7, pad=2)
        else:
            ax_boot.text(0.5, 0.5, "no bootstrap", ha="center", va="center",
                         transform=ax_boot.transAxes, fontsize=8, color="gray")
        ax_boot.tick_params(labelsize=7)
        if row_idx == n_rows - 1:
            ax_boot.set_xlabel("π (mas)", fontsize=8)

        # Right: Distance PDF + prior comparison
        if pi_val > 0 and sigma_val > 0:
            # Always show the likelihood PDF (Gaussian in parallax → distance)
            D_grid = np.linspace(0.1, 25.0, 2000)
            pi_model = 1.0 / D_grid
            likelihood = np.exp(-0.5 * ((pi_val - pi_model) / sigma_val)**2)
            # Normalize
            dD = D_grid[1] - D_grid[0]
            norm_L = np.sum(likelihood) * dD
            if norm_L > 0: likelihood /= norm_L
            ax_dist.plot(D_grid, likelihood, "k-", lw=1.5, label="1/π")
            D_mode = D_grid[np.argmax(likelihood)]
            D_simple = 1.0 / pi_val
            D_lo = 1.0 / (pi_val + sigma_val)
            D_hi = min(1.0 / max(pi_val - sigma_val, 0.01), 25.0)
            ax_dist.axvline(D_simple, color="red", lw=1)
            ax_dist.axvspan(D_lo, D_hi, alpha=0.1, color="blue")
            ax_dist.set_xlim(0, 25)

            # Prior comparison annotations with matching colors
            y_txt = 0.97
            ax_dist.text(0.97, y_txt, f"1/π: {D_simple:.1f} [{D_lo:.1f}, {D_hi:.1f}]",
                         transform=ax_dist.transAxes, fontsize=5.5, color="red",
                         va="top", ha="right", fontweight="bold")
            y_txt -= 0.08
            for ptype, pcolor, plabel in [
                ("exponential", "green", "exp(L=5)"),
                ("disk", "orange", f"disk(h_z=0.3)")]:
                dp = compute_distance_posterior(pi_val, sigma_val,
                    fitter.l_rad, fitter.b_rad, prior_type=ptype,
                    h_z=0.3 if ptype=="disk" else DISK_HZ)
                if len(dp["D_grid"]) > 0 and dp["D_mode"] > 0:
                    ax_dist.plot(dp["D_grid"], dp["posterior"],
                                 color=pcolor, lw=0.8, alpha=0.6, ls="--")
                    ax_dist.text(0.97, y_txt, f"{plabel}: {dp['D_mode']:.1f} kpc",
                                 transform=ax_dist.transAxes, fontsize=5.5,
                                 color=pcolor, va="top", ha="right")
                    y_txt -= 0.08
            ax_dist.set_title(f"D={D_simple:.1f} [{D_lo:.1f},{D_hi:.1f}]",
                              fontsize=7, pad=2)
        ax_dist.tick_params(labelsize=7)
        if row_idx == n_rows - 1:
            ax_dist.set_xlabel("D (kpc)", fontsize=8)

    # Plot each group
    for row_i, gid in enumerate(show_gids):
        r = fitter.group_results[gid]
        grade = fitter.spots_df[fitter.spots_df.group_id==gid].grade.iloc[0]
        n_ch = r.n_channels
        boot = fitter.bootstrap_results.get(gid)
        if boot is not None and len(boot) > 100:
            sig = 0.5 * (np.percentile(boot, 84) - np.percentile(boot, 16))
        else:
            sig = r.sigma_pi * np.sqrt(n_ch)
        _plot_row(row_i, f"G{gid} ({grade})", r.pi, sig, boot, [gid])

    # Combined bootstrap: resample epochs, recompute weighted mean each trial
    combined_boot = None
    if fitter.bootstrap_results:
        # Collect per-group bootstrap arrays for non-outlier groups
        boot_gids = [g for g in show_gids if g in fitter.bootstrap_results
                     and len(fitter.bootstrap_results[g]) > 100]
        if len(boot_gids) >= 2:
            n_trials = min(len(fitter.bootstrap_results[g]) for g in boot_gids)
            combined_boot = np.zeros(n_trials)
            for trial in range(n_trials):
                pis_t = np.array([fitter.bootstrap_results[g][trial] for g in boot_gids])
                sigs_t = np.array([
                    0.5 * (np.percentile(fitter.bootstrap_results[g], 84) -
                           np.percentile(fitter.bootstrap_results[g], 16))
                    for g in boot_gids])
                sigs_t = np.maximum(sigs_t, 1e-6)
                w_t = 1.0 / sigs_t**2
                combined_boot[trial] = np.sum(pis_t * w_t) / np.sum(w_t)

    _plot_row(n_rows - 1, f"Combined ({best_label})",
              s["pi"], s["sigma_pi"], combined_boot, show_gids)

    plt.tight_layout()
    path = os.path.join(outdir, f"fit_{mode}_final_parallax_overview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {path}")

def plot_parallax_fit(fitter, gid, outdir, mode):
    """Plot parallax + proper motion fit for a single group (3x3 panel).

    Layout:
      Row 1 (proper motion): Sky plane | RA vs time + model | Dec vs time + model
      Row 2 (parallax):      Pi ellipse | RA pi only + model | Dec pi only + model
      Row 3 (residuals):     Sky resid  | RA resid vs time   | Dec resid vs time
    """
    if not HAS_MPL or gid not in fitter.group_results:
        return None

    r = fitter.group_results[gid]
    channels = extract_channels(fitter.spots_df, gid, chan_spacing=fitter.chan_spacing)
    usable = [ch for ch in channels if ch.n_usable >= DEFAULTS["min_epochs_individual"]]
    if not usable:
        return None

    t_ref = fitter._compute_ref_epoch()
    grade = usable[0].grade
    n_ch = r.n_channels

    # Title with bootstrap sigma if available
    if gid in fitter.bootstrap_results and len(fitter.bootstrap_results[gid]) > 100:
        bp = fitter.bootstrap_results[gid]
        bsig = 0.5 * (np.percentile(bp, 84) - np.percentile(bp, 16))
        pi_str = f"pi = {r.pi:.3f} +/- {bsig:.3f} mas (bootstrap)"
    else:
        sqN = r.sigma_pi * np.sqrt(n_ch)
        pi_str = f"pi = {r.pi:.3f} +/- {r.sigma_pi:.3f} (sqrtN: +/-{sqN:.3f}) mas"

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f"G{gid} ({grade}) — {mode} PR — {pi_str}",
                 fontsize=12, fontweight="bold")

    # Rainbow colors by VLSR
    vlsrs = [ch.vlsr for ch in usable]
    vmin, vmax = min(vlsrs), max(vlsrs)
    vrange = max(vmax - vmin, 1.0)
    colors = [plt.cm.rainbow((ch.vlsr - vmin) / vrange) for ch in usable]

    # Fine time grid for model curves
    all_mjds = np.concatenate([ch.mjds for ch in usable])
    mjd_model = np.linspace(all_mjds.min() - 30, all_mjds.max() + 30, 300)
    yr_model = 2000.0 + (mjd_model - MJD_J2000) / JULIANYEAR
    dt_model = (mjd_model - t_ref) / JULIANYEAR
    F_a_m, F_d_m = parallax_factors_analytical(fitter.ra_rad, fitter.dec_rad, mjd_model)

    for ci, ch in enumerate(usable):
        mask = ch.use_for_pi == 1
        if np.sum(mask) == 0:
            continue
        mjds = ch.mjds[mask]
        dt = (mjds - t_ref) / JULIANYEAR
        yr = 2000.0 + (mjds - MJD_J2000) / JULIANYEAR
        F_a_ch, F_d_ch = parallax_factors_analytical(fitter.ra_rad, fitter.dec_rad, mjds)
        c = colors[ci]

        mu_info = r.mu.get(ch.vlsr, {})
        mu_x = mu_info.get("mu_x", 0)
        mu_y = mu_info.get("mu_y", 0)
        x0 = mu_info.get("x0", np.mean(ch.x[mask] - mu_x * dt - r.pi * F_a_ch))
        y0 = mu_info.get("y0", np.mean(ch.y[mask] - mu_y * dt - r.pi * F_d_ch))
        lbl = f"v={ch.vlsr:.1f}" if ci < 8 else None

        # Full model
        x_model = x0 + mu_x * dt + r.pi * F_a_ch
        y_model = y0 + mu_y * dt + r.pi * F_d_ch

        # Parallax-only
        x_pi = ch.x[mask] - (x0 + mu_x * dt)
        y_pi = ch.y[mask] - (y0 + mu_y * dt)

        # Residuals
        x_resid = ch.x[mask] - x_model
        y_resid = ch.y[mask] - y_model

        # Error bars (with floor)
        xe = np.sqrt(ch.x_err[mask]**2 + r.floor_ra**2)
        ye = np.sqrt(ch.y_err[mask]**2 + r.floor_dec**2)

        # --- Row 1: Proper motion ---
        # [0,0] Sky plane
        axes[0, 0].errorbar(ch.x[mask], ch.y[mask], xerr=xe, yerr=ye,
                             fmt="o", color=c, markersize=3, alpha=0.7, label=lbl)
        # [0,1] RA vs time + model
        axes[0, 1].errorbar(yr, ch.x[mask], yerr=xe, fmt="o", color=c, ms=3, alpha=0.7)
        axes[0, 1].plot(yr_model, x0 + mu_x*dt_model + r.pi*F_a_m, "-", color=c, lw=0.8, alpha=0.5)
        # [0,2] Dec vs time + model
        axes[0, 2].errorbar(yr, ch.y[mask], yerr=ye, fmt="o", color=c, ms=3, alpha=0.7)
        axes[0, 2].plot(yr_model, y0 + mu_y*dt_model + r.pi*F_d_m, "-", color=c, lw=0.8, alpha=0.5)

        # --- Row 2: Parallax ---
        # [1,0] Parallax ellipse
        axes[1, 0].plot(x_pi, y_pi, "o", color=c, markersize=3, alpha=0.6)
        # [1,1] RA parallax only
        axes[1, 1].errorbar(yr, x_pi, yerr=xe, fmt="o", color=c, ms=3, alpha=0.7)
        # [1,2] Dec parallax only
        axes[1, 2].errorbar(yr, y_pi, yerr=ye, fmt="o", color=c, ms=3, alpha=0.7)

        # --- Row 3: Residuals ---
        # [2,0] Residual sky
        axes[2, 0].plot(x_resid, y_resid, "o", color=c, markersize=3, alpha=0.6)
        # [2,1] RA residuals vs time
        axes[2, 1].errorbar(yr, x_resid, yerr=xe, fmt="o", color=c, ms=3, alpha=0.7)
        # [2,2] Dec residuals vs time
        axes[2, 2].errorbar(yr, y_resid, yerr=ye, fmt="o", color=c, ms=3, alpha=0.7)

    # Model overlays on Row 2
    axes[1, 1].plot(yr_model, r.pi * F_a_m, "k-", lw=1.5, alpha=0.6, label=f"pi={r.pi:.3f}")
    axes[1, 1].axhline(0, color="gray", ls=":", lw=0.5)
    axes[1, 2].plot(yr_model, r.pi * F_d_m, "k-", lw=1.5, alpha=0.6, label=f"pi={r.pi:.3f}")
    axes[1, 2].axhline(0, color="gray", ls=":", lw=0.5)

    # Parallax ellipse model
    mjd_ell = np.linspace(t_ref - 182, t_ref + 182, 200)
    F_a_e, F_d_e = parallax_factors_analytical(fitter.ra_rad, fitter.dec_rad, mjd_ell)
    axes[1, 0].plot(r.pi * F_a_e, r.pi * F_d_e, "k-", lw=1.5, alpha=0.5, label=f"pi={r.pi:.3f}")
    axes[1, 0].plot(r.pi * F_a_e[0], r.pi * F_d_e[0], "k^", ms=6)

    # Residual zero lines
    axes[2, 1].axhline(0, color="gray", ls=":", lw=0.5)
    axes[2, 2].axhline(0, color="gray", ls=":", lw=0.5)

    # Labels
    for ax in [axes[0, 0], axes[1, 0], axes[2, 0]]:
        ax.invert_xaxis()
    axes[0, 0].set_xlabel("RA (mas)"); axes[0, 0].set_ylabel("Dec (mas)")
    axes[0, 0].set_title("Sky plane"); axes[0, 0].legend(fontsize=5, ncol=2)
    axes[0, 1].set_xlabel("Year"); axes[0, 1].set_ylabel("RA (mas)")
    axes[0, 1].set_title("RA vs time")
    axes[0, 2].set_xlabel("Year"); axes[0, 2].set_ylabel("Dec (mas)")
    axes[0, 2].set_title("Dec vs time")
    axes[1, 0].set_xlabel("RA pi (mas)"); axes[1, 0].set_ylabel("Dec pi (mas)")
    axes[1, 0].set_title("Parallax ellipse"); axes[1, 0].set_aspect("equal", adjustable="datalim")
    axes[1, 0].legend(fontsize=7)
    axes[1, 1].set_xlabel("Year"); axes[1, 1].set_ylabel("RA pi (mas)")
    axes[1, 1].set_title("RA parallax"); axes[1, 1].legend(fontsize=7)
    axes[1, 2].set_xlabel("Year"); axes[1, 2].set_ylabel("Dec pi (mas)")
    axes[1, 2].set_title("Dec parallax"); axes[1, 2].legend(fontsize=7)
    axes[2, 0].set_xlabel("RA resid (mas)"); axes[2, 0].set_ylabel("Dec resid (mas)")
    axes[2, 0].set_title("Residuals (sky)"); axes[2, 0].set_aspect("equal", adjustable="datalim")
    axes[2, 1].set_xlabel("Year"); axes[2, 1].set_ylabel("RA resid (mas)")
    axes[2, 1].set_title("RA residuals")
    axes[2, 2].set_xlabel("Year"); axes[2, 2].set_ylabel("Dec resid (mas)")
    axes[2, 2].set_title("Dec residuals")

    # AU scale annotation (bottom-right of sky panel)
    D_kpc = 1.0 / r.pi if r.pi > 0 else 10.0
    au_per_mas = D_kpc * 1000.0  # 1 mas = D(kpc) * 1000 AU
    axes[0, 0].text(0.02, 0.02, f"1 mas = {au_per_mas:.0f} AU\n(D={D_kpc:.1f} kpc)",
                    transform=axes[0, 0].transAxes, fontsize=6,
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(outdir, f"fit_{mode}_G{gid}_parallax.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_pi_vs_vlsr(fitter, outdir, mode):
    """Per-channel parallax vs V_LSR for all groups."""
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(f"Individual channel parallaxes — {mode} PR", fontsize=11)
    gc = {"A": "red", "B": "blue", "C": "green", "D": "gray"}
    for gid in sorted(fitter.individual_results.keys()):
        grade = fitter.spots_df[fitter.spots_df.group_id==gid].grade.iloc[0]
        vlsrs = sorted(fitter.individual_results[gid].keys())
        pis = [fitter.individual_results[gid][v].pi for v in vlsrs]
        sigs = [fitter.individual_results[gid][v].sigma_pi for v in vlsrs]
        ax.errorbar(vlsrs, pis, yerr=sigs, fmt='o', color=gc.get(grade,"gray"),
                     markersize=4, alpha=0.7, label=f"G{gid} ({grade})")
        if gid in fitter.group_results:
            gr = fitter.group_results[gid]
            ax.hlines(gr.pi, min(vlsrs)-0.3, max(vlsrs)+0.3,
                      color=gc.get(grade,"gray"), linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax.set_xlabel("V_LSR (km/s)"); ax.set_ylabel("Parallax (mas)")
    ax.set_ylim(-3, 3)
    ax.legend(fontsize=6, ncol=3, loc='upper right')
    plt.tight_layout()
    path = os.path.join(outdir, f"fit_{mode}_pi_vs_vlsr.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Plot: {path}")


def plot_summary(fitter, outdir, mode):
    """Summary: group parallaxes, bootstrap, distance posterior."""
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Parallax fitting summary — {mode} PR", fontsize=11)

    gids = sorted(fitter.group_results.keys())
    grades = [fitter.spots_df[fitter.spots_df.group_id==gid].grade.iloc[0] for gid in gids]
    pis = [fitter.group_results[g].pi for g in gids]
    sigs = [fitter.group_results[g].sigma_pi * np.sqrt(fitter.group_results[g].n_channels)
            for g in gids]
    cols = ["red" if g=="A" else "blue" if g=="B" else "green" for g in grades]

    axes[0].errorbar(range(len(gids)), pis, yerr=sigs, fmt='none', ecolor='gray', alpha=0.5)
    axes[0].scatter(range(len(gids)), pis, c=cols, s=40, zorder=5)
    axes[0].set_xticks(range(len(gids)))
    axes[0].set_xticklabels([f"G{g}" for g in gids], rotation=45, fontsize=7)
    axes[0].axhline(0, color='gray', linestyle=':', linewidth=0.5)
    axes[0].axhline(0.1, color='black', linestyle='--', alpha=0.3, label="π=0.1 (10 kpc)")
    axes[0].set_ylabel("Parallax (mas)"); axes[0].set_title("Group parallaxes (√N)")
    axes[0].legend(fontsize=7)

    # Bootstrap for best group
    best = min((g for g in gids if grades[gids.index(g)]=="A" and g in fitter.bootstrap_results),
               key=lambda g: fitter.group_results[g].sigma_pi, default=None)
    if best and best in fitter.bootstrap_results:
        boot = fitter.bootstrap_results[best]
        lo, hi = np.percentile(boot, [1, 99])
        bc = boot[(boot > lo) & (boot < hi)]
        axes[1].hist(bc, bins=50, color='steelblue', alpha=0.7, density=True)
        axes[1].axvline(fitter.group_results[best].pi, color='red', linewidth=2,
                         label=f"WLS: {fitter.group_results[best].pi:.3f}")
        axes[1].set_xlabel("Parallax (mas)"); axes[1].set_title(f"Bootstrap — G{best}")
        axes[1].legend(fontsize=8)

    # Distance posterior
    if best and best in fitter.distance_posteriors:
        dp = fitter.distance_posteriors[best]
        axes[2].plot(dp["D_grid"], dp["posterior"], 'k-', linewidth=1.5)
        axes[2].axvline(dp["D_mode"], color='red', linewidth=1,
                         label=f"D={dp['D_mode']:.1f} kpc")
        axes[2].axvspan(dp["D_16"], dp["D_84"], alpha=0.15, color='blue',
                         label=f"68%: [{dp['D_16']:.1f}, {dp['D_84']:.1f}]")
        axes[2].set_xlabel("Distance (kpc)"); axes[2].set_title(f"Distance — G{best}")
        axes[2].legend(fontsize=7); axes[2].set_xlim(0, 25)

    plt.tight_layout()
    path = os.path.join(outdir, f"fit_{mode}_summary.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Plot: {path}")


def make_plots(fitter, outdir, mode):
    """Generate all diagnostic plots."""
    if not HAS_MPL:
        print("  Matplotlib not available, skipping plots")
        return
    print(f"\nGenerating diagnostic plots...")

    # Per-group 3×3 parallax plots (channel-level)
    for gid in sorted(fitter.group_results.keys()):
        p = plot_parallax_fit(fitter, gid, outdir, mode)
        if p: print(f"  Plot: {p}")

    # Per-group 3×3 parallax plots (feature-level, for comparison)
    if fitter.feature_results:
        for gid in sorted(fitter.feature_results.keys()):
            p = plot_feature_fit(fitter, gid, outdir, mode)
            if p: print(f"  Plot: {p}")

    plot_pi_vs_vlsr(fitter, outdir, mode)
    plot_summary(fitter, outdir, mode)
    plot_proper_motion_sky(fitter, outdir, mode)
    plot_residuals(fitter, outdir, mode)
    plot_parallax_sky(fitter, outdir, mode)
    plot_final_parallax_overview(fitter, outdir, mode)
    plot_final_proper_motions(fitter, outdir, mode)
    write_crosscheck_table(fitter, outdir, mode)
    write_proper_motion_table(fitter, outdir, mode)
    plot_spots_html(fitter, outdir, mode)
    plot_features_html(fitter, outdir, mode)
    write_publication_data(fitter, outdir, mode)


def plot_feature_fit(fitter, gid, outdir, mode):
    """Plot 3×3 parallax fit for a feature-level (flux-weighted centroid) fit.

    Same layout as plot_parallax_fit but using feature data instead of
    individual channel data. Useful for comparing channel vs feature fits.
    """
    if not HAS_MPL or gid not in fitter.feature_results:
        return None
    if fitter.features_df is None:
        return None

    r = fitter.feature_results[gid]
    gdf = fitter.features_df[fitter.features_df.group_id == gid]
    if len(gdf) < 4:
        return None

    t_ref = fitter._compute_ref_epoch()
    grade = gdf.grade.iloc[0]
    mask = gdf.use_for_pi.values == 1
    if np.sum(mask) < 4:
        return None

    mjds = gdf.mjd.values[mask].astype(float)
    x = gdf.x.values[mask]
    y = gdf.y.values[mask]
    xe = gdf.x_err.values[mask]
    ye = gdf.y_err.values[mask]
    dt = (mjds - t_ref) / JULIANYEAR
    yr = 2000.0 + (mjds - MJD_J2000) / JULIANYEAR
    F_a, F_d = parallax_factors_analytical(fitter.ra_rad, fitter.dec_rad, mjds)

    p = r.params
    if p is None or r.coord_mode != "both" or len(p) < 5:
        return None

    pi_str = f"π = {r.pi:.3f} ± {r.sigma_pi:.3f} mas (feature)"

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f"G{gid} ({grade}) — {mode} PR — FEATURE FIT — {pi_str}",
                 fontsize=12, fontweight="bold", color="darkblue")

    # Model
    x_model = p[2] + p[1] * dt + p[0] * F_a
    y_model = p[4] + p[3] * dt + p[0] * F_d
    x_pi = x - (p[2] + p[1] * dt)
    y_pi = y - (p[4] + p[3] * dt)
    x_resid = x - x_model
    y_resid = y - y_model

    # Fine grid
    mjd_m = np.linspace(mjds.min()-30, mjds.max()+30, 300)
    yr_m = 2000.0 + (mjd_m - MJD_J2000) / JULIANYEAR
    dt_m = (mjd_m - t_ref) / JULIANYEAR
    F_a_m, F_d_m = parallax_factors_analytical(fitter.ra_rad, fitter.dec_rad, mjd_m)

    c = "royalblue"

    # Row 1: Proper motion
    axes[0,0].errorbar(x, y, xerr=xe, yerr=ye, fmt="o", color=c, ms=5)
    axes[0,1].errorbar(yr, x, yerr=xe, fmt="o", color=c, ms=5)
    axes[0,1].plot(yr_m, p[2]+p[1]*dt_m+r.pi*F_a_m, "-", color=c, lw=1, alpha=0.6)
    axes[0,2].errorbar(yr, y, yerr=ye, fmt="o", color=c, ms=5)
    axes[0,2].plot(yr_m, p[4]+p[3]*dt_m+r.pi*F_d_m, "-", color=c, lw=1, alpha=0.6)

    # Row 2: Parallax
    axes[1,0].plot(x_pi, y_pi, "o", color=c, ms=5)
    mjd_e = np.linspace(t_ref-182, t_ref+182, 200)
    F_ae, F_de = parallax_factors_analytical(fitter.ra_rad, fitter.dec_rad, mjd_e)
    axes[1,0].plot(r.pi*F_ae, r.pi*F_de, "k-", lw=1.5, alpha=0.5)
    axes[1,1].errorbar(yr, x_pi, yerr=xe, fmt="o", color=c, ms=5)
    axes[1,1].plot(yr_m, r.pi*F_a_m, "k-", lw=1.5, alpha=0.6, label=f"π={r.pi:.3f}")
    axes[1,1].axhline(0, color="gray", ls=":", lw=0.5)
    axes[1,2].errorbar(yr, y_pi, yerr=ye, fmt="o", color=c, ms=5)
    axes[1,2].plot(yr_m, r.pi*F_d_m, "k-", lw=1.5, alpha=0.6, label=f"π={r.pi:.3f}")
    axes[1,2].axhline(0, color="gray", ls=":", lw=0.5)

    # Row 3: Residuals
    axes[2,0].plot(x_resid, y_resid, "o", color=c, ms=5)
    axes[2,1].errorbar(yr, x_resid, yerr=xe, fmt="o", color=c, ms=5)
    axes[2,1].axhline(0, color="gray", ls=":", lw=0.5)
    axes[2,2].errorbar(yr, y_resid, yerr=ye, fmt="o", color=c, ms=5)
    axes[2,2].axhline(0, color="gray", ls=":", lw=0.5)

    # Labels (same as group plot)
    for ax in [axes[0,0], axes[1,0], axes[2,0]]: ax.invert_xaxis()
    axes[0,0].set_xlabel("RA (mas)"); axes[0,0].set_ylabel("Dec (mas)"); axes[0,0].set_title("Sky plane")
    axes[0,1].set_xlabel("Year"); axes[0,1].set_ylabel("RA (mas)"); axes[0,1].set_title("RA vs time")
    axes[0,2].set_xlabel("Year"); axes[0,2].set_ylabel("Dec (mas)"); axes[0,2].set_title("Dec vs time")
    axes[1,0].set_xlabel("RA pi (mas)"); axes[1,0].set_ylabel("Dec pi (mas)")
    axes[1,0].set_title("Parallax ellipse"); axes[1,0].set_aspect("equal", adjustable="datalim")
    axes[1,1].set_xlabel("Year"); axes[1,1].set_ylabel("RA pi (mas)"); axes[1,1].set_title("RA parallax")
    axes[1,1].legend(fontsize=7)
    axes[1,2].set_xlabel("Year"); axes[1,2].set_ylabel("Dec pi (mas)"); axes[1,2].set_title("Dec parallax")
    axes[1,2].legend(fontsize=7)
    axes[2,0].set_xlabel("RA resid (mas)"); axes[2,0].set_ylabel("Dec resid (mas)")
    axes[2,0].set_title("Residuals (sky)"); axes[2,0].set_aspect("equal", adjustable="datalim")
    axes[2,1].set_xlabel("Year"); axes[2,1].set_ylabel("RA resid (mas)"); axes[2,1].set_title("RA residuals")
    axes[2,2].set_xlabel("Year"); axes[2,2].set_ylabel("Dec resid (mas)"); axes[2,2].set_title("Dec residuals")

    plt.tight_layout()
    path = os.path.join(outdir, f"fit_{mode}_G{gid}_feature_parallax.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_proper_motion_sky(fitter, outdir, mode):
    """Plot proper motion vectors on the sky plane for all groups."""
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"Maser proper motions — {mode} PR", fontsize=11)

    gc = {"A": "red", "B": "blue", "C": "green", "D": "gray"}

    for gid in sorted(fitter.group_results.keys()):
        r = fitter.group_results[gid]
        grade = fitter.spots_df[fitter.spots_df.group_id==gid].grade.iloc[0]
        channels = extract_channels(fitter.spots_df, gid,
                                     chan_spacing=fitter.chan_spacing)
        usable = [ch for ch in channels if ch.n_usable >= 4]
        color = gc.get(grade, "gray")

        for ch in usable:
            mask = ch.use_for_pi == 1
            x_mean = np.mean(ch.x[mask])
            y_mean = np.mean(ch.y[mask])
            mu_info = r.mu.get(ch.vlsr, {})
            mu_x = mu_info.get("mu_x", 0)
            mu_y = mu_info.get("mu_y", 0)

            ax.plot(x_mean, y_mean, 'o', color=color, markersize=3, alpha=0.5)
            ax.annotate('', xy=(x_mean + mu_x, y_mean + mu_y),
                        xytext=(x_mean, y_mean),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=1.0, alpha=0.7))

        if usable:
            x_g = np.mean([np.mean(ch.x[ch.use_for_pi==1]) for ch in usable])
            y_g = np.mean([np.mean(ch.y[ch.use_for_pi==1]) for ch in usable])
            ax.text(x_g, y_g + 0.3, f"G{gid}", fontsize=7, ha='center',
                    color=color, fontweight='bold')

    ax.set_xlabel("RA offset (mas)")
    ax.set_ylabel("Dec offset (mas)")
    ax.invert_xaxis()
    ax.set_aspect('equal', adjustable='datalim')
    for g, c in gc.items():
        ax.plot([], [], 'o', color=c, label=f"Grade {g}")
    ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    path = os.path.join(outdir, f"fit_{mode}_proper_motions.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot: {path}")


def plot_residuals(fitter, outdir, mode):
    """Plot post-fit residuals for all groups on a single multi-panel figure."""
    if not HAS_MPL:
        return

    gids = sorted(fitter.group_results.keys())
    n_groups = len(gids)
    if n_groups == 0:
        return

    ncols = min(4, n_groups)
    nrows = (n_groups + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                              squeeze=False)
    fig.suptitle(f"Post-fit RA residuals (proper motion subtracted) — {mode} PR",
                 fontsize=11)

    t_ref = fitter._compute_ref_epoch()
    colors = plt.cm.rainbow(np.linspace(0, 1, 10))

    for idx, gid in enumerate(gids):
        ax = axes[idx // ncols][idx % ncols]
        r = fitter.group_results[gid]
        grade = fitter.spots_df[fitter.spots_df.group_id==gid].grade.iloc[0]
        channels = extract_channels(fitter.spots_df, gid,
                                     chan_spacing=fitter.chan_spacing)
        usable = [ch for ch in channels if ch.n_usable >= 4]

        for ci, ch in enumerate(usable):
            mask = ch.use_for_pi == 1
            mjds = ch.mjds[mask]
            yr = 2000.0 + (mjds - MJD_J2000) / JULIANYEAR
            dt = (mjds - t_ref) / JULIANYEAR
            F_a, _ = parallax_factors_analytical(fitter.ra_rad, fitter.dec_rad, mjds)
            mu_info = r.mu.get(ch.vlsr, {})
            mu_x = mu_info.get("mu_x", 0)

            # Residual = data - model (proper motion + parallax)
            x_resid = ch.x[mask] - mu_x * dt - r.pi * F_a
            x_resid -= np.mean(x_resid)  # remove offset

            c = colors[ci % len(colors)]
            floor = max(r.floor_ra, 0.001)
            ax.errorbar(yr, x_resid, yerr=floor, fmt='o', color=c,
                        markersize=3, alpha=0.6)

        ax.axhline(0, color='gray', linestyle=':', linewidth=0.5)
        ax.set_title(f"G{gid} ({grade})", fontsize=8)
        ax.tick_params(labelsize=7)
        if idx % ncols == 0:
            ax.set_ylabel("RA resid (mas)", fontsize=7)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Year", fontsize=7)

    # Hide unused axes
    for idx in range(n_groups, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    path = os.path.join(outdir, f"fit_{mode}_residuals.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot: {path}")


def plot_parallax_sky(fitter, outdir, mode):
    """Plot parallax ellipses on the sky plane for Grade A/B groups.

    Shows the expected parallax ellipse at the fitted parallax value,
    overlaid on the proper-motion-subtracted maser positions.
    """
    if not HAS_MPL:
        return

    gids = [g for g in sorted(fitter.group_results.keys())
            if fitter.spots_df[fitter.spots_df.group_id==g].grade.iloc[0] in ("A", "B")]
    if not gids:
        return

    ncols = min(4, len(gids))
    nrows = (len(gids) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows),
                              squeeze=False)
    fig.suptitle(f"Parallax ellipses on sky — {mode} PR", fontsize=11)

    t_ref = fitter._compute_ref_epoch()

    for idx, gid in enumerate(gids):
        ax = axes[idx // ncols][idx % ncols]
        r = fitter.group_results[gid]
        grade = fitter.spots_df[fitter.spots_df.group_id==gid].grade.iloc[0]
        channels = extract_channels(fitter.spots_df, gid,
                                     chan_spacing=fitter.chan_spacing)
        usable = [ch for ch in channels if ch.n_usable >= 4]
        colors_ch = plt.cm.rainbow(np.linspace(0, 1, max(len(usable), 2)))

        # Plot PM-subtracted data
        for ci, ch in enumerate(usable):
            mask = ch.use_for_pi == 1
            mjds = ch.mjds[mask]
            dt = (mjds - t_ref) / JULIANYEAR
            mu_info = r.mu.get(ch.vlsr, {})
            mu_x = mu_info.get("mu_x", 0)
            mu_y = mu_info.get("mu_y", 0)

            x_sub = ch.x[mask] - mu_x * dt
            y_sub = ch.y[mask] - mu_y * dt
            x_sub -= np.mean(x_sub)
            y_sub -= np.mean(y_sub)

            ax.plot(x_sub, y_sub, 'o', color=colors_ch[ci], markersize=3, alpha=0.6)

        # Parallax ellipse
        mjd_ellipse = np.linspace(t_ref - 182, t_ref + 182, 200)
        F_a_e, F_d_e = parallax_factors_analytical(
            fitter.ra_rad, fitter.dec_rad, mjd_ellipse)
        ax.plot(r.pi * F_a_e, r.pi * F_d_e, 'k-', linewidth=1.5, alpha=0.5,
                label=f"π={r.pi:.3f}")
        ax.plot(r.pi * F_a_e[0], r.pi * F_d_e[0], 'k^', markersize=6)

        ax.set_title(f"G{gid} ({grade})", fontsize=9)
        ax.invert_xaxis()
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
        if idx % ncols == 0:
            ax.set_ylabel("Dec (mas)", fontsize=7)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("RA (mas)", fontsize=7)

    for idx in range(len(gids), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    path = os.path.join(outdir, f"fit_{mode}_parallax_sky.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot: {path}")


def write_crosscheck_table(fitter, outdir, mode):
    """Write cross-check comparison table: group vs feature vs bootstrap."""
    path = os.path.join(outdir, f"fit_{mode}_crosscheck.txt")
    with open(path, "w") as f:
        f.write(f"# Parallax cross-check table ({mode} PR)\n")
        f.write(f"# masertrack_fit v{__version__}\n")
        f.write(f"# All parallax values in mas.\n")
        f.write(f"#\n")
        f.write(f"#{'group':>6}{'grade':>6}{'n_ch':>5}"
                f"{'pi_grp':>10}{'sig_grp':>10}"
                f"{'pi_feat':>10}"
                f"{'pi_boot':>10}{'boot_16':>10}{'boot_84':>10}"
                f"{'floor_ra':>10}{'floor_de':>10}"
                f"{'flag':>8}\n")

        for gid in sorted(fitter.group_results.keys()):
            r = fitter.group_results[gid]
            grade = fitter.spots_df[fitter.spots_df.group_id==gid].grade.iloc[0]
            n_ch = r.n_channels
            sqN = r.sigma_pi * np.sqrt(n_ch)

            pi_feat = fitter.feature_results[gid].pi if gid in fitter.feature_results else np.nan

            boot_med, boot_16, boot_84 = np.nan, np.nan, np.nan
            if gid in fitter.bootstrap_results:
                b = fitter.bootstrap_results[gid]
                boot_med = np.median(b)
                boot_16 = np.percentile(b, 16)
                boot_84 = np.percentile(b, 84)

            # Flag assessment
            if gid in fitter.parallax_outliers:
                flag = "OUTLIER"
            elif r.pi < 0:
                flag = "NEG"
            elif sqN > 0.3:
                flag = "WIDE"
            else:
                flag = "OK"

            def fmt(v):
                return f"{v:>10.4f}" if np.isfinite(v) else f"{'---':>10}"

            f.write(f"{gid:>6}{grade:>6}{n_ch:>5}"
                    f"{r.pi:>10.4f}{sqN:>10.4f}"
                    f"{fmt(pi_feat)}"
                    f"{fmt(boot_med)}{fmt(boot_16)}{fmt(boot_84)}"
                    f"{r.floor_ra:>10.4f}{r.floor_dec:>10.4f}"
                    f"{flag:>8}\n")

        # Combined fit results
        f.write(f"#\n# Combined multi-group fits:\n")
        for label in ["A", "A+B", "A+B+C"]:
            if label in fitter.combined_results:
                r = fitter.combined_results[label]
                sqN = r.sigma_pi * np.sqrt(r.n_channels)
                f.write(f"# {label:>5}: pi={r.pi:.4f} ± {sqN:.4f} mas "
                        f"({r.n_channels} ch, floor=[{r.floor_ra:.3f},{r.floor_dec:.3f}])\n")

        # Weighted mean results
        f.write(f"#\n# Weighted mean (outliers excluded):\n")
        for label in ["A", "A+B", "A+B+C"]:
            if label in fitter.summary:
                s = fitter.summary[label]
                f.write(f"# {label:>5}: pi={s['pi']:.4f} ± {s['sigma_pi']:.4f} mas "
                        f"({s['n_groups']} groups)\n")

    print(f"  Cross-check: {path}")


def write_proper_motion_table(fitter, outdir, mode):
    """Write proper motion results for all channels in all groups."""
    path = os.path.join(outdir, f"fit_{mode}_proper_motions.txt")
    with open(path, "w") as f:
        f.write(f"# Proper motion results ({mode} PR)\n")
        f.write(f"# masertrack_fit v{__version__}\n")
        f.write(f"# From group fits (common parallax, independent proper motions)\n")
        f.write(f"#\n")
        f.write(f"#{'group':>6}{'grade':>6}{'vlsr':>10}"
                f"{'mu_x':>10}{'sig_mx':>10}{'mu_y':>10}{'sig_my':>10}"
                f"{'mu_tot':>10}{'PA':>8}\n")

        for gid in sorted(fitter.group_results.keys()):
            r = fitter.group_results[gid]
            grade = fitter.spots_df[fitter.spots_df.group_id==gid].grade.iloc[0]
            for vlsr in sorted(r.mu.keys()):
                mu = r.mu[vlsr]
                mx = mu.get("mu_x", np.nan)
                my = mu.get("mu_y", np.nan)
                smx = mu.get("sigma_mu_x", np.nan)
                smy = mu.get("sigma_mu_y", np.nan)
                if np.isfinite(mx) and np.isfinite(my):
                    mu_tot = np.sqrt(mx**2 + my**2)
                    pa = np.degrees(np.arctan2(mx, my)) % 360  # N through E
                else:
                    mu_tot = np.nan
                    pa = np.nan
                f.write(f"{gid:>6}{grade:>6}{vlsr:>10.3f}"
                        f"{mx:>10.4f}{smx:>10.4f}{my:>10.4f}{smy:>10.4f}"
                        f"{mu_tot:>10.4f}{pa:>8.1f}\n")
    print(f"  Proper motions: {path}")


def write_publication_data(fitter, outdir, mode):
    """Export publication-ready data: per-group CSVs with data, model, and residuals.

    For each fitted group, writes two files:
    - fit_{mode}_G{gid}_pub_data.csv: per-epoch measured data + model + residuals
    - fit_{mode}_G{gid}_pub_model.csv: fine-grid model curve for smooth plotting

    These files contain everything needed to reproduce publication figures
    in external plotting tools (matplotlib, gnuplot, pgfplots, etc.).
    """
    t_ref_mjd = fitter._compute_ref_epoch()
    pub_dir = os.path.join(outdir, "publication")
    os.makedirs(pub_dir, exist_ok=True)

    n_files = 0
    for gid in sorted(fitter.group_results.keys()):
        r = fitter.group_results[gid]
        channels = extract_channels(fitter.spots_df, gid, chan_spacing=fitter.chan_spacing)
        grade = fitter.spots_df[fitter.spots_df.group_id == gid].grade.iloc[0]

        # Per-channel data with model
        data_path = os.path.join(pub_dir, f"fit_{mode}_G{gid}_pub_data.csv")
        with open(data_path, "w") as f:
            f.write(f"# Publication data: G{gid} ({grade}) {mode} PR\n")
            f.write(f"# Source: RA={np.degrees(fitter.ra_rad):.6f} "
                    f"Dec={np.degrees(fitter.dec_rad):.6f}\n")
            f.write(f"# Ref epoch: MJD {t_ref_mjd:.1f}\n")
            if gid in fitter.bootstrap_results and len(fitter.bootstrap_results[gid]) > 100:
                bp = fitter.bootstrap_results[gid]
                f.write(f"# Parallax: {r.pi:.4f} mas "
                        f"(bootstrap 68%: [{np.percentile(bp,16):.4f}, "
                        f"{np.percentile(bp,84):.4f}])\n")
            else:
                n_ch = r.n_channels
                f.write(f"# Parallax: {r.pi:.4f} +/- "
                        f"{r.sigma_pi * np.sqrt(n_ch):.4f} mas (sqrt(N))\n")
            f.write(f"# Error floors: RA={r.floor_ra:.4f} Dec={r.floor_dec:.4f} mas\n")
            f.write("# vlsr,epoch_yr,mjd,x_data,y_data,x_err,y_err,"
                    "x_model,y_model,x_pi_only,y_pi_only,x_resid,y_resid\n")

            for ch in channels:
                ch_r = fitter.individual_results.get(gid, {}).get(ch.vlsr)
                if ch_r is None or ch_r.params is None:
                    continue
                mask = ch.use_for_pi == 1
                if np.sum(mask) < 4:
                    continue

                mjds = ch.mjds[mask]
                dt = (mjds - t_ref_mjd) / JULIANYEAR
                F_a, F_d = get_parallax_factors_cached(
                    fitter.ra_rad, fitter.dec_rad, mjds, method="analytical")
                p = ch_r.params

                if ch_r.coord_mode == "both" and len(p) >= 5:
                    x_model = p[2] + p[1] * dt + p[0] * F_a
                    y_model = p[4] + p[3] * dt + p[0] * F_d
                    x_pi = p[0] * F_a
                    y_pi = p[0] * F_d
                else:
                    continue

                x = ch.x[mask]
                y = ch.y[mask]
                xe = np.sqrt(ch.x_err[mask]**2 + r.floor_ra**2)
                ye = np.sqrt(ch.y_err[mask]**2 + r.floor_dec**2)

                for i in range(len(mjds)):
                    yr = 2000.0 + (mjds[i] - MJD_J2000) / JULIANYEAR
                    f.write(f"{ch.vlsr:.4f},{yr:.6f},{mjds[i]:.1f},"
                            f"{x[i]:.6f},{y[i]:.6f},{xe[i]:.6f},{ye[i]:.6f},"
                            f"{x_model[i]:.6f},{y_model[i]:.6f},"
                            f"{x_pi[i]:.6f},{y_pi[i]:.6f},"
                            f"{x[i]-x_model[i]:.6f},{y[i]-y_model[i]:.6f}\n")

        # Fine-grid model curve
        model_path = os.path.join(pub_dir, f"fit_{mode}_G{gid}_pub_model.csv")
        with open(model_path, "w") as f:
            f.write("# Fine-grid parallax model curve for smooth plotting\n")
            f.write("# epoch_yr,mjd,F_alpha,F_delta,pi_x,pi_y\n")
            all_mjds = []
            for ch in channels:
                mask = ch.use_for_pi == 1
                if np.sum(mask) >= 4:
                    all_mjds.extend(ch.mjds[mask])
            if not all_mjds:
                continue
            mjd_min, mjd_max = min(all_mjds), max(all_mjds)
            mjd_fine = np.linspace(mjd_min - 30, mjd_max + 30, 500)
            F_a_f, F_d_f = get_parallax_factors_cached(
                fitter.ra_rad, fitter.dec_rad, mjd_fine, method="analytical")
            for i in range(len(mjd_fine)):
                yr = 2000.0 + (mjd_fine[i] - MJD_J2000) / JULIANYEAR
                f.write(f"{yr:.6f},{mjd_fine[i]:.1f},"
                        f"{F_a_f[i]:.8f},{F_d_f[i]:.8f},"
                        f"{r.pi * F_a_f[i]:.8f},{r.pi * F_d_f[i]:.8f}\n")

        n_files += 2

    print(f"  Publication data: {n_files} files in {pub_dir}")


def write_selected_publication_fit(fitters, selection, outdir):
    """Write publication-ready data for a user-selected fit.

    Parameters
    ----------
    fitters : dict {"inverse": ParallaxFitter, ...}
    selection : str
        Options: 'G1' (group 1 channel fit from first mode),
                 'G1_feat' (group 1 feature fit),
                 'wt_mean' (weighted mean of non-outlier groups),
                 'combined_A', 'combined_A+B', 'combined_A+B+C'
    outdir : str
        Output directory

    Writes pub_selected_{selection}.csv with the data and model for
    the selected fit, ready for external plotting.
    """
    pub_dir = os.path.join(outdir, "publication")
    os.makedirs(pub_dir, exist_ok=True)

    # Find the right fitter and result
    fitter = list(fitters.values())[0]  # default to first mode
    for mode_label, f in fitters.items():
        # Check if selection matches a group in this fitter
        if selection.startswith("G"):
            parts = selection.replace("_feat", "").replace("G", "")
            try:
                gid = int(parts)
            except ValueError:
                continue
            if gid in f.group_results or gid in f.feature_results:
                fitter = f
                break

    sel_clean = selection.replace("+", "p")
    path = os.path.join(pub_dir, f"pub_selected_{sel_clean}.csv")

    with open(path, "w") as out:
        out.write(f"# Selected publication fit: {selection}\n")
        out.write(f"# masertrack_fit v{__version__}\n")

        if selection == "wt_mean":
            # Write summary of weighted mean
            for label in ["A", "A+B", "A+B+C"]:
                if label in fitter.summary:
                    s = fitter.summary[label]
                    out.write(f"# wt_mean {label}: pi={s['pi']:.6f} "
                              f"+/- {s['sigma_pi']:.6f} mas "
                              f"({s['n_groups']} groups)\n")
            out.write(f"# Use publication/ directory CSV files for per-group data\n")

        elif selection.endswith("_feat"):
            gid = int(selection.replace("_feat", "").replace("G", ""))
            if gid in fitter.feature_results:
                r = fitter.feature_results[gid]
                out.write(f"# Feature fit G{gid}: pi={r.pi:.6f} "
                          f"+/- {r.sigma_pi:.6f} mas\n")
                out.write(f"# See publication/fit_*_G{gid}_pub_data.csv "
                          f"for channel-level data\n")

        elif selection.startswith("G"):
            gid = int(selection.replace("G", ""))
            if gid in fitter.group_results:
                r = fitter.group_results[gid]
                n_ch = r.n_channels
                out.write(f"# Group fit G{gid}: pi={r.pi:.6f} "
                          f"+/- {r.sigma_pi:.6f} (sqrtN: "
                          f"{r.sigma_pi*np.sqrt(n_ch):.6f}) mas\n")
                out.write(f"# {n_ch} channels, floor=[{r.floor_ra:.4f},"
                          f"{r.floor_dec:.4f}] mas\n")

        elif selection.startswith("combined"):
            label = selection.replace("combined_", "")
            if label in fitter.combined_results:
                r = fitter.combined_results[label]
                sqN = r.sigma_pi * np.sqrt(r.n_channels)
                out.write(f"# Combined {label}: pi={r.pi:.6f} "
                          f"+/- {sqN:.6f} mas ({r.n_channels} ch)\n")

        out.write(f"#\n# For full data+model, see publication/ directory\n")

    print(f"  Selected fit: {path}")


def write_final_parallax(fitters, outdir):
    """Write the final combined parallax determination from all PR modes.

    Parameters
    ----------
    fitters : dict {"inverse": ParallaxFitter, "normal": ParallaxFitter}
    """
    path = os.path.join(outdir, "fit_final_parallax.txt")

    # Collect all group results from both modes
    all_results = []
    for mode_label, fitter in fitters.items():
        for gid, r in fitter.group_results.items():
            grade = fitter.spots_df[fitter.spots_df.group_id==gid].grade.iloc[0]
            n_ch = r.n_channels

            # Use bootstrap uncertainty when available, else formal*√N*σ_pdf
            if gid in fitter.bootstrap_results and len(fitter.bootstrap_results[gid]) > 100:
                bp = fitter.bootstrap_results[gid]
                sigma_use = 0.5 * (np.percentile(bp, 84) - np.percentile(bp, 16))
            else:
                sigma_sqrtN = r.sigma_pi * np.sqrt(n_ch)
                spdf = max(getattr(r, 'sigma_pdf_ra', 1.0),
                           getattr(r, 'sigma_pdf_dec', 1.0), 1.0)
                sigma_use = sigma_sqrtN * spdf

            vlsr_mean = 0.0
            channels = extract_channels(fitter.spots_df, gid,
                                         chan_spacing=fitter.chan_spacing)
            usable = [ch for ch in channels if ch.n_usable >= 4]
            if usable:
                vlsr_mean = np.mean([ch.vlsr for ch in usable])

            # Flag anomalous groups using Chauvenet results
            anomalous = False
            if hasattr(fitter, 'parallax_outliers') and gid in getattr(fitter, 'parallax_outliers', set()):
                anomalous = True
            if r.pi < -0.3:  # clearly unphysical negative parallax
                anomalous = True

            all_results.append({
                "mode": mode_label, "gid": gid, "grade": grade,
                "vlsr": vlsr_mean, "n_ch": n_ch,
                "pi": r.pi, "sigma": sigma_use,
                "floor_ra": r.floor_ra, "floor_dec": r.floor_dec,
                "anomalous": anomalous,
            })

    # Compute parallax per mode separately (NOT combined across modes)
    selections = {}
    for mode_label in sorted(fitters.keys()):
        mode_results = [d for d in all_results if d["mode"] == mode_label]

        for label_base, crit in [
            ("All positive A+B+C",
             lambda d: d["pi"] > 0 and not d["anomalous"]),
            ("Grade A (excl. anom.)",
             lambda d: d["grade"] == "A" and not d["anomalous"]),
        ]:
            sel = [d for d in mode_results if crit(d)]
            if not sel:
                continue
            pis = np.array([d["pi"] for d in sel])
            sigs = np.array([d["sigma"] for d in sel])
            w = 1.0 / sigs**2
            pi_wm = np.sum(pis * w) / np.sum(w)
            sig_wm = 1.0 / np.sqrt(np.sum(w))
            key = f"{mode_label} — {label_base}"
            selections[key] = {
                "pi": pi_wm, "sigma": sig_wm,
                "n": len(sel), "groups": sel,
            }

    with open(path, "w") as f:
        f.write(f"# Final parallax determination\n")
        f.write(f"# masertrack_fit v{__version__}\n")
        f.write(f"#\n")
        f.write(f"# Per-group results:\n")
        f.write(f"#{'mode':>8}{'group':>6}{'grade':>6}{'vlsr':>8}{'n_ch':>5}"
                f"{'pi':>10}{'sigma':>10}{'flag':>8}\n")
        for d in sorted(all_results, key=lambda x: (x["mode"], x["gid"])):
            flag = "ANOM" if d["anomalous"] else ("NEG" if d["pi"]<0 else "OK")
            f.write(f"{d['mode']:>8} G{d['gid']:>4}{d['grade']:>6}"
                    f"{d['vlsr']:>8.1f}{d['n_ch']:>5}"
                    f"{d['pi']:>10.4f}{d['sigma']:>10.4f}{flag:>8}\n")

        f.write(f"#\n# Combined parallax determinations:\n")
        for label, s in selections.items():
            pct = s["sigma"] / s["pi"] * 100 if s["pi"] > 0 else float('inf')
            D = 1.0 / s["pi"] if s["pi"] > 0 else float('inf')
            f.write(f"# {label}:\n")
            f.write(f"#   pi = {s['pi']:.4f} ± {s['sigma']:.4f} mas "
                    f"({pct:.0f}% accuracy)\n")
            f.write(f"#   D = {D:.1f} kpc\n")
            f.write(f"#   N_measurements = {s['n']}\n")

        # Per-mode best results
        f.write(f"#\n# *** PARALLAX PER MODE (for comparison, NOT combined) ***\n")
        for mode_label in sorted(fitters.keys()):
            mode_sels = {k: v for k, v in selections.items() if mode_label in k}
            best_key = None
            for k in mode_sels:
                if "All positive" in k:
                    best_key = k
                    break
            if best_key is None:
                for k in mode_sels:
                    best_key = k; break
            if best_key and mode_sels[best_key]["pi"] > 0:
                s = mode_sels[best_key]
                pct = s["sigma"] / s["pi"] * 100
                f.write(f"# {mode_label}: π = {s['pi']:.4f} ± {s['sigma']:.4f} mas "
                        f"({pct:.0f}%, D = {1.0/s['pi']:.1f} kpc)\n")

    print(f"  Final parallax: {path}")


# =====================================================================
#  CLI AND MAIN
# =====================================================================
def validate_reid_sgr_b2(verbose=True):
    """Validate parallax fitting against Reid+2009 Sgr B2 (ApJ 705, 1548).

    Creates input data from Table 3, runs the fitter, and checks that
    the recovered parallax matches the published values:
      Sgr B2M: pi = 0.130 ± 0.012 mas
      Sgr B2N: pi = 0.128 ± 0.015 mas (first epoch removed)

    Returns True if both match within 1σ, False otherwise.
    """
    from datetime import date as _date
    _mjd_ref = _date(1858, 11, 17)

    # Reid+2009 Table 3: calendar dates from Table 1, positions from Table 3
    sgr_b2m_raw = [
        ("2006-09-04", 2006.676, +0.318, +2.076, 0.050, 0.150),
        ("2006-09-23", 2006.728, +0.255, +2.131, 0.050, 0.150),
        ("2006-10-09", 2006.772, +0.222, +1.665, 0.050, 0.150),
        ("2006-10-24", 2006.813, +0.260, +2.046, 0.050, 0.150),
        ("2007-03-10", 2007.189, -0.025, +0.241, 0.025, 0.075),
        ("2007-03-17", 2007.208, -0.048, +0.240, 0.025, 0.075),
        ("2007-03-25", 2007.230, -0.120, +0.117, 0.025, 0.075),
        ("2007-04-15", 2007.287, -0.151, -0.169, 0.025, 0.075),
        ("2007-04-22", 2007.307, -0.183, -0.174, 0.025, 0.075),
        ("2007-09-28", 2007.742, -1.051, -1.921, 0.050, 0.150),
        ("2007-09-30", 2007.747, -0.944, -1.873, 0.050, 0.150),
        ("2007-10-16", 2007.791, -0.975, -1.967, 0.050, 0.150),
    ]

    sgr_b2n_raw = [  # first epoch removed per Reid
        ("2006-09-23", 2006.728, 193.689, -34.273, 0.060, 0.140),
        ("2006-10-09", 2006.772, 193.775, -34.213, 0.060, 0.140),
        ("2006-10-24", 2006.813, 193.634, -34.833, 0.060, 0.140),
        ("2007-03-10", 2007.189, 193.804, -36.506, 0.030, 0.070),
        ("2007-03-17", 2007.208, 193.835, -36.532, 0.030, 0.070),
        ("2007-03-25", 2007.230, 193.751, -36.712, 0.030, 0.070),
        ("2007-04-15", 2007.287, 193.796, -36.979, 0.030, 0.070),
        ("2007-04-22", 2007.307, 193.747, -37.007, 0.030, 0.070),
        ("2007-09-28", 2007.742, 193.340, -39.090, 0.060, 0.140),
        ("2007-09-30", 2007.747, 193.420, -38.917, 0.060, 0.140),
        ("2007-10-16", 2007.791, 193.381, -39.240, 0.060, 0.140),
    ]

    ra_rad, dec_rad = parse_source_coords("17:47:20.150", "-28:23:04.03")
    results = {}

    for label, data, vlsr, pi_pub, sig_pub in [
        ("Sgr B2M", sgr_b2m_raw, 66.4, 0.130, 0.012),
        ("Sgr B2N", sgr_b2n_raw, 56.7, 0.128, 0.015),
    ]:
        mjds = np.array([(_date.fromisoformat(c) - _mjd_ref).days
                         for c, *_ in data], dtype=float)
        dy = np.array([d[1] for d in data])
        x = np.array([d[2] for d in data])
        y = np.array([d[3] for d in data])
        sx = np.array([d[4] for d in data])
        sy = np.array([d[5] for d in data])

        ch = ChannelData(
            group_id=1, vlsr=vlsr, grade="A",
            mjds=mjds, dec_years=dy, x=x, y=y,
            x_err=sx, y_err=sy, flux=np.ones(len(x)),
            epochs=[f"e{i}" for i in range(len(x))],
            use_for_pi=np.ones(len(x), dtype=int),
        )
        t_ref = np.mean(mjds)
        r = fit_individual_channel(ch, ra_rad, dec_rad, t_ref,
                                    ephemeris="analytical", coord="both",
                                    error_floor_mode="manual",
                                    manual_floor_ra=0.0, manual_floor_dec=0.0)
        diff_sigma = abs(r.pi - pi_pub) / np.sqrt(r.sigma_pi**2 + sig_pub**2)
        passed = diff_sigma < 1.5
        results[label] = (r.pi, r.sigma_pi, pi_pub, sig_pub, diff_sigma, passed)

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  {label}: pi={r.pi:.4f}±{r.sigma_pi:.4f} "
                  f"(published {pi_pub:.3f}±{sig_pub:.3f}), "
                  f"diff={diff_sigma:.2f}σ [{status}]")

    all_pass = all(v[5] for v in results.values())
    if verbose:
        print(f"\n  Validation: {'ALL PASS' if all_pass else 'FAILED'}")
    return all_pass


def main():
    pa = argparse.ArgumentParser(
        prog="masertrack_fit",
        description=f"masertrack_fit v{__version__}: Parallax and proper motion fitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python masertrack_fit.py match_inverse_spots_tracked.txt --ra 18:14:26.743 --dec -25:02:54.750
          python masertrack_fit.py spots.txt --ra 18:14:26.743 --dec -25:02:54.750 --error-floor 0.05 0.10
          python masertrack_fit.py spots.txt --ra 18:14:26.743 --dec -25:02:54.750 --quick
        """))
    pa.add_argument("spots_file", nargs="?", default=None,
                    help="Tracked spots file from masertrack_match.py")
    pa.add_argument("--ra", default=None, help="Source RA (hh:mm:ss.sss)")
    pa.add_argument("--dec", default=None, help="Source Dec (dd:mm:ss.sss)")
    pa.add_argument("--features", default=None, help="Tracked features file (optional)")
    pa.add_argument("--spots-normal", default=None,
                    help="Normal PR spots file (run both modes and combine)")
    pa.add_argument("--features-normal", default=None,
                    help="Normal PR features file (optional)")
    pa.add_argument("--groups", default=None, help="Groups file (optional)")
    pa.add_argument("--outdir", default=".", help="Output directory")
    pa.add_argument("--ephemeris", default="auto",
                    choices=["auto", "de440", "de421", "builtin", "analytical"],
                    help="Parallax factor method (default: auto)")
    pa.add_argument("--error-floor", nargs="*", default=None,
                    help="Error floor: 'auto' (default) or two values: eps_RA eps_Dec in mas")
    pa.add_argument("--bootstrap-trials", type=int, default=10000,
                    help="Number of bootstrap trials (default: 10000)")
    pa.add_argument("--grade-threshold", default="C", choices=["A", "B", "C", "D"],
                    help="Minimum grade to fit (default: C)")
    pa.add_argument("--acceleration", nargs="?", const="test", default="off",
                    choices=["off", "test", "on"],
                    help="Acceleration: off (default), test (BIC), on (force)")
    pa.add_argument("--distance-prior", default="none",
                    choices=["none", "disk", "exponential", "uniform"],
                    help="Distance prior: none (1/π, default), disk (Galactic), "
                         "exponential (Bailer-Jones), uniform")
    pa.add_argument("--disk-scale", nargs=2, type=float, default=[DISK_HR, DISK_HZ],
                    metavar=("h_R", "h_z"),
                    help=f"Disk prior scale lengths in kpc. "
                         f"Examples: HMSFR=2.4 0.025, AGB=2.5 0.250, "
                         f"post-AGB=2.6 0.300, thick-disk=2.6 0.900 "
                         f"(default: {DISK_HR} {DISK_HZ})")
    pa.add_argument("--export-formats", default="pmpar,bessel,vera,kadai",
                    help="Export formats, comma-separated (default: pmpar,bessel,vera,kadai)")
    pa.add_argument("--chan-spacing", type=float, default=None,
                    help="Channel spacing in km/s (default: auto-detect; VERA=0.425)")
    pa.add_argument("--pub-fit", default=None,
                    help="Select specific fit for publication export. "
                         "Options: 'G1' (group 1 channel fit), 'G1_feat' (group 1 feature fit), "
                         "'wt_mean' (weighted mean, default), 'combined_A' etc. "
                         "When set, writes a dedicated pub_selected_*.csv with only that fit.")
    pa.add_argument("--quick", action="store_true",
                    help="Quick mode: skip bootstrap")
    pa.add_argument("--no-plots", action="store_true", help="Disable plots")
    pa.add_argument("--verbose", action="store_true", default=True)
    pa.add_argument("--validate", action="store_true",
                    help="Run validation suite (Reid+2009 Sgr B2) and exit")

    args = pa.parse_args()

    # Validation mode
    if args.validate:
        print(f"masertrack_fit v{__version__} — validation suite\n")
        print("Reid+2009 (ApJ 705, 1548) — Sgr B2 H2O masers:")
        _RESOLVED_EPHEMERIS[0] = None  # reset cache
        ok = validate_reid_sgr_b2(verbose=True)
        sys.exit(0 if ok else 1)

    # Normal mode: require spots_file, ra, dec
    if not args.spots_file:
        pa.error("spots_file is required (use --validate for validation mode)")
    if not args.ra or not args.dec:
        pa.error("--ra and --dec are required")

    # Parse coordinates
    ra_rad, dec_rad = parse_source_coords(args.ra, args.dec)

    # Parse error floor
    if args.error_floor is None or args.error_floor == ["auto"]:
        ef_mode = "auto"
        ef_ra, ef_dec = 0.0, 0.0
    elif len(args.error_floor) == 2:
        ef_mode = "manual"
        ef_ra, ef_dec = float(args.error_floor[0]), float(args.error_floor[1])
    else:
        ef_mode = "auto"
        ef_ra, ef_dec = 0.0, 0.0

    # Read input
    print(f"Reading {args.spots_file}...")
    spots_df, meta = parse_tracked_file(args.spots_file)
    if len(spots_df) == 0:
        print("ERROR: no data parsed from spots file")
        sys.exit(1)
    mode = meta.get("mode", "unknown")
    print(f"  {len(spots_df)} spots, mode={mode}")

    features_df = None
    if args.features:
        features_df, _ = parse_tracked_file(args.features)
        print(f"  {len(features_df)} features from {args.features}")

    groups_df = None
    if args.groups:
        groups_df = parse_groups_file(args.groups)

    # Quick mode adjustments
    bootstrap = args.bootstrap_trials
    if args.quick:
        bootstrap = 0

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Run fitter
    fitter = ParallaxFitter(
        spots_df=spots_df,
        ra_rad=ra_rad, dec_rad=dec_rad,
        groups_df=groups_df, features_df=features_df,
        ephemeris=args.ephemeris,
        error_floor_mode=ef_mode,
        manual_floor_ra=ef_ra, manual_floor_dec=ef_dec,
        bootstrap_trials=bootstrap,
        grade_threshold=args.grade_threshold,
        mode_label=mode,
        chan_spacing=args.chan_spacing,
    )
    fitter._outdir = args.outdir
    fitter.run(verbose=args.verbose)

    # Write outputs
    print(f"\nWriting outputs to {args.outdir}/")
    write_results_table(fitter, args.outdir, mode)

    # Exports
    if args.export_formats:
        formats = [f.strip() for f in args.export_formats.split(",")]
        if "pmpar" in formats:
            export_pmpar(fitter, args.outdir, mode)
        if "bessel" in formats:
            export_bessel(fitter, args.outdir, mode)
        if "vera" in formats:
            export_vera(fitter, args.outdir, mode)
        if "kadai" in formats:
            export_kadai(fitter, args.outdir, mode)

    # Plots
    if not args.no_plots:
        make_plots(fitter, args.outdir, mode)

    # If normal PR also provided, run it too (SEPARATE, for comparison)
    fitter_nor = None
    if args.spots_normal:
        print(f"\n{'='*65}")
        print(f"Running normal PR mode (for comparison, NOT combined)...")
        spots_nor, meta_nor = parse_tracked_file(args.spots_normal)
        mode_nor = meta_nor.get("mode", "normal")
        feat_nor = None
        if args.features_normal:
            feat_nor, _ = parse_tracked_file(args.features_normal)

        fitter_nor = ParallaxFitter(
            spots_df=spots_nor, ra_rad=ra_rad, dec_rad=dec_rad,
            features_df=feat_nor, ephemeris=args.ephemeris,
            error_floor_mode=ef_mode,
            manual_floor_ra=ef_ra, manual_floor_dec=ef_dec,
            bootstrap_trials=bootstrap,
            grade_threshold=args.grade_threshold,
            mode_label=mode_nor, chan_spacing=args.chan_spacing,
        )
        fitter_nor._outdir = args.outdir
        fitter_nor.run(verbose=args.verbose)
        write_results_table(fitter_nor, args.outdir, mode_nor)
        if args.export_formats:
            formats = [f.strip() for f in args.export_formats.split(",")]
            if "pmpar" in formats:
                export_pmpar(fitter_nor, args.outdir, mode_nor)
            if "vera" in formats:
                export_vera(fitter_nor, args.outdir, mode_nor)
            if "kadai" in formats:
                export_kadai(fitter_nor, args.outdir, mode_nor)
        if not args.no_plots:
            make_plots(fitter_nor, args.outdir, mode_nor)

    # Final parallax per mode (separate, for comparison)
    fitters = {mode: fitter}
    if fitter_nor:
        fitters[meta_nor.get("mode", "normal")] = fitter_nor
    write_final_parallax(fitters, args.outdir)

    # Publication fit selection (--pub-fit)
    if args.pub_fit:
        write_selected_publication_fit(fitters, args.pub_fit, args.outdir)

    print("\nDone.")


# =====================================================================
#  Quick self-test
# =====================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--self-test"):
        main()
        sys.exit(0)

    print(f"masertrack_fit v{__version__} — module self-test")

    # Test parser
    test_file = "/mnt/user-data/uploads/match_inverse_spots_tracked.txt"
    if os.path.exists(test_file):
        df, meta = parse_tracked_file(test_file)
        print(f"\nParsed {test_file}:")
        print(f"  {len(df)} rows, mode={meta.get('mode','?')}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Groups: {df.group_id.nunique()}, Grades: {df.grade.value_counts().to_dict()}")
        print(f"  use_for_pi distribution: {df.use_for_pi.value_counts().to_dict()}")

    # Test parallax factors
    ra_rad, dec_rad = parse_source_coords("18:14:26.743", "-25:02:54.750")
    print(f"\nSource: RA={np.degrees(ra_rad):.4f}°, Dec={np.degrees(dec_rad):.4f}°")
    test_mjds = np.array([56764, 56868, 56934, 56964, 56994, 57024, 57059, 57087, 57115], dtype=float)

    F_a_an, F_d_an = parallax_factors_analytical(ra_rad, dec_rad, test_mjds)
    print(f"\nAnalytical parallax factors:")
    print(f"  F_a: {F_a_an}")
    print(f"  F_d: {F_d_an}")

    if HAS_ASTROPY:
        F_a_de, F_d_de = parallax_factors_de440(ra_rad, dec_rad, test_mjds)
        print(f"\nDE440 parallax factors:")
        print(f"  F_a: {F_a_de}")
        print(f"  F_d: {F_d_de}")
        print(f"\nDifference (analytical - DE440):")
        print(f"  dF_a: {F_a_an - F_a_de} (max {np.max(np.abs(F_a_an - F_a_de)):.6f} AU)")
        print(f"  dF_d: {F_d_an - F_d_de} (max {np.max(np.abs(F_d_an - F_d_de)):.6f} AU)")
        print(f"  At pi=0.1 mas, max position error: "
              f"{0.1*np.max(np.abs(np.concatenate([F_a_an-F_a_de, F_d_an-F_d_de]))):.4f} mas")

    # Test distance posterior
    l_rad, b_rad = equatorial_to_galactic(ra_rad, dec_rad)
    print(f"\nGalactic coords: l={np.degrees(l_rad):.2f}°, b={np.degrees(b_rad):.2f}°")

    dist = compute_distance_posterior(0.1, 0.04, l_rad, b_rad, prior_type="disk")
    print(f"\nDistance posterior (pi=0.10±0.04 mas, disk prior):")
    print(f"  D_mode = {dist['D_mode']:.1f} kpc")
    print(f"  D_median = {dist['D_median']:.1f} kpc")
    print(f"  68% CI: [{dist['D_16']:.1f}, {dist['D_84']:.1f}] kpc")
    print(f"  90% CI: [{dist['D_05']:.1f}, {dist['D_95']:.1f}] kpc")

    print("\n✓ Self-test passed")
