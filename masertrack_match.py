#!/usr/bin/env python3
"""
masertrack_match.py - Part 2 of the masertrack pipeline
=========================================================
Version 1.1

Cross-epoch matching of maser features and channel-level tracking
for parallax (pi) and proper motion (mu) fitting.

Input:  Feature/spot catalogs from masertrack_identify.py + epoch table
Output: Matched group catalog, tracked spots/features, diagnostic plots

See README.md for full documentation and CODE_BREAKDOWN.md for internals.

Changelog:
  v1.1  Code cleanup, unified commenting style, tutorial.
  v1.0  First release. Growing-chain matching with velocity drift
        tracking, normal/inverse PR separation, quasar-based alignment,
        reference spot correction, grading system (A-F), trajectory
        smoothness checking, corrections file workflow, multi-panel
        Plotly HTML.

Credits: Gabor Orosz (design), Claude (code, 2026)
License: BSD-3-Clause
Repository: https://github.com/chickin/masertrack
"""
# =====================================================================
#  IMPORTS -- graceful failure with install instructions
# =====================================================================
from __future__ import annotations
import argparse, os, sys, textwrap, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Note: matplotlib not found; no PNG plots.  pip3 install matplotlib")
try:
    from adjustText import adjust_text
    HAS_ADJTEXT = True
except ImportError:
    HAS_ADJTEXT = False
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

CMAP_NAME = "rainbow"
# =====================================================================
#  RAINBOW COLORMAP -- red=redshifted, blue/violet=blueshifted
# =====================================================================
PLOTLY_RAINBOW = [[0,"rgb(128,0,255)"],[0.15,"rgb(0,0,255)"],[0.3,"rgb(0,180,255)"],
    [0.45,"rgb(0,220,0)"],[0.6,"rgb(255,255,0)"],[0.75,"rgb(255,165,0)"],
    [0.9,"rgb(255,50,0)"],[1.0,"rgb(200,0,0)"]]

# =====================================================================
#  DEFAULT PARAMETERS -- all documented, all overridable via CLI
# =====================================================================
# All spatial thresholds scale with beam geometric mean.
# All velocity thresholds scale with channel spacing.
DEFAULTS = {
    # Match radius = this factor x beam geometric mean (generous to
    # allow for proper motion + position uncertainty between epochs)
    "match_radius_factor": 3.0,
    # Velocity tolerance in channel spacings (catches V drift of
    # ~1-3 km/s/yr common in H2O masers over ~1 yr baseline)
    "vel_tolerance_channels": 4,
    # Max proper motion in mas/yr (upper limit for Galactic masers:
    # solar motion + rotation ~2-4 mas/yr, internal ~1-5 mas/yr)
    "max_pm": 5.0,
    # Min epochs for a group to be displayed (3 = minimum for pi fitting)
    "min_epochs": 3,
    # Trajectory outlier threshold (MAD-based robust sigma)
    "outlier_sigma": 3.0,
}

# =====================================================================
#  FILE I/O
# =====================================================================
def read_mt(path):
    """Read a masertrack_identify output file (features or spots).

    Handles the 'sidelobe_reason' column which can contain spaces
    or be empty, causing split() to produce the wrong number of tokens.

    Returns (DataFrame, beam_string).
    """
    hdr, data, beam = None, [], None
    for ln in open(path):
        ls = ln.rstrip()
        if ls.strip().startswith("#"):
            p = ls.lstrip("# ").split()
            if any(c in p for c in ["vlsr","vlsr_peak","feature_id"]):
                hdr = p
            if "Beam" in ls and "mas" in ls:
                beam = ls
            continue
        if ls.strip():
            data.append(ls)
    if not hdr or not data:
        return pd.DataFrame(), beam

    nc = len(hdr)
    si = hdr.index("sidelobe_reason") if "sidelobe_reason" in hdr else -1
    rows = []
    for ln in data:
        p = ln.split()
        # Handle empty or multi-word sidelobe_reason column
        if si >= 0:
            if len(p) == nc - 1:
                p.insert(si, "")
            elif len(p) > nc:
                extra = len(p) - nc
                merged = " ".join(p[si:si+extra+1])
                p = p[:si] + [merged] + p[si+extra+1:]
        while len(p) < nc:
            p.append("")
        rows.append(p[:nc])

    df = pd.DataFrame(rows, columns=hdr)
    for c in df.columns:
        if c in ["sidelobe_reason", "sidelobe"]:
            continue
        try:
            df[c] = pd.to_numeric(df[c])
        except (ValueError, TypeError):
            pass
    return df, beam


def parse_beam(bl):
    """Extract beam major, minor (mas) and channel spacing (km/s) from header string."""
    if not bl:
        return None, None, None
    m = re.search(r'(\d+\.\d+)\s*x\s*(\d+\.\d+)\s*mas.*?dv\s*=\s*(\d+\.\d+)', bl)
    return (float(m.group(1)),float(m.group(2)),float(m.group(3))) if m else (None,None,None)

def write_table(df, path, header=""):
    int_cols = ["group_id","n_epochs","n_channels","mjd","n_stations",
                "use_for_pi","n_feat_channels"]
    for c in int_cols:
        if c in df.columns:
            try: df[c] = df[c].astype(int)
            except (ValueError, TypeError): pass
    lines = [f"# {header}", f"# masertrack_match v{__version__}"]
    cw = {}
    for c in df.columns:
        fm = []
        for v in df[c]:
            if c in int_cols and not isinstance(v,str): fm.append(f"{int(v):8d}")
            elif isinstance(v,(float,np.floating)): fm.append(f"{v:12.4f}")
            else: fm.append(str(v))
        cw[c] = max(len(c), max((len(s) for s in fm), default=1)) + 1
    lines.append("# " + "".join(f"{c:>{cw[c]}s}" for c in df.columns))
    for _, row in df.iterrows():
        p = []
        for c in df.columns:
            w,v = cw[c], row[c]
            if c in int_cols and not isinstance(v,str): p.append(f"{int(v):{w}d}")
            elif isinstance(v,(float,np.floating)):
                p.append(f"{v:{w}.4f}" if not np.isnan(v) else f"{'nan':>{w}s}")
            else: p.append(f"{str(v):>{w}s}")
        lines.append("  " + "".join(p))
    open(path,"w").write("\n".join(lines)+"\n")

# =====================================================================
#  EPOCH + GRADING + CORRECTIONS
# =====================================================================
@dataclass
class EpochInfo:
    code:str; date:str; mjd:int; dec_year:float; n_stations:int
    inv_shift_ra:float=0.0; inv_shift_dec:float=0.0
    inv_qso_ra_s:float=0.0; inv_qso_dec_s:float=0.0
    flag:str=""; use_for_pi:int=1

def parse_epoch_table(path):
    epochs = []
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        p = ln.split()
        if len(p) < 5: continue
        try:
            e = EpochInfo(code=p[0],date=p[1],mjd=int(p[2]),dec_year=float(p[3]),
                         n_stations=int(p[4]),
                         inv_shift_ra=float(p[5]) if len(p)>5 else 0,
                         inv_shift_dec=float(p[6]) if len(p)>6 else 0,
                         inv_qso_ra_s=float(p[7]) if len(p)>7 else 0,
                         inv_qso_dec_s=float(p[8]) if len(p)>8 else 0)
            if e.n_stations < 4: e.flag = f"{e.n_stations}sta"
            epochs.append(e)
        except (ValueError, IndexError): continue
    return epochs

def grade_group(n_ep, n_unflagged, n_ch):
    """Grade a group. n_unflagged = total epochs minus flagged ones."""
    frac = n_ep / max(n_unflagged,1)
    if n_ep >= n_unflagged: return "A"
    if frac >= 2.0/3.0 and n_ep >= 4: return "B"
    if n_ep >= 4: return "C"
    if n_ep >= 3: return "D"
    if n_ep >= 2: return "E"
    return "F"

def parse_corrections(path):
    """Parse a corrections file for manual overrides.

    Format (one per line, # = comment):
      EPOCH  r14307a  use_for_pi=0
      GROUP  G6  r15037b  EXCLUDE
      GROUP  G22 r14307a  EXCLUDE
    """
    if not path or not Path(path).exists():
        return {"epochs":{},"groups":{}}
    corr = {"epochs":{},"groups":{}}
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        p = ln.split()
        if len(p) < 3: continue
        if p[0].upper() == "EPOCH":
            ec = p[1]
            opts = {}
            for tok in p[2:]:
                if "=" in tok:
                    k,v = tok.split("=",1)
                    opts[k] = v
            corr["epochs"][ec] = opts
        elif p[0].upper() == "GROUP":
            gid_str = p[1].replace("G","").replace("g","")
            try: gid = int(gid_str)
            except ValueError: continue
            if gid not in corr["groups"]: corr["groups"][gid] = []
            if len(p) >= 4 and p[3].upper() == "EXCLUDE":
                corr["groups"][gid].append(("exclude",p[2]))
    return corr

# =====================================================================
#  REFERENCE MASER CORRECTION
# =====================================================================
def compute_ref_corrections(spot_files, epochs, chan_spacing):
    """Compute reference maser position corrections for inverse PR."""
    if not spot_files: return {}
    ref_info = {}
    for ec, sdf in sorted(spot_files.items()):
        r = np.hypot(sdf["x"], sdf["y"])
        idx = r.idxmin()
        ref_info[ec] = {"vlsr":float(sdf.loc[idx,"vlsr"]),
                        "x":float(sdf.loc[idx,"x"]),"y":float(sdf.loc[idx,"y"])}
    ec_list = sorted(ref_info.keys(),
                     key=lambda c: next((e.dec_year for e in epochs if e.code==c), 0))
    if not ec_list: return {}
    anchor_ec = ec_list[0]; anchor_v = ref_info[anchor_ec]["vlsr"]
    corrections = {}
    for ec in ec_list:
        if ec == anchor_ec:
            corrections[ec] = (0.0, 0.0, anchor_v, anchor_v); continue
        sdf = spot_files[ec]
        dv = np.abs(sdf["vlsr"] - anchor_v)
        idx = dv.idxmin()
        if dv[idx] > chan_spacing * 3:
            corrections[ec] = (0.0, 0.0, ref_info[ec]["vlsr"], float(sdf.loc[idx,"vlsr"]))
            continue
        cx = float(sdf.loc[idx, "x"]); cy = float(sdf.loc[idx, "y"])
        corrections[ec] = (-cx, -cy, ref_info[ec]["vlsr"], float(sdf.loc[idx,"vlsr"]))
    return corrections

# =====================================================================
#  SINGLE-MODE MATCHER
# =====================================================================
class SingleModeMatch:
    def __init__(self, mode, epochs, beam_geo, chan_spacing, mrf, vtc, max_pm):
        self.mode=mode; self.epochs=epochs; self.beam_geo=beam_geo
        self.chan_spacing=chan_spacing; self.mrf=mrf; self.vtc=vtc
        self.max_pm=max_pm
        # n_unflagged = epochs without flags (for grading)
        self.n_unflagged = sum(1 for e in epochs if not e.flag)
        self.n_total = len(epochs)
        self.features:Dict[str,pd.DataFrame]={}
        self.spots:Dict[str,pd.DataFrame]={}
        self.groups=[]

    def add_epoch(self, ec, feat_df, spot_df=None):
        self.features[ec] = feat_df
        if spot_df is not None: self.spots[ec] = spot_df

    def _ep_time(self, code):
        for e in self.epochs:
            if e.code==code: return e.dec_year
        return 0.0

    def _ep_flagged(self, code):
        for e in self.epochs:
            if e.code==code: return bool(e.flag)
        return False

    def match(self):
        """Growing-chain matching with velocity drift tracking.

        Key improvement in v0.6: velocity is interpolated/extrapolated
        the same way as position, so features that drift in velocity
        (common for H2O masers) are still matched correctly.
        """
        match_r = self.mrf * self.beam_geo
        vel_tol = self.vtc * self.chan_spacing
        ec_list = sorted(self.features.keys(), key=self._ep_time)
        if not ec_list: return []
        vcol = "vlsr_peak" if "vlsr_peak" in self.features[ec_list[0]].columns else "vlsr"
        seed = max(ec_list, key=lambda c: len(self.features[c]))
        seed_df = self.features[seed]; seed_t = self._ep_time(seed)
        others = sorted([c for c in ec_list if c!=seed],
                       key=lambda c: abs(self._ep_time(c)-seed_t))
        used = {c:set() for c in ec_list}; groups = []
        for si, srow in seed_df.iterrows():
            sv,sx,sy = srow[vcol],srow["x"],srow["y"]
            matches = {seed:si}; used[seed].add(si)
            ts,xs,ys,vs = [seed_t],[sx],[sy],[sv]
            for oc in others:
                odf = self.features[oc]; ot = self._ep_time(oc)
                exp_r = match_r + self.max_pm * abs(ot-seed_t)
                # Interpolate/extrapolate expected position
                if len(ts)>=2:
                    order = np.argsort(ts)
                    px = float(np.interp(ot,np.array(ts)[order],np.array(xs)[order]))
                    py = float(np.interp(ot,np.array(ts)[order],np.array(ys)[order]))
                else: px,py = sx,sy
                # v0.6: Interpolate/extrapolate expected velocity
                if len(ts)>=2:
                    pv = float(np.interp(ot,np.array(ts)[order],np.array(vs)[order]))
                else:
                    pv = sv
                ovcol = "vlsr_peak" if "vlsr_peak" in odf.columns else "vlsr"
                best_i,best_d = None, exp_r+1
                for oi, orow in odf.iterrows():
                    if oi in used[oc]: continue
                    if abs(orow[ovcol]-pv)>vel_tol: continue
                    d = np.hypot(orow["x"]-px,orow["y"]-py)
                    if d<exp_r and d<best_d: best_d,best_i = d,oi
                if best_i is not None:
                    matches[oc]=best_i; used[oc].add(best_i)
                    ts.append(ot)
                    xs.append(float(odf.loc[best_i,"x"]))
                    ys.append(float(odf.loc[best_i,"y"]))
                    vs.append(float(odf.loc[best_i,ovcol]))
            n_ep=len(matches); n_ch=self._count_channels(matches)
            # Count only unflagged epochs present for grading
            n_unflagged_present = sum(1 for ec in matches if not self._ep_flagged(ec))
            gr = grade_group(n_unflagged_present, self.n_unflagged, n_ch)
            all_x=[self.features[c].loc[i,"x"] for c,i in matches.items()]
            all_y=[self.features[c].loc[i,"y"] for c,i in matches.items()]
            pattern="".join(str(ei+1) if ep.code in matches else "*"
                          for ei,ep in enumerate(self.epochs))
            # Position scatter
            scatter_ra = np.std(all_x) if len(all_x)>2 else 0
            scatter_dec = np.std(all_y) if len(all_y)>2 else 0
            dt_range = max(ts)-min(ts) if len(ts)>1 else 0
            expected_scatter = self.max_pm * dt_range / 2
            scatter_flag = ""
            if scatter_ra > expected_scatter * 2 or scatter_dec > expected_scatter * 2:
                scatter_flag = "large_scatter"
            groups.append({"group_id":len(groups)+1,"vlsr_peak":sv,
                "n_epochs":n_ep,"n_channels":n_ch,"grade":gr,
                "x_mean":np.mean(all_x),"y_mean":np.mean(all_y),
                "scatter_ra":scatter_ra,"scatter_dec":scatter_dec,
                "scatter_flag":scatter_flag,
                "vel_drift":0.0,"vel_range":0.0,
                "pattern":pattern,"matches":matches,
                "flagged_epochs":[]})
        self.groups = groups
        # v0.6: position trajectory smoothness check
        self._check_trajectories()
        # v0.6: compute velocity drift per group
        self._compute_vel_drift()
        return groups

    def _compute_vel_drift(self):
        """Compute velocity drift rate (km/s/yr) per group via linear fit."""
        for g in self.groups:
            if g["n_epochs"] < 3:
                g["vel_drift"] = 0.0
                g["vel_range"] = 0.0
                continue
            ts, vs = [], []
            vcol_g = None
            for ec, fidx in g["matches"].items():
                fdf = self.features[ec]
                if vcol_g is None:
                    vcol_g = "vlsr_peak" if "vlsr_peak" in fdf.columns else "vlsr"
                ts.append(self._ep_time(ec))
                vs.append(float(fdf.loc[fidx, vcol_g]))
            ts, vs = np.array(ts), np.array(vs)
            g["vel_range"] = float(vs.max() - vs.min())
            if len(ts) >= 2:
                c = np.polyfit(ts - ts[0], vs, 1)
                g["vel_drift"] = float(c[0])  # km/s/yr
            else:
                g["vel_drift"] = 0.0

    def _check_trajectories(self):
        """Fit linear motion to each group and flag outlier epochs."""
        for g in self.groups:
            if g["n_epochs"] < 4: continue
            ts,xs,ys,ecs = [],[],[],[]
            for ec,fidx in g["matches"].items():
                t = self._ep_time(ec)
                fdf = self.features[ec]
                ts.append(t); xs.append(fdf.loc[fidx,"x"])
                ys.append(fdf.loc[fidx,"y"]); ecs.append(ec)
            ts,xs,ys = np.array(ts),np.array(xs),np.array(ys)
            # Fit linear motion
            t0 = ts[0]
            if len(ts) < 3: continue
            cx = np.polyfit(ts-t0, xs, 1)
            cy = np.polyfit(ts-t0, ys, 1)
            res_x = np.abs(xs - np.polyval(cx, ts-t0))
            res_y = np.abs(ys - np.polyval(cy, ts-t0))
            sig_x = np.median(res_x)*1.4826 if np.median(res_x)>0 else 0.1
            sig_y = np.median(res_y)*1.4826 if np.median(res_y)>0 else 0.1
            sig = DEFAULTS["outlier_sigma"]
            flagged = []
            for i,ec in enumerate(ecs):
                if res_x[i] > sig*sig_x or res_y[i] > sig*sig_y:
                    flagged.append(ec)
            g["flagged_epochs"] = flagged

    def _count_channels(self, matches):
        epoch_chs = {}
        for ec,fidx in matches.items():
            if ec not in self.spots: continue
            sdf=self.spots[ec]; fdf=self.features[ec]
            if "feature_id" in sdf.columns and "feature_id" in fdf.columns:
                fid=fdf.loc[fidx,"feature_id"]; mask=sdf["feature_id"]==fid
            else:
                fx,fy=fdf.loc[fidx,"x"],fdf.loc[fidx,"y"]
                mask=np.hypot(sdf["x"]-fx,sdf["y"]-fy)<self.beam_geo*2
            if mask.sum()==0: continue
            chs=set(np.round(sdf.loc[mask,"vlsr"].values/self.chan_spacing).astype(int))
            epoch_chs[ec]=chs
        if len(epoch_chs)<2: return 0
        all_ch=set()
        for c in epoch_chs.values(): all_ch.update(c)
        return sum(1 for ch in all_ch
                   if sum(1 for c in epoch_chs.values() if ch in c)>=2)

    def build_tracked_spots(self):
        rows=[]
        for g in self.groups:
            if g["n_epochs"]<2: continue
            for ec,fidx in g["matches"].items():
                if ec not in self.spots: continue
                sdf=self.spots[ec]; fdf=self.features[ec]
                ep=next((e for e in self.epochs if e.code==ec),None)
                if "feature_id" in sdf.columns and "feature_id" in fdf.columns:
                    fid=fdf.loc[fidx,"feature_id"]; mask=sdf["feature_id"]==fid
                else:
                    fx,fy=fdf.loc[fidx,"x"],fdf.loc[fidx,"y"]
                    mask=np.hypot(sdf["x"]-fx,sdf["y"]-fy)<self.beam_geo*2
                ep_flagged = ec in g.get("flagged_epochs",[])
                pi_flag = ep.use_for_pi if ep else 1
                if ep_flagged: pi_flag = 0
                for si,sr in sdf[mask].iterrows():
                    xe = float(sr.get("x_err",0.0))
                    ye = float(sr.get("y_err",0.0))
                    # Error floor: beam_geo/(2*SNR), minimum 0.001 mas
                    flux = float(sr.get("I",1.0))
                    rms = float(sr.get("rms",0.0))
                    if rms > 0 and flux > 0:
                        snr = flux / rms
                        floor = self.beam_geo / (2 * snr)
                    else:
                        floor = 0.01  # conservative default
                    floor = max(floor, 0.001)
                    xe = max(xe, floor)
                    ye = max(ye, floor)
                    rows.append({"group_id":g["group_id"],"grade":g["grade"],
                        "epoch":ec,"mjd":ep.mjd if ep else 0,
                        "dec_year":ep.dec_year if ep else 0,
                        "n_stations":ep.n_stations if ep else 0,
                        "flag":ep.flag if ep else "",
                        "use_for_pi":pi_flag,
                        "vlsr":sr["vlsr"],"x":sr["x"],"y":sr["y"],
                        "x_err":xe,"y_err":ye,
                        "flux":flux})
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def build_tracked_features(self):
        rows=[]
        for g in self.groups:
            for ec,fidx in g["matches"].items():
                fdf=self.features[ec]
                ep=next((e for e in self.epochs if e.code==ec),None)
                row=fdf.loc[fidx]
                vcol="vlsr_peak" if "vlsr_peak" in fdf.columns else "vlsr"
                ep_flagged = ec in g.get("flagged_epochs",[])
                pi_flag = ep.use_for_pi if ep else 1
                if ep_flagged: pi_flag = 0
                # Use wmean error (best centroid uncertainty estimate)
                x_err = float(row.get("x_err_wmean", row.get("x_err_formal", 0)))
                y_err = float(row.get("y_err_wmean", row.get("y_err_formal", 0)))
                # Error floor: beam_geo/(2*SNR), min 0.001 mas
                flux = float(row.get("I_peak", row.get("I", 1.0)))
                rms = float(row.get("rms", 0.0))
                if rms > 0 and flux > 0:
                    floor = self.beam_geo / (2 * flux / rms)
                else:
                    floor = 0.01
                floor = max(floor, 0.001)
                x_err = max(x_err, floor)
                y_err = max(y_err, floor)
                rows.append({"group_id":g["group_id"],"grade":g["grade"],
                    "feature_id":int(row.get("feature_id",0)),
                    "epoch":ec,"mjd":ep.mjd if ep else 0,
                    "dec_year":ep.dec_year if ep else 0,
                    "n_stations":ep.n_stations if ep else 0,
                    "flag":ep.flag if ep else "",
                    "use_for_pi":pi_flag,
                    "vlsr":row[vcol],"x":row["x"],"y":row["y"],
                    "x_err":x_err,"y_err":y_err,
                    "flux":flux,
                    "n_feat_channels":row.get("n_channels",1)})
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def apply_corrections(self, corr):
        """Apply corrections file overrides to matched groups."""
        if not corr or not corr.get("groups"): return
        n_applied = 0
        for gid, actions in corr["groups"].items():
            # Find group by ID
            g = next((g for g in self.groups if g["group_id"]==gid), None)
            if g is None: continue
            for action, ec in actions:
                if action == "exclude" and ec in g["matches"]:
                    del g["matches"][ec]
                    n_applied += 1
            # Recalculate stats
            g["n_epochs"] = len(g["matches"])
            if g["n_epochs"] == 0: continue
            all_x = [self.features[c].loc[i,"x"] for c,i in g["matches"].items()]
            all_y = [self.features[c].loc[i,"y"] for c,i in g["matches"].items()]
            g["x_mean"] = np.mean(all_x); g["y_mean"] = np.mean(all_y)
            g["scatter_ra"] = np.std(all_x) if len(all_x)>2 else 0
            g["scatter_dec"] = np.std(all_y) if len(all_y)>2 else 0
            n_unflagged_present = sum(1 for ec in g["matches"] if not self._ep_flagged(ec))
            g["grade"] = grade_group(n_unflagged_present, self.n_unflagged,
                                     self._count_channels(g["matches"]))
            g["n_channels"] = self._count_channels(g["matches"])
            g["pattern"] = "".join(str(ei+1) if ep.code in g["matches"] else "*"
                                  for ei,ep in enumerate(self.epochs))
        if n_applied: print(f"  Applied {n_applied} group corrections ({self.mode})")

# =====================================================================
#  MAIN PIPELINE
# =====================================================================
class MaserTrackMatch:
    def __init__(self, epoch_table, input_dir="./input", outdir=".",
                 beam_major=None, beam_minor=None, chan_spacing=None,
                 match_radius_factor=None, vel_tolerance=None,
                 max_pm=None, min_epochs=None,
                 corrections_file=None,
                 save_plots=True, show_plots=False):
        self.epoch_table_path=Path(epoch_table)
        self.input_dir=Path(input_dir)
        self.outdir=Path(outdir); self.outdir.mkdir(parents=True,exist_ok=True)
        self._user_bmaj=beam_major; self._user_bmin=beam_minor; self._user_dv=chan_spacing
        self.mrf=match_radius_factor or DEFAULTS["match_radius_factor"]
        self.vtc=vel_tolerance or DEFAULTS["vel_tolerance_channels"]
        self.max_pm=max_pm or DEFAULTS["max_pm"]
        self.min_epochs=min_epochs or DEFAULTS["min_epochs"]
        self.corrections_file=corrections_file
        self.corrections=None
        self.save_plots=save_plots and HAS_MPL
        self.show_plots=show_plots and HAS_MPL
        self.epochs=[]; self.beam_geo=None; self.chan_spacing=None
        self.matchers:Dict[str,SingleModeMatch]={}
        self._prefix=str(self.outdir/"match")
        self._files_by_mode={}

    def _get_epoch(self,code):
        for e in self.epochs:
            if e.code==code: return e
        return None

    def step1_parse(self):
        print(f"\n{'='*60}\n  masertrack_match v{__version__}\n{'='*60}")
        self.epochs=parse_epoch_table(self.epoch_table_path)
        self._flag_outliers()
        # Apply epoch corrections from file
        if self.corrections_file:
            self.corrections = parse_corrections(self.corrections_file)
            for ec, opts in self.corrections.get("epochs",{}).items():
                ep = self._get_epoch(ec)
                if ep and "use_for_pi" in opts:
                    ep.use_for_pi = int(opts["use_for_pi"])
                    if not ep.use_for_pi and "no_pi" not in ep.flag:
                        ep.flag = (ep.flag+",corr_no_pi" if ep.flag else "corr_no_pi")
            print(f"  Corrections: {self.corrections_file}")

        print(f"\nStep 1 - {len(self.epochs)} epochs")
        n_flagged = sum(1 for e in self.epochs if e.flag)
        print(f"  ({n_flagged} flagged, {len(self.epochs)-n_flagged} unflagged for grading)")
        for e in self.epochs:
            fl=[e.flag] if e.flag else []
            if not e.use_for_pi: fl.append("no_pi")
            fs=f" [{','.join(fl)}]" if fl else ""
            print(f"  {e.code} {e.date} N={e.n_stations}{fs}")
        search=self.input_dir if self.input_dir.exists() else Path(".")
        self._files_by_mode={"normal":{},"inverse":{}}
        for ep in self.epochs:
            for mode in ["normal","inverse"]:
                ff=search/f"{ep.code}_{mode}_features.txt"
                sf=search/f"{ep.code}_{mode}_spots.txt"
                feat=spot=None
                if ff.exists():
                    feat,bl=read_mt(ff)
                    if len(feat)==0: feat=None
                    elif self.beam_geo is None:
                        bj,bn,dv=parse_beam(bl)
                        if bj: self._user_bmaj=self._user_bmaj or bj
                        if bn: self._user_bmin=self._user_bmin or bn
                        if dv: self._user_dv=self._user_dv or dv
                if sf.exists():
                    spot,_=read_mt(sf)
                    if len(spot)==0: spot=None
                if feat is not None or spot is not None:
                    self._files_by_mode[mode][ep.code]=(feat,spot)
        bj=self._user_bmaj or 2.1; bn=self._user_bmin or 0.75
        self.chan_spacing=self._user_dv or 0.42
        self.beam_geo=float(np.sqrt(bj*bn))
        for mode in ["normal","inverse"]:
            n=len(self._files_by_mode[mode])
            if n==0: continue
            print(f"\n  {mode.upper()}: {n} epochs")
            for ec in sorted(self._files_by_mode[mode]):
                ft,sp=self._files_by_mode[mode][ec]
                nf=len(ft) if ft is not None else 0
                ns=len(sp) if sp is not None else 0
                print(f"    {ec}: {nf} feat, {ns} spots")
        print(f"\n  Beam geo: {self.beam_geo:.2f} mas, dv: {self.chan_spacing:.3f} km/s")
        print(f"  Vel tolerance: {self.vtc} channels = {self.vtc*self.chan_spacing:.2f} km/s")

    def _flag_outliers(self):
        shifts=[(e.inv_shift_ra,e.inv_shift_dec,e.dec_year) for e in self.epochs]
        if len(shifts)<4: return
        t=np.array([s[2] for s in shifts]); ra=np.array([s[0] for s in shifts])
        dec=np.array([s[1] for s in shifts])
        c_ra=np.polyfit(t-t[0],ra,1); c_dec=np.polyfit(t-t[0],dec,1)
        res_ra=np.abs(ra-np.polyval(c_ra,t-t[0]))
        res_dec=np.abs(dec-np.polyval(c_dec,t-t[0]))
        sig_ra=np.median(res_ra)*1.4826; sig_dec=np.median(res_dec)*1.4826
        sig=DEFAULTS["outlier_sigma"]
        for i,e in enumerate(self.epochs):
            if (sig_ra>0 and res_ra[i]>sig*sig_ra) or \
               (sig_dec>0 and res_dec[i]>sig*sig_dec):
                if "outlier" not in e.flag:
                    e.flag=(e.flag+",outlier" if e.flag else "outlier")
                e.use_for_pi=0

    def step2_align(self):
        print(f"\nStep 2 - Alignment")

        # First pass: load normal PR (no alignment needed)
        if self._files_by_mode.get("normal"):
            matcher_n = SingleModeMatch("normal",self.epochs,self.beam_geo,
                self.chan_spacing,self.mrf,self.vtc,self.max_pm)
            for ec,(feat,spot) in self._files_by_mode["normal"].items():
                if feat is not None: matcher_n.add_epoch(ec,feat,spot)
                elif spot is not None:
                    matcher_n.add_epoch(ec,self._spots_to_features(spot),spot)
            print(f"  NORMAL: {len(self._files_by_mode['normal'])} epochs (no shift)")
            self.matchers["normal"] = matcher_n

        # Second pass: inverse PR with alignment
        if self._files_by_mode.get("inverse"):
            matcher_i = SingleModeMatch("inverse",self.epochs,self.beam_geo,
                self.chan_spacing,self.mrf,self.vtc,self.max_pm)
            files = self._files_by_mode["inverse"]

            # Reference spot corrections
            raw_spots = {ec:sp for ec,(ft,sp) in files.items() if sp is not None}
            ref_corr = compute_ref_corrections(raw_spots, self.epochs, self.chan_spacing)
            if ref_corr:
                print(f"  Reference spot corrections:")
                for ec,(cx,cy,rv,mv) in sorted(ref_corr.items()):
                    if abs(cx)>0.001 or abs(cy)>0.001:
                        print(f"    {ec}: ref_v={rv:.3f} -> corr ({cx:+.4f}, {cy:+.4f}) mas")
            self._save_ref_table(ref_corr)

            # v0.6: Normal PR anchor — find reference maser position in normal PR
            # and use it to anchor inverse data in the same frame
            norm_anchor_ra, norm_anchor_dec = 0.0, 0.0
            if "normal" in self.matchers and ref_corr:
                norm_anchor_ra, norm_anchor_dec = self._find_normal_anchor(ref_corr)
                if abs(norm_anchor_ra) > 0.01 or abs(norm_anchor_dec) > 0.01:
                    print(f"  Normal PR anchor: ({norm_anchor_ra:+.3f}, {norm_anchor_dec:+.3f}) mas")

            for ec,(feat,spot) in files.items():
                ep=self._get_epoch(ec)
                if ep is None: continue
                # Total shift = quasar-based + ref spot correction + normal anchor
                sx,sy = ep.inv_shift_ra, ep.inv_shift_dec
                if ec in ref_corr:
                    sx += ref_corr[ec][0]; sy += ref_corr[ec][1]
                # Apply normal anchor (shifts all epochs uniformly into normal PR frame)
                sx += norm_anchor_ra; sy += norm_anchor_dec
                if feat is not None:
                    f=feat.copy(); f["x"]=f["x"]+sx; f["y"]=f["y"]+sy
                    s=None
                    if spot is not None:
                        s=spot.copy(); s["x"]=s["x"]+sx; s["y"]=s["y"]+sy
                    matcher_i.add_epoch(ec,f,s)
                elif spot is not None:
                    s=spot.copy(); s["x"]=s["x"]+sx; s["y"]=s["y"]+sy
                    matcher_i.add_epoch(ec,self._spots_to_features(s),s)
                print(f"  INV {ec}: total shift ({sx:+.3f}, {sy:+.3f}) mas")

            # Cross-check
            if "normal" in self.matchers:
                self._crosscheck(matcher_i)

            self.matchers["inverse"] = matcher_i

    def _find_normal_anchor(self, ref_corr):
        """Find position of the reference maser in normal PR epoch 0.

        The inverse PR anchor is at (0,0) = reference maser position.
        Normal PR positions are relative to phase tracking center.
        The offset between reference maser and PTC in normal PR gives
        the shift needed to place inverse data in the normal PR frame.
        """
        norm_m = self.matchers.get("normal")
        if not norm_m: return 0.0, 0.0

        # Get reference maser velocity from epoch 0 of inverse
        ec0 = sorted(ref_corr.keys(),
                     key=lambda c: next((e.dec_year for e in self.epochs if e.code==c),0))[0]
        ref_v = ref_corr[ec0][2]  # reference velocity at anchor epoch

        # Find this velocity in normal PR epoch 0
        if ec0 not in norm_m.features: return 0.0, 0.0
        nf = norm_m.features[ec0]
        vcol = "vlsr_peak" if "vlsr_peak" in nf.columns else "vlsr"
        dv = np.abs(nf[vcol] - ref_v)
        if dv.min() > self.chan_spacing * 3:
            print(f"  WARNING: reference maser not found in normal PR epoch {ec0}")
            return 0.0, 0.0

        idx = dv.idxmin()
        anchor_ra = float(nf.loc[idx, "x"])
        anchor_dec = float(nf.loc[idx, "y"])
        return anchor_ra, anchor_dec

    def _spots_to_features(self, sdf):
        rows=[]
        for v in sdf["vlsr"].unique():
            mask=sdf["vlsr"]==v; sub=sdf[mask]
            rows.append({"vlsr_peak":v,"vlsr":v,
                        "x":sub["x"].mean(),"y":sub["y"].mean(),
                        "I_peak":sub["I"].max() if "I" in sub.columns else 1,
                        "feature_id":len(rows)+1})
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _save_ref_table(self, ref_corr):
        lines=["# Reference maser table (auto-generated)",
               "# Edit corrections and re-run if needed",
               f"# {'epoch':<10s} {'ref_vlsr':>10s} {'match_vlsr':>10s} "
               f"{'corr_x':>10s} {'corr_y':>10s}"]
        for ec,(cx,cy,rv,mv) in sorted(ref_corr.items()):
            lines.append(f"  {ec:<10s} {rv:>10.3f} {mv:>10.3f} "
                        f"{cx:>10.4f} {cy:>10.4f}")
        path=f"{self._prefix}_ref_maser_table.txt"
        open(path,"w").write("\n".join(lines)+"\n")
        print(f"  Reference table: {path}")

    def _crosscheck(self, inv_matcher):
        norm_files=self._files_by_mode.get("normal",{})
        dra,ddec=[],[]
        for ec in inv_matcher.features:
            if ec not in norm_files: continue
            nf,_=norm_files[ec]
            if nf is None: continue
            inv_f=inv_matcher.features[ec]
            vc_n="vlsr_peak" if "vlsr_peak" in nf.columns else "vlsr"
            vc_i="vlsr_peak" if "vlsr_peak" in inv_f.columns else "vlsr"
            for _,nr in nf.iterrows():
                dv=np.abs(inv_f[vc_i]-nr[vc_n])
                if dv.min()<self.chan_spacing*1.5:
                    ii=dv.idxmin()
                    dra.append(inv_f.loc[ii,"x"]-nr["x"])
                    ddec.append(inv_f.loc[ii,"y"]-nr["y"])
        if dra:
            print(f"  Cross-check inv vs norm: "
                  f"median dRA={np.median(dra):+.3f}, dDec={np.median(ddec):+.3f} mas "
                  f"(RMS {np.std(dra):.3f}, {np.std(ddec):.3f})")

    def step3_match(self):
        print(f"\nStep 3 - Matching")
        for mode, matcher in self.matchers.items():
            print(f"\n  --- {mode.upper()} PR ---")
            groups=matcher.match()
            # Apply corrections if provided
            if self.corrections:
                matcher.apply_corrections(self.corrections)
                groups = matcher.groups
            n_good=sum(1 for g in groups if g["n_epochs"]>=self.min_epochs)
            print(f"  {len(groups)} groups, {n_good} with >={self.min_epochs} epochs")
            for gr in "ABCDEF":
                n=sum(1 for g in groups if g["grade"]==gr)
                if n: print(f"    Grade {gr}: {n}")
            # Trajectory flagging report
            traj_flags = sum(1 for g in groups if g.get("flagged_epochs"))
            if traj_flags:
                print(f"  Trajectory check: {traj_flags} groups with flagged epochs")
                for g in groups:
                    if g.get("flagged_epochs"):
                        print(f"    G{g['group_id']} V={g['vlsr_peak']:.1f}: "
                              f"flagged {','.join(g['flagged_epochs'])}")
            # Scatter warnings
            bad=[g for g in groups if g["scatter_flag"] and g["n_epochs"]>=3]
            if bad:
                print(f"  WARNING: {len(bad)} groups with large position scatter:")
                for g in bad[:5]:
                    print(f"    G{g['group_id']} V={g['vlsr_peak']:.1f}: "
                          f"scatter RA={g['scatter_ra']:.2f}, Dec={g['scatter_dec']:.2f} mas")
            # Summary table
            good=sorted([g for g in groups if g["n_epochs"]>=2],
                       key=lambda g:(-g["n_epochs"],-g["n_channels"]))
            if good:
                print(f"\n  {'GID':>4s} {'V_LSR':>8s} {'Nep':>4s} {'Nch':>4s} "
                      f"{'Gr':>3s} {'X':>7s} {'Y':>7s} {'sRA':>5s} {'sDec':>5s} "
                      f"{'dV/dt':>6s} {'Pattern':<12s}")
                for g in good:
                    sf="!" if g["scatter_flag"] else " "
                    tf="T" if g.get("flagged_epochs") else " "
                    vd = g.get("vel_drift",0)
                    vds = f"{vd:+.2f}" if abs(vd) > 0.01 else "  0.0"
                    print(f"  {g['group_id']:>4d} {g['vlsr_peak']:>8.1f} "
                          f"{g['n_epochs']:>4d} {g['n_channels']:>4d} "
                          f"{g['grade']:>3s} {g['x_mean']:>7.1f} {g['y_mean']:>7.1f} "
                          f"{g['scatter_ra']:>5.2f} {g['scatter_dec']:>5.2f}"
                          f" {vds:>6s}"
                          f"{sf}{tf}{g['pattern']:<12s}")
        if len(self.matchers)==2: self._compare()
        if self.save_plots:
            for mode,matcher in self.matchers.items():
                self._plot_mode(matcher,mode)
            if len(self.matchers)==2: self._plot_comparison()
            for mode,matcher in self.matchers.items():
                self._plot_dashboard(matcher,mode)
            if HAS_PLOTLY:
                for mode,matcher in self.matchers.items():
                    self._plot_html(matcher,mode)

    def _compare(self):
        nm=self.matchers.get("normal"); im=self.matchers.get("inverse")
        if not nm or not im: return
        print(f"\n  --- COMPARISON ---")
        n_both=n_no=n_io=inv_better=0
        for ng in nm.groups:
            if ng["n_epochs"]<2: continue
            for ig in im.groups:
                if ig["n_epochs"]<2: continue
                if abs(ng["vlsr_peak"]-ig["vlsr_peak"])<self.chan_spacing*2 and \
                   np.hypot(ng["x_mean"]-ig["x_mean"],ng["y_mean"]-ig["y_mean"])<self.beam_geo*5:
                    n_both+=1
                    if ig["n_epochs"]>ng["n_epochs"]: inv_better+=1
                    break
            else: n_no+=1
        for ig in im.groups:
            if ig["n_epochs"]<2: continue
            if not any(abs(ig["vlsr_peak"]-ng["vlsr_peak"])<self.chan_spacing*2 and
                np.hypot(ig["x_mean"]-ng["x_mean"],ig["y_mean"]-ng["y_mean"])<self.beam_geo*5
                for ng in nm.groups if ng["n_epochs"]>=2):
                n_io+=1
        print(f"  Both: {n_both}, Normal-only: {n_no}, Inverse-only: {n_io}")
        if n_both>0: print(f"  Inverse better in {inv_better}/{n_both} shared groups")

    def step4_save(self):
        print(f"\nStep 4 - Save")
        for mode, matcher in self.matchers.items():
            pfx=f"{self._prefix}_{mode}"
            rows=[{"group_id":g["group_id"],"vlsr_peak":g["vlsr_peak"],
                   "n_epochs":g["n_epochs"],"n_channels":g["n_channels"],
                   "grade":g["grade"],"x_mean":g["x_mean"],"y_mean":g["y_mean"],
                   "scatter_ra":g["scatter_ra"],"scatter_dec":g["scatter_dec"],
                   "vel_drift":g.get("vel_drift",0.0),
                   "vel_range":g.get("vel_range",0.0),
                   "pattern":g["pattern"]} for g in matcher.groups]
            gdf=pd.DataFrame(rows)
            gp=f"{pfx}_groups.txt"
            write_table(gdf,gp,f"Matched groups ({mode} PR)")
            print(f"  {gp} ({len(gdf)} groups)")
            sdf=matcher.build_tracked_spots()
            if len(sdf)>0:
                sp=f"{pfx}_spots_tracked.txt"
                write_table(sdf,sp,f"Tracked spots ({mode} PR)")
                print(f"  {sp} ({len(sdf)} rows)")
            fdf=matcher.build_tracked_features()
            if len(fdf)>0:
                fp=f"{pfx}_features_tracked.txt"
                write_table(fdf,fp,f"Tracked features ({mode} PR)")
                print(f"  {fp} ({len(fdf)} rows)")
            # Feature-group ID mapping
            map_rows = []
            for g in matcher.groups:
                if g["n_epochs"] < 2: continue
                for ec, fidx in g["matches"].items():
                    fdf_ep = matcher.features[ec]
                    fid = int(fdf_ep.loc[fidx].get("feature_id", 0))
                    map_rows.append({"group_id": g["group_id"], "epoch": ec,
                                     "feature_id": fid, "grade": g["grade"],
                                     "vlsr_peak": g["vlsr_peak"]})
            if map_rows:
                mdf = pd.DataFrame(map_rows)
                mp = f"{pfx}_id_mapping.txt"
                write_table(mdf, mp, f"Feature-to-group ID mapping ({mode} PR)")
                print(f"  {mp} ({len(mdf)} rows)")
        # Auto-generate corrections template
        self._save_corrections_template()

    def _save_corrections_template(self):
        """Write an editable corrections template listing all groups and flagged epochs."""
        lines = [
            "# ================================================================",
            "# masertrack_match — Corrections File (auto-generated template)",
            "# ================================================================",
            "#",
            "# HOW TO USE:",
            "#   1. Review the group list below and the diagnostic plots",
            "#   2. Uncomment lines to activate corrections (remove the leading #)",
            "#   3. Re-run: python masertrack_match.py epoch_table.txt --corrections <this_file>",
            "#   4. Check output, repeat if needed",
            "#",
            "# SYNTAX:",
            "#   EPOCH  <epoch_code>  use_for_pi=0    Exclude epoch from parallax fitting",
            "#   GROUP  G<id>  <epoch_code>  EXCLUDE   Remove one epoch from a group",
            "#",
            "# NOTES:",
            "#   - Lines starting with # are comments (inactive)",
            "#   - Uncommented lines starting with EPOCH or GROUP are active",
            "#   - Group IDs (G1, G2...) refer to the group_id in the groups output",
            "#   - 'T' in the pattern column of console output = trajectory flagged",
            "#   - Suggested exclusions are pre-populated below (commented out)",
            "#   - pi = parallax, use_for_pi = include in parallax fitting (1=yes, 0=no)",
            "#",
            "# ================================================================",
            "",
            "# === EPOCH-LEVEL FLAGS ===",
            "# Exclude entire epochs from parallax fitting.",
            "# The epoch is still used for matching, but use_for_pi=0 in output.",
            "",
        ]
        for e in self.epochs:
            status = "# " if e.use_for_pi else "  "
            comment = f"  # {e.flag}" if e.flag else ""
            lines.append(f"{status}EPOCH  {e.code}  use_for_pi={e.use_for_pi}{comment}")
        lines.append("#")
        lines.append("# === GROUP CORRECTIONS ===")
        lines.append("# Exclude a specific epoch from a group:")
        lines.append("#   GROUP  G<id>  <epoch_code>  EXCLUDE")
        lines.append("#")
        for mode, matcher in self.matchers.items():
            lines.append(f"# --- {mode.upper()} PR ---")
            for g in sorted(matcher.groups, key=lambda x: -x["n_epochs"]):
                if g["n_epochs"] < 2: continue
                flagged = g.get("flagged_epochs", [])
                flag_str = f"  # traj_flagged: {','.join(flagged)}" if flagged else ""
                scatter_str = f"  # scatter!" if g.get("scatter_flag") else ""
                lines.append(f"# G{g['group_id']:>3d} V={g['vlsr_peak']:>8.1f} "
                           f"Nep={g['n_epochs']} {g['grade']} "
                           f"{g['pattern']}{flag_str}{scatter_str}")
                # Pre-populate flagged epoch exclusions as commented-out suggestions
                for ec in flagged:
                    lines.append(f"#  GROUP  G{g['group_id']}  {ec}  EXCLUDE")
        path = f"{self._prefix}_corrections_template.txt"
        open(path, "w").write("\n".join(lines) + "\n")
        print(f"  Corrections template: {path}")

    # =================================================================
    #  PLOTS
    # =================================================================
    def _plot_mode(self, matcher, mode):
        """Per-mode diagnostic: sky, V tracking, RA/Dec vs time, RA/Dec residuals."""
        if not HAS_MPL: return
        groups=[g for g in matcher.groups if g["n_epochs"]>=2]
        if not groups: return
        fig,axes=plt.subplots(3,2,figsize=(16,20))
        markers=['o','s','^','D','v','<','>','p','h','*']
        ep_list=[e.code for e in self.epochs]
        all_v=[g["vlsr_peak"] for g in groups]
        vmin,vmax=min(all_v),max(all_v)
        norm_v=plt.Normalize(vmin=vmin,vmax=vmax)
        cmap=matplotlib.colormaps.get_cmap(CMAP_NAME)

        # (0,0) Sky map
        ax=axes[0,0]
        for g in groups:
            c=cmap(norm_v(g["vlsr_peak"]))
            pts=sorted([(matcher._ep_time(ec),
                        matcher.features[ec].loc[i,"x"],
                        matcher.features[ec].loc[i,"y"])
                       for ec,i in g["matches"].items()])
            if len(pts)>1:
                ax.plot([p[1] for p in pts],[p[2] for p in pts],"-",c=c,alpha=0.3,lw=0.7)
            for ec,fidx in g["matches"].items():
                ei=ep_list.index(ec) if ec in ep_list else 0
                fdf=matcher.features[ec]
                ax.scatter(fdf.loc[fidx,"x"],fdf.loc[fidx,"y"],
                    marker=markers[ei%len(markers)],
                    facecolors=[c],s=30,edgecolors="k",linewidths=0.3,zorder=2)
        texts=[]
        for g in groups:
            if g["grade"] in "ABC":
                t=ax.text(g["x_mean"],g["y_mean"],f"G{g['group_id']}",
                         fontsize=5.5,fontweight="bold",ha="left",va="bottom")
                texts.append(t)
        if HAS_ADJTEXT and texts: adjust_text(texts,ax=ax)
        ax.set_xlabel("RA (mas)"); ax.set_ylabel("Dec (mas)")
        ax.invert_xaxis(); ax.set_aspect("equal")
        try: ax.set_box_aspect(1)
        except AttributeError: pass
        sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm_v)
        plt.colorbar(sm,ax=ax,label="V_LSR (km/s)")
        for ei,ec in enumerate(ep_list):
            if any(ec in g["matches"] for g in groups):
                ax.scatter([],[],marker=markers[ei%len(markers)],
                          facecolors="gray",s=25,edgecolors="k",linewidths=0.3,label=ec)
        ax.legend(fontsize=5,ncol=2,loc="lower left")
        ax.set_title(f"Sky map ({mode} PR)")

        # (0,1) V vs time
        ax=axes[0,1]
        for g in groups:
            c=cmap(norm_v(g["vlsr_peak"]))
            pts=[]
            for ec,fidx in g["matches"].items():
                fdf=matcher.features[ec]
                vc="vlsr_peak" if "vlsr_peak" in fdf.columns else "vlsr"
                pts.append((matcher._ep_time(ec),fdf.loc[fidx,vc]))
            pts.sort()
            for t,v in pts:
                ax.scatter(t,v,c=[c],s=25,edgecolors="k",linewidths=0.2,zorder=2)
            if len(pts)>1:
                ax.plot([p[0] for p in pts],[p[1] for p in pts],"-",c=c,alpha=0.5,lw=0.8)
            if g["grade"] in "ABC":
                ax.text(pts[-1][0],pts[-1][1],f" G{g['group_id']}({g['grade']})",
                       fontsize=5,va="center",color=c)
        ax.set_xlabel("Epoch (yr)"); ax.set_ylabel("V_LSR (km/s)")
        ax.set_title("Velocity tracking")

        # Rows 1-2: RA/Dec vs time + residuals for A/B groups
        ab=[g for g in groups if g["grade"] in "AB"]
        cm_ab=plt.cm.tab10
        for ci,(coord,label) in enumerate([("x","RA"),("y","Dec")]):
            ax_pos=axes[1,ci]; ax_res=axes[2,ci]
            for gi,g in enumerate(ab):
                c=cm_ab(gi%10)
                pts=sorted([(matcher._ep_time(ec),
                    float(matcher.features[ec].loc[i,coord]),ec)
                    for ec,i in g["matches"].items()])
                ts=np.array([p[0] for p in pts])
                vals=np.array([p[1] for p in pts])
                ecs_list=[p[2] for p in pts]
                ax_pos.plot(ts,vals,"o-",c=c,ms=4,lw=1,alpha=0.8,
                    label=f"G{g['group_id']} ({g['vlsr_peak']:.0f})")
                if len(ts)>=3:
                    coeff=np.polyfit(ts-ts[0],vals,1)
                    resid=vals-np.polyval(coeff,ts-ts[0])
                else:
                    resid=np.zeros_like(vals)
                ax_res.plot(ts,resid,"o-",c=c,ms=4,lw=1,alpha=0.8,
                    label=f"G{g['group_id']}")
                for k,ec in enumerate(ecs_list):
                    if ec in g.get("flagged_epochs",[]):
                        ax_pos.scatter(ts[k],vals[k],s=100,facecolors="none",
                            edgecolors="red",linewidths=2,zorder=3)
                        ax_res.scatter(ts[k],resid[k],s=100,facecolors="none",
                            edgecolors="red",linewidths=2,zorder=3)
            ax_pos.set_xlabel("Epoch (yr)"); ax_pos.set_ylabel(f"{label} (mas)")
            ax_pos.legend(fontsize=6,ncol=2); ax_pos.set_title(f"{label} vs time (A/B)")
            ax_res.set_xlabel("Epoch (yr)"); ax_res.set_ylabel(f"Residual (mas)")
            ax_res.axhline(0,color="k",lw=0.5,ls="--")
            ax_res.legend(fontsize=6,ncol=2)
            ax_res.set_title(f"{label} residuals (red ring = flagged)")

        n_ab=len(ab)
        fig.suptitle(f"masertrack_match v{__version__} — {mode.upper()} PR\n"
                     f"{len(groups)} groups (≥2 ep), {n_ab} Grade A/B",fontsize=12)
        plt.tight_layout(rect=[0,0,1,0.96])
        path=f"{self._prefix}_{mode}_diagnostic.png"
        fig.savefig(path,dpi=150,bbox_inches="tight")
        if self.show_plots: plt.show()
        else: plt.close(fig)
        print(f"  Plot: {path}")

    def _plot_comparison(self):
        """Comparison: sky maps (common labels only) + grade hist + stats text."""
        if not HAS_MPL: return
        nm=self.matchers.get("normal"); im=self.matchers.get("inverse")
        if not nm or not im: return
        fig=plt.figure(figsize=(18,12))
        ax1=fig.add_subplot(2,2,1); ax2=fig.add_subplot(2,2,2)
        ax3=fig.add_subplot(2,2,3); ax4=fig.add_subplot(2,2,4)

        # Find common groups (matched in both modes by V and position)
        common_vlsr=set()
        n_both=n_no=n_io=inv_better=0
        for ng in nm.groups:
            if ng["n_epochs"]<2: continue
            found=False
            for ig in im.groups:
                if ig["n_epochs"]<2: continue
                if abs(ng["vlsr_peak"]-ig["vlsr_peak"])<self.chan_spacing*2 and \
                   np.hypot(ng["x_mean"]-ig["x_mean"],ng["y_mean"]-ig["y_mean"])<self.beam_geo*5:
                    n_both+=1; found=True
                    common_vlsr.add(round(ng["vlsr_peak"],1))
                    if ig["n_epochs"]>ng["n_epochs"]: inv_better+=1
                    break
            if not found: n_no+=1
        for ig in im.groups:
            if ig["n_epochs"]<2: continue
            if not any(abs(ig["vlsr_peak"]-ng["vlsr_peak"])<self.chan_spacing*2 and
                np.hypot(ig["x_mean"]-ng["x_mean"],ig["y_mean"]-ng["y_mean"])<self.beam_geo*5
                for ng in nm.groups if ng["n_epochs"]>=2):
                n_io+=1

        # Sky maps — label only common groups
        all_x,all_y=[],[]
        for matcher in [nm,im]:
            for g in matcher.groups:
                if g["n_epochs"]>=2:
                    all_x.append(g["x_mean"]); all_y.append(g["y_mean"])
        if all_x:
            span=max(max(all_x)-min(all_x),max(all_y)-min(all_y))
            pad=span*0.15+2
            cx,cy=np.mean(all_x),np.mean(all_y)
            xlim=(cx+span/2+pad,cx-span/2-pad)  # inverted
            ylim=(cy-span/2-pad,cy+span/2+pad)

        for ax,matcher,title in [(ax1,nm,"Normal PR"),(ax2,im,"Inverse PR")]:
            good=[g for g in matcher.groups if g["n_epochs"]>=2]
            vs=[g["vlsr_peak"] for g in good]; xs=[g["x_mean"] for g in good]
            ys=[g["y_mean"] for g in good]
            sz=[max(40,g["n_epochs"]*15) for g in good]
            if vs:
                sc=ax.scatter(xs,ys,c=vs,s=sz,cmap=CMAP_NAME,edgecolors="k",
                             linewidths=0.5,vmin=-155,vmax=0)
                for g in good:
                    if round(g["vlsr_peak"],1) in common_vlsr and g["grade"] in "ABC":
                        ax.annotate(f"{g['vlsr_peak']:.0f}",
                                   (g["x_mean"],g["y_mean"]),fontsize=5.5,
                                   fontweight="bold",xytext=(3,3),
                                   textcoords="offset points")
                plt.colorbar(sc,ax=ax,label="V_LSR (km/s)",shrink=0.8)
            ax.set_xlabel("RA (mas)"); ax.set_ylabel("Dec (mas)")
            if all_x: ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_aspect("equal")
            try: ax.set_box_aspect(1)
            except AttributeError: pass
            n_ab=sum(1 for g in good if g["grade"] in "AB")
            ax.set_title(f"{title}: {len(good)} groups, {n_ab} A/B")

        # (1,0) Grade histogram with bar numbers
        ax=ax3
        grade_labels=list("ABCDEF")
        x_pos=np.arange(6); width=0.35
        colors_g=["#2ecc71","#3498db","#f39c12","#e67e22","#e74c3c","#95a5a6"]
        for mi,(mname,matcher) in enumerate([("normal",nm),("inverse",im)]):
            counts=[sum(1 for g in matcher.groups if g["grade"]==gr) for gr in grade_labels]
            offset=(mi-0.5)*width
            bars=ax.bar(x_pos+offset,counts,width,label=mname.upper(),
                  color=colors_g,edgecolor="k",linewidth=0.5,
                  alpha=0.8 if mi==0 else 0.5)
            for b,cnt in zip(bars,counts):
                if cnt>0: ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.2,
                                 str(cnt),ha="center",fontsize=7,fontweight="bold")
        ax.set_xticks(x_pos); ax.set_xticklabels(grade_labels)
        ax.set_xlabel("Grade"); ax.set_ylabel("Count")
        ax.legend(); ax.set_title("Grade comparison")

        # (1,1) Comparison statistics text box
        ax=ax4; ax.axis("off")
        n_n=sum(1 for g in nm.groups if g["n_epochs"]>=2)
        n_i=sum(1 for g in im.groups if g["n_epochs"]>=2)
        nab_n=sum(1 for g in nm.groups if g["grade"] in "AB")
        nab_i=sum(1 for g in im.groups if g["grade"] in "AB")
        stats=[
            "COMPARISON STATISTICS",
            "",
            f"Normal PR:  {n_n} groups (≥2 ep), {nab_n} A/B",
            f"Inverse PR: {n_i} groups (≥2 ep), {nab_i} A/B",
            "",
            f"Common groups:     {n_both}",
            f"  (matched by V within 2ch",
            f"   and position within 5×beam)",
            f"Normal-only:       {n_no}",
            f"Inverse-only:      {n_io}",
            f"Inverse better in: {inv_better}/{n_both} shared",
            "",
            f"Sky map labels = V_LSR of",
            f"groups found in both modes",
        ]
        ax.text(0.1,0.9,"\n".join(stats),transform=ax.transAxes,
               fontsize=11,fontfamily="monospace",verticalalignment="top",
               bbox=dict(boxstyle="round",fc="lightyellow",alpha=0.8))

        fig.suptitle("Normal vs Inverse PR — Comparison",fontsize=13)
        plt.tight_layout(rect=[0,0,1,0.96])
        path=f"{self._prefix}_comparison.png"
        fig.savefig(path,dpi=150,bbox_inches="tight")
        if self.show_plots: plt.show()
        else: plt.close(fig)
        print(f"  Comparison: {path}")

    def _plot_dashboard(self, matcher, mode):
        """Per-mode dashboard: grade dist, stats, velocity coverage, drift."""
        if not HAS_MPL: return
        fig,axes=plt.subplots(2,2,figsize=(16,12))
        good=[g for g in matcher.groups if g["n_epochs"]>=2]
        n_unflagged=sum(1 for e in self.epochs if not e.flag)

        # (0,0) Grade distribution
        ax=axes[0,0]
        grade_labels=list("ABCDEF")
        colors_g=["#2ecc71","#3498db","#f39c12","#e67e22","#e74c3c","#95a5a6"]
        counts=[sum(1 for g in matcher.groups if g["grade"]==gr) for gr in grade_labels]
        bars=ax.bar(range(6),counts,color=colors_g,edgecolor="k",linewidth=0.5)
        for b,cnt in zip(bars,counts):
            if cnt>0: ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.2,
                             str(cnt),ha="center",fontsize=8,fontweight="bold")
        ax.set_xticks(range(6)); ax.set_xticklabels(grade_labels)
        ax.set_xlabel("Grade"); ax.set_ylabel("Count")
        ax.set_title(f"Grade distribution ({mode} PR)")

        # (0,1) Statistics text
        ax=axes[0,1]; ax.axis("off")
        n_all=len(matcher.groups)
        n2=sum(1 for g in matcher.groups if g["n_epochs"]>=2)
        nab=sum(1 for g in matcher.groups if g["grade"] in "AB")
        n_tracked=len(matcher.build_tracked_spots())
        traj_f=sum(1 for g in matcher.groups if g.get("flagged_epochs"))
        stats=[
            f"masertrack_match v{__version__}",
            f"{mode.upper()} phase referencing",
            f"",
            f"Epochs: {len(self.epochs)} total, {n_unflagged} unflagged",
            f"Beam geo: {self.beam_geo:.2f} mas",
            f"Chan spacing: {self.chan_spacing:.3f} km/s",
            f"Vel tolerance: {self.vtc} ch = {self.vtc*self.chan_spacing:.2f} km/s",
            f"Match radius: {self.mrf:.1f} x beam = {self.mrf*self.beam_geo:.2f} mas",
            f"",
            f"Groups: {n_all} total, {n2} with ≥2 epochs",
            f"Grade A/B: {nab}",
            f"Tracked spots: {n_tracked}",
        ]
        if traj_f: stats.append(f"Trajectory flagged: {traj_f} groups")
        ax.text(0.05,0.95,"\n".join(stats),transform=ax.transAxes,
               fontsize=10,fontfamily="monospace",verticalalignment="top",
               bbox=dict(boxstyle="round",fc="lightyellow",alpha=0.8))

        # (1,0) Velocity coverage
        ax=axes[1,0]
        if good:
            vs=[g["vlsr_peak"] for g in good]
            ns=[g["n_epochs"] for g in good]
            cmap_d=matplotlib.colormaps.get_cmap(CMAP_NAME)
            vmin_d,vmax_d=min(vs),max(vs)
            colors=[(cmap_d((v-vmin_d)/(vmax_d-vmin_d)) if vmax_d>vmin_d else cmap_d(0.5)) for v in vs]
            ax.barh(range(len(good)),ns,color=colors,edgecolor="k",linewidth=0.3)
            ax.set_yticks(range(len(good)))
            ax.set_yticklabels([f"G{g['group_id']} {g['vlsr_peak']:.0f}" for g in good],fontsize=6)
            ax.set_xlabel("Epochs matched")
            ax.axvline(x=n_unflagged,color="red",ls="--",alpha=0.5,
                      label=f"All unflagged ({n_unflagged})")
            ax.legend(fontsize=7)
        ax.set_title("Velocity coverage")

        # (1,1) Velocity drift
        ax=axes[1,1]
        drifters=[g for g in good if g["n_epochs"]>=3]
        if drifters:
            vd=[g.get("vel_drift",0) for g in drifters]
            labels=[f"G{g['group_id']}" for g in drifters]
            ax.barh(range(len(drifters)),vd,color="steelblue",edgecolor="k",linewidth=0.3)
            ax.set_yticks(range(len(drifters)))
            ax.set_yticklabels(labels,fontsize=6)
            ax.axvline(x=0,color="k",lw=0.5)
            ax.set_xlabel("Velocity drift (km/s/yr)")
            for i,g in enumerate(drifters):
                if abs(g.get("vel_drift",0))>0.5:
                    ax.text(g["vel_drift"],i,f" {g.get('vel_range',0):.1f}",
                           fontsize=6,va="center",color="red")
        ax.set_title("Velocity drift (≥3 epochs)")

        fig.suptitle(f"masertrack_match v{__version__} — {mode.upper()} PR Dashboard",
                     fontsize=13,fontweight="bold")
        plt.tight_layout(rect=[0,0,1,0.96])
        path=f"{self._prefix}_{mode}_dashboard.png"
        fig.savefig(path,dpi=150,bbox_inches="tight")
        if self.show_plots: plt.show()
        else: plt.close(fig)
        print(f"  Dashboard: {path}")

    def _plot_html(self, matcher, mode):
        """Interactive HTML with 4 panels: sky map, V vs time, RA vs time, Dec vs time.

        Sky maps share linked axes. Hover shows epoch details and residuals.
        Legend toggles are linked across all panels via legendgroup.
        """
        if not HAS_PLOTLY: return
        groups=[g for g in matcher.groups if g["n_epochs"]>=2]
        if not groups: return
        all_v=[g["vlsr_peak"] for g in groups]
        vmin,vmax=min(all_v),max(all_v)
        fig=make_subplots(rows=3,cols=2,
            subplot_titles=("Sky map (hover for details)",
                            "Velocity tracking",
                            "RA offset vs time",
                            "Dec offset vs time",
                            "Spot sky map (per group, by epoch)",
                            "Spot velocity (per group, by epoch)"),
            vertical_spacing=0.07, horizontal_spacing=0.08,
            row_heights=[0.35,0.30,0.35])

        for g in groups:
            vnorm=(g["vlsr_peak"]-vmin)/(vmax-vmin) if vmax>vmin else 0.5
            ci=min(int(vnorm*(len(PLOTLY_RAINBOW)-1)),len(PLOTLY_RAINBOW)-2)
            fc=PLOTLY_RAINBOW[ci][1]
            gid=g["group_id"]
            lgr=f"g{gid}"
            label=f"G{gid} ({g['grade']}) V={g['vlsr_peak']:.1f}"

            # Collect per-epoch data
            pts=[]
            for ec,fidx in sorted(g["matches"].items(), key=lambda x: matcher._ep_time(x[0])):
                fdf=matcher.features[ec]
                vc="vlsr_peak" if "vlsr_peak" in fdf.columns else "vlsr"
                x,y,v=fdf.loc[fidx,"x"],fdf.loc[fidx,"y"],fdf.loc[fidx,vc]
                ep=next((e for e in self.epochs if e.code==ec),None)
                t=matcher._ep_time(ec)
                flag_str=""
                if ep and ep.flag: flag_str=f"<br>⚠ {ep.flag}"
                if ec in g.get("flagged_epochs",[]): flag_str+=f"<br>⚠ traj_outlier"
                ht=(f"<b>G{gid}</b> ({g['grade']})<br>"
                    f"Epoch: {ec}{flag_str}<br>"
                    f"V: {v:.3f} km/s<br>"
                    f"X: {x:.3f} mas<br>Y: {y:.3f} mas<br>"
                    f"dV/dt: {g.get('vel_drift',0):+.2f} km/s/yr")
                pts.append({"t":t,"x":x,"y":y,"v":v,"ec":ec,"ht":ht})

            # Compute opacity per epoch: earliest=0.3, latest=1.0
            all_times = sorted(set(matcher._ep_time(ec) for ec in matcher.features))
            def _opacity(t):
                if len(all_times)<2: return 1.0
                return 0.3 + 0.7*(t-all_times[0])/(all_times[-1]-all_times[0])
            opacities = [_opacity(p["t"]) for p in pts]

            # Sky map (row 1, col 1)
            if len(pts)>1:
                fig.add_trace(go.Scatter(
                    x=[p["x"] for p in pts],y=[p["y"] for p in pts],
                    mode="lines",line=dict(color=fc,width=1,dash="dot"),
                    showlegend=False,hoverinfo="skip",legendgroup=lgr,
                ),row=1,col=1)
            fig.add_trace(go.Scatter(
                x=[p["x"] for p in pts],y=[p["y"] for p in pts],
                mode="markers+text",
                marker=dict(size=9,color=fc,opacity=opacities,
                    line=dict(width=0.5,color="black")),
                text=[f"G{gid}" if i==0 else "" for i in range(len(pts))],
                textposition="top center",textfont=dict(size=7),
                hovertext=[p["ht"] for p in pts],hoverinfo="text",
                name=label,legendgroup=lgr,
            ),row=1,col=1)

            # V vs time (row 1, col 2) — with label
            fig.add_trace(go.Scatter(
                x=[p["t"] for p in pts],y=[p["v"] for p in pts],
                mode="markers+lines+text",
                marker=dict(size=7,color=fc,opacity=opacities,
                    line=dict(width=0.5,color="black")),
                line=dict(color=fc,width=1),
                text=[f"G{gid}" if i==len(pts)-1 else "" for i in range(len(pts))],
                textposition="middle right",textfont=dict(size=6),
                hovertext=[p["ht"] for p in pts],hoverinfo="text",
                name=label,legendgroup=lgr,showlegend=False,
            ),row=1,col=2)

            # RA vs time (row 2, col 1) — with label
            fig.add_trace(go.Scatter(
                x=[p["t"] for p in pts],y=[p["x"] for p in pts],
                mode="markers+lines+text",
                marker=dict(size=6,color=fc,opacity=opacities,
                    line=dict(width=0.5,color="black")),
                line=dict(color=fc,width=1),
                text=[f"G{gid}" if i==len(pts)-1 else "" for i in range(len(pts))],
                textposition="middle right",textfont=dict(size=6),
                hovertext=[p["ht"] for p in pts],hoverinfo="text",
                name=label,legendgroup=lgr,showlegend=False,
            ),row=2,col=1)

            # Dec vs time (row 2, col 2) — with label
            fig.add_trace(go.Scatter(
                x=[p["t"] for p in pts],y=[p["y"] for p in pts],
                mode="markers+lines+text",
                marker=dict(size=6,color=fc,opacity=opacities,
                    line=dict(width=0.5,color="black")),
                line=dict(color=fc,width=1),
                text=[f"G{gid}" if i==len(pts)-1 else "" for i in range(len(pts))],
                textposition="middle right",textfont=dict(size=6),
                hovertext=[p["ht"] for p in pts],hoverinfo="text",
                name=label,legendgroup=lgr,showlegend=False,
            ),row=2,col=2)

            # Row 3: Spot-level sky map and velocity (within this group)
            ep_colors_rgb = ["rgb(31,119,180)","rgb(255,127,14)","rgb(44,160,44)",
                "rgb(214,39,40)","rgb(148,103,189)","rgb(140,86,75)",
                "rgb(227,119,194)","rgb(127,127,127)","rgb(188,189,34)","rgb(23,190,207)"]
            for ec,fidx in g["matches"].items():
                if ec not in matcher.spots: continue
                sdf=matcher.spots[ec]; fdf_ep=matcher.features[ec]
                if "feature_id" in sdf.columns and "feature_id" in fdf_ep.columns:
                    fid=fdf_ep.loc[fidx,"feature_id"]
                    smask=sdf["feature_id"]==fid
                else:
                    fx,fy=fdf_ep.loc[fidx,"x"],fdf_ep.loc[fidx,"y"]
                    smask=np.hypot(sdf["x"]-fx,sdf["y"]-fy)<matcher.beam_geo*2
                sub=sdf[smask]
                if len(sub)==0: continue
                ei=all_times.index(matcher._ep_time(ec)) if matcher._ep_time(ec) in all_times else 0
                ec_rgb=ep_colors_rgb[ei%len(ep_colors_rgb)]
                # Marker size proportional to flux (sqrt scaling, min 4, max 20)
                fluxes = sub["I"].values if "I" in sub.columns else np.ones(len(sub))
                fmax = max(fluxes.max(), 0.01)
                spot_sz = np.clip(4 + 10*np.sqrt(fluxes/fmax), 4, 14)
                spot_ht=[f"<b>G{gid}</b> spot<br>Epoch: {ec}<br>"
                         f"V: {r['vlsr']:.3f}<br>X: {r['x']:.3f}<br>Y: {r['y']:.3f}<br>"
                         f"Flux: {r.get('I',0):.3f}"
                         for _,r in sub.iterrows()]
                fig.add_trace(go.Scatter(
                    x=sub["x"].values,y=sub["y"].values,mode="markers",
                    marker=dict(size=spot_sz,color=ec_rgb,
                        line=dict(width=0.3,color="black"),
                        opacity=_opacity(matcher._ep_time(ec))),
                    hovertext=spot_ht,hoverinfo="text",
                    name=label,legendgroup=lgr,showlegend=False,
                ),row=3,col=1)
                fig.add_trace(go.Scatter(
                    x=sub["vlsr"].values,
                    y=[matcher._ep_time(ec)]*len(sub),
                    mode="markers",
                    marker=dict(size=spot_sz,color=ec_rgb,
                        line=dict(width=0.3,color="black"),
                        opacity=_opacity(matcher._ep_time(ec))),
                    hovertext=spot_ht,hoverinfo="text",
                    name=label,legendgroup=lgr,showlegend=False,
                ),row=3,col=2)

        fig.update_xaxes(title_text="RA (mas)",autorange="reversed",row=1,col=1)
        fig.update_yaxes(title_text="Dec (mas)",row=1,col=1)
        fig.update_xaxes(title_text="Epoch (yr)",row=1,col=2)
        fig.update_yaxes(title_text="V_LSR (km/s)",row=1,col=2)
        fig.update_xaxes(title_text="Epoch (yr)",row=2,col=1)
        fig.update_yaxes(title_text="RA offset (mas)",row=2,col=1)
        fig.update_xaxes(title_text="Epoch (yr)",row=2,col=2)
        fig.update_yaxes(title_text="Dec offset (mas)",row=2,col=2)
        fig.update_xaxes(title_text="RA (mas)",autorange="reversed",row=3,col=1)
        fig.update_yaxes(title_text="Dec (mas)",row=3,col=1)
        fig.update_xaxes(title_text="V_LSR (km/s)",row=3,col=2)
        fig.update_yaxes(title_text="Epoch (yr)",row=3,col=2)

        n_ab = sum(1 for g in groups if g["grade"] in "AB")
        fig.update_layout(
            title=f"masertrack_match v{__version__} — {mode.upper()} PR | "
                  f"{len(groups)} groups (≥2 ep), {n_ab} A/B | "
                  f"Click legend to toggle | Faint=early, solid=late",
            height=1300,width=1400,hovermode="closest",
            updatemenus=[dict(
                type="buttons",direction="left",x=0.0,y=1.02,
                xanchor="left",yanchor="bottom",
                buttons=[
                    dict(label="Show all",method="update",
                         args=[{"visible":True}]),
                    dict(label="Hide all",method="update",
                         args=[{"visible":"legendonly"}]),
                ])])

        hp=f"{self._prefix}_{mode}_interactive.html"
        fig.write_html(hp,include_plotlyjs="cdn")
        print(f"  HTML: {hp}")

    def run(self):
        self.step1_parse(); self.step2_align()
        self.step3_match(); self.step4_save()
        return self.matchers

def main():
    pa=argparse.ArgumentParser(prog="masertrack_match",
        description=f"masertrack_match v{__version__}: Cross-epoch matching of maser features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python masertrack_match.py epoch_table.txt --input-dir ./input --outdir results
          python masertrack_match.py epoch_table.txt --corrections corrections.txt
          python masertrack_match.py epoch_table.txt --no-plots

        Corrections file format (one per line, # = comment):
          EPOCH  r14307a  use_for_pi=0
          GROUP  G6  r15037b  EXCLUDE

        Output files (per mode):
          match_{mode}_groups.txt           Matched group catalog
          match_{mode}_spots_tracked.txt    Channel positions for pi fitting
          match_{mode}_features_tracked.txt Feature positions with errors
          match_{mode}_id_mapping.txt       Feature ID to group ID mapping
          match_{mode}_diagnostic.png       6-panel diagnostic plot
          match_{mode}_dashboard.png        4-panel dashboard
          match_{mode}_interactive.html     Interactive plotly (open in browser)
          match_comparison.png              Normal vs inverse comparison
          match_corrections_template.txt    Editable corrections template
          match_ref_maser_table.txt         Reference maser corrections
        """))
    pa.add_argument("epoch_table",help="Epoch table file (space-separated)")
    pa.add_argument("--input-dir",default="./input",
                    help="Directory with feature/spot files (default: ./input)")
    pa.add_argument("--outdir",default=".",help="Output directory (default: .)")
    pa.add_argument("--beam-major",type=float,help="Beam major axis in mas (default: auto-detect)")
    pa.add_argument("--beam-minor",type=float,help="Beam minor axis in mas (default: auto-detect)")
    pa.add_argument("--chan-spacing",type=float,help="Channel spacing in km/s (default: auto-detect)")
    pa.add_argument("--match-radius",type=float,
                    help=f"Match radius as multiple of beam geo mean (default: {DEFAULTS['match_radius_factor']})")
    pa.add_argument("--vel-tolerance",type=float,
                    help=f"Velocity tolerance in channels (default: {DEFAULTS['vel_tolerance_channels']})")
    pa.add_argument("--max-pm",type=float,
                    help=f"Max proper motion in mas/yr (default: {DEFAULTS['max_pm']})")
    pa.add_argument("--min-epochs",type=int,
                    help=f"Min epochs for 'good' group (default: {DEFAULTS['min_epochs']})")
    pa.add_argument("--corrections",type=str,default=None,
                    help="Corrections file for manual overrides (see template in output)")
    pa.add_argument("--no-plots",action="store_true",help="Disable all plot generation")
    pa.add_argument("--show-plots",action="store_true",
                    help="Open matplotlib windows (default: save PNG only)")
    args=pa.parse_args()
    m=MaserTrackMatch(args.epoch_table,input_dir=args.input_dir,outdir=args.outdir,
        beam_major=args.beam_major,beam_minor=args.beam_minor,chan_spacing=args.chan_spacing,
        match_radius_factor=args.match_radius,vel_tolerance=args.vel_tolerance,
        max_pm=args.max_pm,min_epochs=args.min_epochs,
        corrections_file=args.corrections,
        save_plots=not args.no_plots,show_plots=args.show_plots)
    m.run()

if __name__=="__main__": main()
