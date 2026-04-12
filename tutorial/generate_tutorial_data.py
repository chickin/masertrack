#!/usr/bin/env python3
"""
Generate synthetic VLBI maser data for the masertrack tutorial.

Creates input files in BOTH normal and inverse phase referencing modes.
The synthetic source has a KNOWN parallax (pi = 0.500 mas) so the user
can verify the pipeline recovers the correct answer in both modes.

Normal PR:  maser positions are absolute (parallax + PM visible directly)
Inverse PR: positions relative to reference maser at (0,0); quasar's
            apparent motion encodes the inverse maser motion

Both modes should give the same parallax result.

Reproducibility: fixed seeds (42=normal, 43=inverse).
"""
import numpy as np
from pathlib import Path

# =====================================================================
#  SOURCE PARAMETERS (the "truth" the user should recover)
# =====================================================================
RA_STR, DEC_STR = "06:10:00.000", "+20:30:00.000"
RA_HR   = 6 + 10/60.0
RA_RAD  = RA_HR * np.pi / 12.0
DEC_DEG = 20 + 30/60.0
DEC_RAD = DEC_DEG * np.pi / 180.0

PI_TRUE   = 0.500     # mas
MU_X_TRUE = -2.00     # mas/yr (RA*cos(dec))
MU_Y_TRUE = +1.00     # mas/yr (Dec)
QSO_RA_S  = 10.0      # arbitrary quasar RA seconds for inverse PR
QSO_DEC_S = 0.0       # arbitrary quasar Dec arcsec for inverse PR

# =====================================================================
#  ARRAY PARAMETERS
# =====================================================================
N_STATIONS  = 4
BEAM_MAJ    = 1.80     # mas
BEAM_MIN    = 0.80     # mas
BEAM_PA     = -20.0    # degrees
CHAN_SPACING = 0.42     # km/s
BEAM_GEO    = np.sqrt(BEAM_MAJ * BEAM_MIN)

# =====================================================================
#  EPOCHS
# =====================================================================
EPOCHS = [
    {"code":"e01","date":"2022-03-15","mjd":59653,"n_sta":4},
    {"code":"e02","date":"2022-06-01","mjd":59731,"n_sta":4},
    {"code":"e03","date":"2022-08-20","mjd":59811,"n_sta":4},
    {"code":"e04","date":"2022-11-10","mjd":59893,"n_sta":4},
    {"code":"e05","date":"2023-01-25","mjd":59969,"n_sta":4},
    {"code":"e06","date":"2023-04-15","mjd":60049,"n_sta":4},
]
for e in EPOCHS:
    e["dec_year"] = 2000.0 + (e["mjd"] - 51544.5) / 365.25
T_REF_MJD = np.mean([e["mjd"] for e in EPOCHS])

# =====================================================================
#  FEATURES (F1 = inverse PR reference maser)
# =====================================================================
FEATURES = [
    {"id":1, "vlsr_center":-35.0, "n_spots":5, "base_x":5.0, "base_y":3.0,
     "int_mu_x":0.0, "int_mu_y":0.0, "flux_peak":10.0,
     "flux_profile":[0.15,0.45,1.00,0.60,0.10],
     "present_epochs":[0,1,2,3,4,5], "is_reference":True},
    {"id":2, "vlsr_center":-42.0, "n_spots":4, "base_x":-8.0, "base_y":6.0,
     "int_mu_x":+0.50, "int_mu_y":-0.30, "flux_peak":5.0,
     "flux_profile":[0.30,1.00,0.70,0.15],
     "present_epochs":[0,1,2,3,4,5], "is_reference":False},
    {"id":3, "vlsr_center":-28.0, "n_spots":5, "base_x":2.0, "base_y":-10.0,
     "int_mu_x":-0.30, "int_mu_y":+0.80, "flux_peak":3.0,
     "flux_profile":[0.10,0.50,1.00,0.80,0.20],
     "present_epochs":[0,1,2,3,4,5], "is_reference":False},
    {"id":4, "vlsr_center":-50.0, "n_spots":3, "base_x":-4.0, "base_y":-5.0,
     "int_mu_x":+0.20, "int_mu_y":+0.10, "flux_peak":2.0,
     "flux_profile":[0.25,1.00,0.40],
     "present_epochs":[0,1,3,4], "is_reference":False},  # Grade B
]

# =====================================================================
#  PARALLAX FACTORS
# =====================================================================
def parallax_factors(ra_rad, dec_rad, mjd):
    n = mjd - 51544.5
    L = np.radians((280.460 + 0.9856474 * n) % 360.0)
    g = np.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = L + np.radians(1.915)*np.sin(g) + np.radians(0.020)*np.sin(2*g)
    eps = np.radians(23.439 - 0.0000004 * n)
    R = 1.00014 - 0.01671*np.cos(g) - 0.00014*np.cos(2*g)
    X, Y, Z = -R*np.cos(lam), -R*np.sin(lam)*np.cos(eps), -R*np.sin(lam)*np.sin(eps)
    sa, ca = np.sin(ra_rad), np.cos(ra_rad)
    sd, cd = np.sin(dec_rad), np.cos(dec_rad)
    return X*sa - Y*ca, X*ca*sd + Y*sa*sd - Z*cd

def get_ref_position(epoch_idx):
    """Noiseless absolute position of the reference maser at an epoch."""
    dt = (EPOCHS[epoch_idx]["mjd"] - T_REF_MJD) / 365.25
    F_a, F_d = parallax_factors(RA_RAD, DEC_RAD, EPOCHS[epoch_idx]["mjd"])
    ref = next(f for f in FEATURES if f.get("is_reference"))
    return (ref["base_x"] + MU_X_TRUE*dt + PI_TRUE*F_a + ref["int_mu_x"]*dt,
            ref["base_y"] + MU_Y_TRUE*dt + PI_TRUE*F_d + ref["int_mu_y"]*dt)

# =====================================================================
#  SPOT GENERATION
# =====================================================================
def generate_spots(epoch_idx, rng, mode="normal"):
    """Generate spots for one epoch in normal or inverse PR."""
    mjd = EPOCHS[epoch_idx]["mjd"]
    dt = (mjd - T_REF_MJD) / 365.25
    F_a, F_d = parallax_factors(RA_RAD, DEC_RAD, mjd)

    # For inverse: reference maser position to subtract
    ref_x, ref_y = (0.0, 0.0)
    if mode == "inverse":
        ref_x, ref_y = get_ref_position(epoch_idx)

    spots = []
    for feat in FEATURES:
        if epoch_idx not in feat["present_epochs"]:
            continue
        n_spots = feat["n_spots"]
        v_start = feat["vlsr_center"] - (n_spots-1)/2.0 * CHAN_SPACING

        # Absolute position
        fx = feat["base_x"] + MU_X_TRUE*dt + PI_TRUE*F_a + feat["int_mu_x"]*dt
        fy = feat["base_y"] + MU_Y_TRUE*dt + PI_TRUE*F_d + feat["int_mu_y"]*dt

        # In inverse PR, subtract reference to get relative positions
        fx -= ref_x
        fy -= ref_y

        for si in range(n_spots):
            vlsr = v_start + si * CHAN_SPACING
            flux = feat["flux_peak"] * feat["flux_profile"][si]
            rms = 0.05
            pos_err = BEAM_GEO / (2 * flux / rms)
            dx = rng.normal(0, 0.05) * si * 0.3
            dy = rng.normal(0, 0.05) * si * 0.3
            spots.append({
                "vlsr": vlsr,
                "x": fx + dx + rng.normal(0, pos_err),
                "y": fy + dy + rng.normal(0, pos_err),
                "x_err": pos_err, "y_err": pos_err,
                "I": flux + rng.normal(0, rms*0.3), "I_err": rms,
                "S": flux + rng.normal(0, rms*0.5), "S_err": rms,
                "bmaj": BEAM_MAJ*(1+rng.normal(0,0.03)),
                "bmaj_err": BEAM_MAJ*0.02,
                "bmin": BEAM_MIN*(1+rng.normal(0,0.03)),
                "bmin_err": BEAM_MIN*0.02,
                "pa": BEAM_PA + rng.normal(0,2.0), "pa_err": 2.0,
                "rms": rms,
            })

    # Sidelobes
    brightest = max(spots, key=lambda s: s["I"])
    for _ in range(2):
        sep = rng.uniform(1.0,2.5) * BEAM_MAJ
        angle = rng.uniform(0, 2*np.pi)
        spots.append({
            "vlsr": brightest["vlsr"],
            "x": brightest["x"] + sep*np.cos(angle),
            "y": brightest["y"] + sep*np.sin(angle),
            "x_err": 0.1, "y_err": 0.1,
            "I": brightest["I"]*rng.uniform(0.08,0.20), "I_err": 0.05,
            "S": brightest["I"]*rng.uniform(0.08,0.20), "S_err": 0.05,
            "bmaj": BEAM_MAJ*rng.uniform(1.3,2.0),
            "bmaj_err": BEAM_MAJ*0.05,
            "bmin": BEAM_MIN*rng.uniform(0.5,0.8),
            "bmin_err": BEAM_MIN*0.05,
            "pa": BEAM_PA + rng.normal(0,15), "pa_err": 5.0,
            "rms": 0.05,
        })
    spots.sort(key=lambda s: s["vlsr"])
    return spots

# =====================================================================
#  WRITE FILES
# =====================================================================
def write_csad(spots, path, mode):
    cols = ["vlsr","x","x_err","y","y_err","I","I_err","S","S_err",
            "bmaj","bmaj_err","bmin","bmin_err","pa","pa_err","rms"]
    with open(path, "w") as f:
        f.write(f"# csad output (synthetic — {mode} PR)\n")
        f.write(f"# Beam: {BEAM_MAJ:.2f} x {BEAM_MIN:.2f} mas, PA = {BEAM_PA:.1f} deg\n")
        f.write(f"# Channel spacing: {CHAN_SPACING:.3f} km/s\n")
        f.write(f"# N_stations: {N_STATIONS}\n")
        f.write(f"# {'  '.join(f'{c:>12s}' for c in cols)}\n")
        for s in spots:
            f.write("  "+"  ".join(f"{s[c]:12.4f}" for c in cols)+"\n")

def compute_inverse_shifts():
    ref0_x, ref0_y = get_ref_position(0)
    shifts = {}
    for ei, ep in enumerate(EPOCHS):
        rx, ry = get_ref_position(ei)
        shift_ra = rx - ref0_x
        shift_dec = ry - ref0_y
        cos_dec = np.cos(DEC_RAD)
        qso_ra_s = QSO_RA_S + (-(rx)) / (15.0 * cos_dec * 1000.0)
        qso_dec_s = QSO_DEC_S + (-(ry)) / 1000.0
        shifts[ep["code"]] = (shift_ra, shift_dec, qso_ra_s, qso_dec_s)
    return shifts

def write_epoch_table(path, inv_shifts):
    with open(path, "w") as f:
        f.write("# Epoch table for masertrack tutorial\n")
        f.write("# code  date  mjd  dec_year  n_sta  inv_shift_ra  inv_shift_dec  inv_qso_ra_s  inv_qso_dec_s\n")
        f.write(f"# Source: RA={RA_STR}  Dec={DEC_STR}  pi={PI_TRUE:.3f} mas\n")
        f.write(f"# Inverse PR ref: F1 at V_LSR=-35.0 km/s\n#\n")
        for e in EPOCHS:
            sr,sd,qr,qd = inv_shifts[e["code"]]
            f.write(f"  {e['code']}  {e['date']}  {e['mjd']}  "
                    f"{e['dec_year']:.4f}  {e['n_sta']}"
                    f"  {sr:12.4f}  {sd:12.4f}  {qr:16.8f}  {qd:12.6f}\n")

def write_truth(path, inv_shifts):
    with open(path, "w") as f:
        f.write("# Ground truth for tutorial verification\n#\n")
        f.write(f"# Parallax: {PI_TRUE:.3f} mas  (D = {1/PI_TRUE:.3f} kpc)\n")
        f.write(f"# PM(RA):   {MU_X_TRUE:+.3f} mas/yr\n")
        f.write(f"# PM(Dec):  {MU_Y_TRUE:+.3f} mas/yr\n#\n")
        f.write(f"# Both PR modes should recover the same parallax.\n")
        f.write(f"# Normal PR: positions are absolute -> fit directly\n")
        f.write(f"# Inverse PR: positions are relative to reference maser;\n")
        f.write(f"#   quasar shifts in epoch table restore absolute motion\n#\n")
        f.write(f"# Features:\n")
        for feat in FEATURES:
            ep_str = ",".join(EPOCHS[i]["code"] for i in feat["present_epochs"])
            grade = "A" if len(feat["present_epochs"])==6 else "B"
            ref = " [INV REF]" if feat.get("is_reference") else ""
            f.write(f"#  F{feat['id']}: V={feat['vlsr_center']:.0f} km/s, "
                    f"{feat['n_spots']} spots, {feat['flux_peak']:.0f} Jy, "
                    f"Grade {grade}{ref} ({ep_str})\n")

# =====================================================================
#  MAIN
# =====================================================================
def main():
    outdir = Path(__file__).parent / "input"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Generating tutorial data for masertrack v1.1")
    print(f"  pi={PI_TRUE:.3f} mas, PM=({MU_X_TRUE:+.2f},{MU_Y_TRUE:+.2f}) mas/yr")
    print(f"  {len(EPOCHS)} epochs, {len(FEATURES)} features\n")

    # Normal PR (seed=42)
    rng_n = np.random.default_rng(42)
    print("  NORMAL PR:")
    for ei, ep in enumerate(EPOCHS):
        spots = generate_spots(ei, rng_n, "normal")
        write_csad(spots, outdir / f"{ep['code']}_normal.txt", "normal")
        nf = sum(1 for f in FEATURES if ei in f["present_epochs"])
        print(f"    {ep['code']}: {len(spots)} spots ({nf} feat + 2 sidelobes)")

    # Inverse PR (seed=43)
    rng_i = np.random.default_rng(43)
    print("\n  INVERSE PR:")
    for ei, ep in enumerate(EPOCHS):
        spots = generate_spots(ei, rng_i, "inverse")
        write_csad(spots, outdir / f"{ep['code']}_inverse.txt", "inverse")
        nf = sum(1 for f in FEATURES if ei in f["present_epochs"])
        ref = next((s for s in spots if abs(s["vlsr"]-(-35.0))<0.01 and s["I"]>5),None)
        rs = f"  ref@({ref['x']:+.3f},{ref['y']:+.3f})" if ref else ""
        print(f"    {ep['code']}: {len(spots)} spots{rs}")

    inv_shifts = compute_inverse_shifts()
    write_epoch_table(Path(__file__).parent / "epoch_table.txt", inv_shifts)
    write_truth(Path(__file__).parent / "ground_truth.txt", inv_shifts)
    print(f"\n  Epoch table and ground truth written.")
    print(f"\n  Inverse PR shifts:")
    for e in EPOCHS:
        sr,sd,_,_ = inv_shifts[e["code"]]
        print(f"    {e['code']}: dRA={sr:+.4f}, dDec={sd:+.4f} mas")
    print(f"\nDone. Files in {outdir}/")

if __name__ == "__main__":
    main()
