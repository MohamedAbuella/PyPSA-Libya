#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_carrier_energy_conservation.py
=======================================

Multi-carrier ENERGY CONSERVATION & NODAL BALANCE validation for solved
PyPSA / PyPSA-Earth networks (.nc). Iterates through all network carriers
and automatically loops over multiple predefined scenarios/study cases,
saving individual text reports, nodal dispatch maps, and stacked bar charts.
"""

import os
import sys
import argparse
import logging
import warnings
import re

import numpy as np
import pandas as pd
import pypsa

warnings.filterwarnings("ignore")
logging.getLogger("pypsa").setLevel(logging.ERROR)

# ----------------------------------------------------------------------------
# USER SCENARIOS DEFINITION (Automated Batch Loop)
# ----------------------------------------------------------------------------

OUTPUT_DIR_BASE = r"C:\Users\Mhdella\Documents\pypsa-libya\viz_outputs\Validate_Energy_Conservation_All_Carriers"

# --- PAPER PDF SUPPORT FOLDER ---
PAPER_DIR = os.path.join(OUTPUT_DIR_BASE, "Paper_Figures")
os.makedirs(PAPER_DIR, exist_ok=True)

def save_paper_pdf(fig, ax, filename):
    """
    Automatically clears the title and saves a high-res PDF to the Paper_Figures folder.
    Optimizes map file sizes by rasterizing geographic layers while keeping text as vectors.
    """
    original_title = ax.get_title() # Store the title
    ax.set_title("")                # Clear the title for the paper version
    
    # Apply naming logic: start with 'fig_' and ensure .pdf extension
    clean_name = re.sub(r'^figure\d+[a-z]?_', '', filename)
    if not clean_name.startswith('fig_'):
        clean_name = 'fig_' + clean_name
    if not clean_name.endswith('.pdf'):
        clean_name += '.pdf'
        
    pdf_path = os.path.join(PAPER_DIR, clean_name)
    
    # --- OPTIMIZATION FOR MAPS ---
    # Rasterize layers with low z-orders (map background/lines) while keeping bubbles/text vector
    if "map" in filename.lower():
        ax.set_rasterization_zorder(2.5) 
    
    # Save with dpi=300 for high-resolution journal quality
    fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches="tight")
    
    ax.set_title(original_title)    # Restore title for the PNG/display version
    print(f" Saved Paper PDF (Optimized): {clean_name}")

# ============================================================================
# ADJUSTABLE CONFIGURATION: MAP & BAR CHART FORMATTING FOR READABILITY
# ============================================================================

### --- 1. General Scaling & Filtering Thresholds ---
PIE_SCALE_FACTOR = 1.5                # Scales pie chart sizes globally 
                                      # (1.0 = baseline size, 1.5 = +50% larger, etc.)

CARRIER_PERCENTAGE = 0.5              # Minimum global energy share (%) required for a 
                                      # carrier to appear in map pie charts and the legend

### --- 2. Map Node Label Styling & Positioning ---
MAP_NODE_LABEL_SIZE = 11              # Font size for rank numbers placed next to nodes (e.g., 9, 11, 13)
MAP_NODE_LABEL_WEIGHT = "bold"        # Font weight for node labels ("bold" or "normal")

# Label coordinate offsets (adjusts position relative to node coordinates)
MAP_LABEL_OFFSET_X = -0.35            # Horizontal offset (used primarily in electricity-only mode)
MAP_LABEL_OFFSET_Y = -0.15            # Vertical offset (used primarily in electricity-only mode)
MAP_PIXEL_OFFSET_Y = 7                # Vertical pixel shift above nodes (used in sector-coupling mode)

### --- 3. Axis & Tick Typography Controls ---
# Map Figure Axis Settings
MAP_AXIS_LABEL_FONT_SIZE = 14         # Font size for map axis titles (Longitude / Latitude)
MAP_AXIS_TICK_FONT_SIZE = 10          # Font size for map tick values along the axes
MAP_LEGEND_FONT_SIZE = 14             # Font size for the map legend text

# Bar Chart Figure Axis Settings
BAR_LABEL_FONT_SIZE = 16              # Font size for bar chart axis titles (Node Rank / Energy Dispatch)
BAR_TICK_FONT_SIZE = 18               # Font size for bar chart tick marks and values
BAR_LEGEND_FONT_SIZE = 14             # Font size for the bar chart legend text (increased for better readability)


SCENARIOS = [
    {
        "name": "LCO2 (Electricity-only)",
        "prefix": "LCO2",
        "nc_path": r"Output\libya-year-36\LCO2\3h\networks\elec_s_36_ec_lcopt_Co2L-3h.nc",
    },
    {
        "name": "No_LCO2 (Electricity-only)",
        "prefix": "No_LCO2",
        "nc_path": r"Output\libya-year-36\No_LCO2\3h\networks\elec_s_36_ec_lcopt_Co2L-3h.nc",
    },
    {
        "name": "Scenario A (Sector-Coupling)",
        "prefix": "Scenario_a",
        "nc_path": r"Output\libya_h2export_test\scenario_a\3hrs\postnetworks\elec_s_12_ec_lcopt_Co2L-3h__2030_0.071_AB_70export.nc",
    },
    {
        "name": "Scenario B (Sector-Coupling)",
        "prefix": "Scenario_b",
        "nc_path": r"Output\libya_h2export_test\scenario_b\3hrs\postnetworks\elec_s_12_ec_lcopt_Co2L-3h__2030_0.071_AB_70export.nc",
    },
    {
        "name": "Scenario C (Sector-Coupling)",
        "prefix": "Scenario_c",
        "nc_path": r"Output\libya_h2export_test\scenario_c\3hrs\postnetworks\elec_s_12_ec_lcopt_Co2L-3h__2030_0.071_AB_70export.nc",
    },
    {
        "name": "Hybrid Pathway",
        "prefix": "Hybrid",
        "nc_path": r"Output/libya_h2export_hybrid_emispr_100/3hrs/postnetworks/orig_elec_s_12_ec_lcopt_Co2L-Ep-3h__2030_0.071_AB_70export.nc",
    }
]

# Desired presentation and iteration order for all network carriers
CARRIER = [
    "AC",
    "low voltage",
    "home battery",
    "battery",
    "gas",
    "gas for industry",
    "H2",
    "H2 export",
    "Ammonia NH3",
    "solid biomass for industry",
    "co2",
    "co2 stored",
    "process emissions"
]

TOL_MW = 0.01          # max |nodal residual| accepted as numerical noise [MW]
REL_TOL = 1e-6         # accepted relative energy imbalance (of carrier throughput)

# Bus carriers whose "energy" is really a MASS (1 MWh in PyPSA == 1 tonne)
MASS_CARRIERS = {"co2", "co2 stored", "co2 atmosphere", "process emissions"}


# ----------------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------------
def fmt_energy(mwh, mass=False):
    """MWh -> MWh / GWh / TWh (>=1000 GWh -> TWh). Mass carriers: t / kt / Mt."""
    a, s = abs(mwh), ("-" if mwh < 0 else "")
    if mass:
        if a >= 1e6:
            return f"{s}{a / 1e6:,.3f} Mt"
        if a >= 1e3:
            return f"{s}{a / 1e3:,.3f} kt"
        return f"{s}{a:,.3f} t"
    if a >= 1e6:
        return f"{s}{a / 1e6:,.3f} TWh"
    if a >= 1e3:
        return f"{s}{a / 1e3:,.3f} GWh"
    return f"{s}{a:,.3f} MWh"


def fmt_power(mw):
    a = abs(mw)
    s = "-" if mw < 0 else ""
    if a >= 1e3:
        return f"{s}{a / 1e3:,.3f} GW"
    return f"{s}{a:,.4f} MW"


# ----------------------------------------------------------------------------
# Data access helpers (robust across PyPSA versions)
# ----------------------------------------------------------------------------
def dyn(comp_t, attr, index):
    """Return dynamic dataframe (snapshots x components) or an empty frame."""
    df = getattr(comp_t, attr, None)
    if df is None:
        return pd.DataFrame(index=index)
    return df


def load_power(n):
    """loads_t.p if stored, else rebuilt from p_set (dynamic + static)."""
    p = dyn(n.loads_t, "p", n.snapshots)
    if not p.empty and set(n.loads.index) <= set(p.columns):
        return p[n.loads.index]
    out = pd.DataFrame(0.0, index=n.snapshots, columns=n.loads.index)
    static = n.loads["p_set"] if "p_set" in n.loads else 0.0
    for c in n.loads.index:
        out[c] = static[c] if np.isscalar(static) is False else static
    ps = dyn(n.loads_t, "p_set", n.snapshots)
    for c in ps.columns:
        out[c] = ps[c]
    return out


def link_terminals(n):
    """Indices i of the available link terminals (bus0, bus1, bus2, ...)."""
    t = []
    for i in range(0, 10):
        if f"bus{i}" in n.links.columns:
            col = n.links[f"bus{i}"]
            if (col.astype(str).str.len() > 0).any():
                t.append(i)
    return t


def sum_by_bus(df, bus_of):
    """Sum a (snapshots x components) frame to (snapshots x buses)."""
    cols = [c for c in df.columns if c in bus_of.index]
    if not cols:
        return pd.DataFrame(index=df.index)
    return df[cols].T.groupby(bus_of.loc[cols].values).sum().T


# ----------------------------------------------------------------------------
# Main validation function for a single carrier
# ----------------------------------------------------------------------------
def validate_carrier(n, carrier="AC", tol_mw=TOL_MW, rel_tol=REL_TOL):
    rep = []                      # report lines
    flags = []                    # (status, text) for the final verdict

    def out(s=""):
        rep.append(s)

    snaps = n.snapshots
    w = n.snapshot_weightings.generators

    bus_carrier = n.buses["carrier"]
    available_carriers = list(bus_carrier.unique())
    matching_carriers = [c for c in available_carriers if str(c).lower() == str(carrier).lower()]
    
    if not matching_carriers:
        return f"\n{'='*100}\n CARRIER NOT FOUND IN NETWORK: {carrier.upper()}\n{'='*100}\n"
    
    actual_carrier = matching_carriers[0]
    mass = actual_carrier.lower() in MASS_CARRIERS
    F = lambda x: fmt_energy(x, mass)

    buses = list(n.buses.index[bus_carrier == actual_carrier])
    bus_set = set(buses)

    step_h = (snaps[1] - snaps[0]).total_seconds() / 3600.0 if len(snaps) > 1 else float("nan")
    window_h = step_h * len(snaps) if np.isfinite(step_h) else float("nan")
    out("=" * 100)
    out(f" ENERGY CONSERVATION & NODAL BALANCE VALIDATION  --  CARRIER: {actual_carrier.upper()}")
    out("=" * 100)
    out(f" PyPSA version   : {pypsa.__version__}")
    out(f" Snapshots       : {len(snaps)}  ({snaps[0]} -> {snaps[-1]}), step = {step_h:g} h")
    out(f" Weighting       : sum of weights = {w.sum():,.1f} h  "
        f"(per snapshot: {w.min():,.2f} - {w.max():,.2f} h)")
    if np.isfinite(window_h) and abs(w.sum() - window_h) > 1:
        out(f" ANNUALISATION   : simulated window = {window_h:,.0f} h  ->  scaled x{w.sum() / window_h:,.2f} "
            f"to {w.sum():,.0f} h (all energies below are annualised)")
    out(f" Buses of carrier: {len(buses)}  ->  {buses[0]} ... {buses[-1]}")
    if "objective" in dir(n) and getattr(n, "objective", None) is not None:
        out(f" Objective       : {n.objective / 1e9:,.4f} bn EUR/yr")
    out(f" Unit mode       : {'MASS (t / kt / Mt)' if mass else 'ENERGY (MWh / GWh / TWh)'}")

    zeros = lambda: pd.DataFrame(0.0, index=snaps, columns=buses)
    groups = {k: zeros() for k in
              ["Generators", "Loads", "Cross-carrier links", "Stores", "Storage units", "Transfers (lines/internal links)"]}
    records = []

    def register(inj, bus_of, label_of, category, group):
        cols = [c for c in inj.columns if c in bus_of.index and bus_of[c] in bus_set]
        if not cols:
            return
        sub = inj[cols]
        e_sup = sub.clip(lower=0).mul(w, axis=0).sum()
        e_dem = (-sub).clip(lower=0).mul(w, axis=0).sum()
        records.append(pd.DataFrame({
            "category": category, "label": label_of.loc[cols].values,
            "bus": bus_of.loc[cols].values, "sup": e_sup.values, "dem": e_dem.values}))
        by_bus = sum_by_bus(sub, bus_of.loc[cols])
        groups[group][by_bus.columns] += by_bus

    # --- generators
    gp = dyn(n.generators_t, "p", snaps)
    if not gp.empty:
        register(gp, n.generators["bus"], "Generator: " + n.generators["carrier"].astype(str),
                 "Generator", "Generators")

    # --- loads
    lp = load_power(n)
    lbus = n.loads["bus"]
    undef = n.loads[~lbus.isin(n.buses.index)]
    register(-lp, lbus, "Load: " + n.loads["carrier"].astype(str), "Load", "Loads")

    # --- stores / storage units
    sp = dyn(n.stores_t, "p", snaps)
    if not sp.empty:
        register(sp, n.stores["bus"], "Store: " + n.stores["carrier"].astype(str), "Store", "Stores")
    sup_ = dyn(n.storage_units_t, "p", snaps)
    if not sup_.empty:
        register(sup_, n.storage_units["bus"], "StorageUnit: " + n.storage_units["carrier"].astype(str),
                 "StorageUnit", "Storage units")

    # --- links
    terms = link_terminals(n)
    internal_links = set()
    for l in n.links.index:
        tb = [n.links.at[l, f"bus{i}"] for i in terms
              if isinstance(n.links.at[l, f"bus{i}"], str) and n.links.at[l, f"bus{i}"] in n.buses.index]
        if tb and all(b in bus_set for b in tb) and len(tb) >= 2:
            internal_links.add(l)

    for i in terms:
        pi = dyn(n.links_t, f"p{i}", snaps)
        if pi.empty:
            continue
        bus_i = n.links[f"bus{i}"]
        ext = [c for c in pi.columns if c not in internal_links]
        reg = pi[ext]
        register(-reg, bus_i, "Link: " + n.links["carrier"].astype(str), "Link", "Cross-carrier links")
        intl = [c for c in pi.columns if c in internal_links]
        if intl:
            sub = -pi[intl]
            sub = sub[[c for c in sub.columns if bus_i[c] in bus_set]]
            if not sub.empty:
                by_bus = sum_by_bus(sub, bus_i.loc[sub.columns])
                groups["Transfers (lines/internal links)"][by_bus.columns] += by_bus

    # --- lines
    transfer_loss_series = pd.Series(0.0, index=snaps)
    if len(n.lines):
        for i in (0, 1):
            pl = dyn(n.lines_t, f"p{i}", snaps)
            if pl.empty:
                continue
            bus_l = n.lines[f"bus{i}"]
            cols = [c for c in pl.columns if bus_l[c] in bus_set]
            if cols:
                by_bus = sum_by_bus(-pl[cols], bus_l.loc[cols])
                groups["Transfers (lines/internal links)"][by_bus.columns] += by_bus
    transfers = groups["Transfers (lines/internal links)"]
    transfer_loss_series = -transfers.sum(axis=1)
    transfer_loss_E = float(transfer_loss_series.mul(w, axis=0).sum())

    rec = pd.concat(records, ignore_index=True) if records else pd.DataFrame(
        columns=["category", "label", "bus", "sup", "dem"])

    # ===================================================== 1. NODAL BALANCE
    inj_total = sum(groups.values())
    abs_res = inj_total.abs()
    max_res = float(abs_res.values.max())
    max_idx = np.unravel_index(abs_res.values.argmax(), abs_res.shape)
    worst_bus, worst_snap = abs_res.columns[max_idx[1]], abs_res.index[max_idx[0]]
    res_E = abs_res.mul(w, axis=0).sum()
    throughput = float(sum(rec["sup"]) + sum(rec["dem"])) / 2 if len(rec) else 1.0

    out("\n" + "#" * 100)
    out(" 1. NODAL BALANCE (Kirchhoff current law per bus and per snapshot)")
    out("#" * 100)
    out(f" Tolerance (max |residual|) : {tol_mw} MW      Relative tolerance : {rel_tol:g}")
    out(f" Max |residual|             : {fmt_power(max_res)}  at bus '{worst_bus}', snapshot {worst_snap}")
    out(f" Sum |residual| energy      : {F(float(res_E.sum()))}  "
        f"({float(res_E.sum()) / max(throughput, 1e-9):.2e} of carrier throughput)")
    ok_nodal = max_res <= tol_mw
    flags.append(("PASS" if ok_nodal else "FAIL", f"Nodal balance: max |residual| = {fmt_power(max_res)}"))
    out(f" RESULT                     : {'PASSED' if ok_nodal else 'FAILED'}")

    out("\n Per-bus ledger (annualised, signed:  + into bus / - out of bus):")
    node = pd.DataFrame(index=buses)
    for k, df in groups.items():
        node[k] = df.mul(w, axis=0).sum()
    node["Residual"] = node.sum(axis=1)
    node["max|res| MW"] = abs_res.max()
    show = node.copy()
    for c in node.columns[:-1]:
        show[c] = node[c].map(F)
    show["max|res| MW"] = node["max|res| MW"].map(lambda x: f"{x:.2e}")
    out(show.to_string())
    tot = node.sum(numeric_only=True)
    out("-" * 100)
    out(" TOTAL".ljust(14) + " | " + " | ".join(f"{c}: {F(tot[c])}" for c in node.columns[:-1]))

    # ============================================ 2. CARRIER ENERGY BALANCE
    out("\n" + "#" * 100)
    out(f" 2. CARRIER-LEVEL ENERGY BALANCE  ({actual_carrier.upper()}, all {len(buses)} buses aggregated)")
    out("#" * 100)
    agg = rec.groupby("label")[["sup", "dem"]].sum()
    sup = agg["sup"][agg["sup"] > 1e-6].sort_values(ascending=False)
    dem = agg["dem"][agg["dem"] > 1e-6].sort_values(ascending=False)
    dem_items = dem.to_dict()
    if transfer_loss_E > 1e-6:
        dem_items["Transmission losses (lines + internal links)"] = transfer_loss_E
    dem_items = dict(sorted(dem_items.items(), key=lambda kv: kv[1], reverse=True))
    tot_s, tot_d = float(sup.sum()), float(sum(dem_items.values()))
    net = tot_s - tot_d
    pct = abs(net) / max(tot_s, tot_d, 1e-9) * 100

    out(f" {'SUPPLY':<60}{'Energy':>18}{'Share':>10}")
    out(" " + "-" * 88)
    for k, v in sup.items():
        out(f"   {k:<58}{F(v):>18}{v / tot_s * 100:>9.2f}%")
    out(f" {'TOTAL SUPPLY':<60}{F(tot_s):>18}")
    out(f"\n {'DEMAND':<60}{'Energy':>18}{'Share':>10}")
    out(" " + "-" * 88)
    for k, v in dem_items.items():
        out(f"   {k:<58}{F(v):>18}{v / tot_d * 100:>9.2f}%")
    out(f" {'TOTAL DEMAND':<60}{F(tot_d):>18}")
    out("\n " + "=" * 88)
    out(f" NET (Supply - Demand) : {F(net)}   ({pct:.3e} % of the larger side)")
    ok_bal = pct <= rel_tol * 100 or abs(net) <= max(tol_mw * w.sum() * 1e-3, 1e-3)
    flags.append(("PASS" if ok_bal else "FAIL", f"Carrier balance: net = {F(net)} ({pct:.2e} %)"))
    out(f" RESULT                : {'PASSED - supply equals demand' if ok_bal else 'FAILED - supply does not equal demand'}")

    # ============================================== 3. PER-SNAPSHOT CLOSURE
    out("\n" + "#" * 100)
    out(" 3. PER-SNAPSHOT SYSTEM CLOSURE (sum over all buses of the carrier)")
    out("#" * 100)
    sys_res = inj_total.sum(axis=1)
    worst = sys_res.abs().idxmax()
    out(f" Max |system residual| : {fmt_power(float(sys_res.abs().max()))} at {worst}")
    out(f" Mean |system residual|: {fmt_power(float(sys_res.abs().mean()))}")
    flags.append(("PASS" if sys_res.abs().max() <= tol_mw else "FAIL",
                  f"Per-snapshot closure: max |res| = {fmt_power(float(sys_res.abs().max()))}"))
    sup_t = pd.Series(0.0, index=snaps)
    dem_t = pd.Series(0.0, index=snaps)
    for g in ["Generators", "Loads", "Cross-carrier links", "Stores", "Storage units"]:
        sup_t += groups[g].clip(lower=0).sum(axis=1)
        dem_t += (-groups[g]).clip(lower=0).sum(axis=1)
    dem_t += transfer_loss_series
    out(f" Peak supply   : {fmt_power(float(sup_t.max()))} at {sup_t.idxmax()}")
    out(f" Peak demand   : {fmt_power(float(dem_t.max()))} at {dem_t.idxmax()}")
    out(f" Min  demand   : {fmt_power(float(dem_t.min()))} at {dem_t.idxmin()}")

    # ============================================== 4. TRANSMISSION AUDIT
    out("\n" + "#" * 100)
    out(" 4. TRANSMISSION AUDIT (lines and internal transport links)")
    out("#" * 100)
    n_lines_in = 0
    if len(n.lines):
        p0, p1 = dyn(n.lines_t, "p0", snaps), dyn(n.lines_t, "p1", snaps)
        sel = [l for l in n.lines.index if n.lines.at[l, "bus0"] in bus_set and n.lines.at[l, "bus1"] in bus_set]
        n_lines_in = len(sel)
        if sel and not p0.empty:
            loss_t = (p0[sel] + p1[sel])
            send = p0[sel].clip(lower=0).mul(w, axis=0).sum().sum() + p1[sel].clip(lower=0).mul(w, axis=0).sum().sum()
            loss_E = float(loss_t.mul(w, axis=0).sum().sum())
            neg_loss = float((loss_t < -1e-6).sum().sum())
            out(f" Lines inside carrier   : {n_lines_in}")
            out(f" Gross energy sent      : {F(float(send))}")
            out(f" Line losses (p0 + p1)  : {F(loss_E)}  ({loss_E / max(float(send), 1e-9) * 100:.3f} % of gross sent)")
            out(f" Snapshots with negative loss (should be 0): {int(neg_loss)}")
            flags.append(("PASS" if neg_loss == 0 else "WARN", f"Line losses non-negative (violations: {int(neg_loss)})"))
            top = (loss_t.mul(w, axis=0).sum()).sort_values(ascending=False).head(5)
            out("\n Top lossy lines:")
            for l, v in top.items():
                out(f"   line {l:<10} {n.lines.at[l, 'bus0']:>10} <-> {n.lines.at[l, 'bus1']:<10} loss {F(float(v))}")
    out(f" Internal transport links (same-carrier at all terminals): {len(internal_links)}")
    out(f" TOTAL transmission losses entering the balance: {F(transfer_loss_E)}")

    # ============================================ 5. CONVERSION-LINK AUDIT
    out("\n" + "#" * 100)
    out(" 5. CONVERSION-LINK AUDIT (links touching this carrier)")
    out("#" * 100)
    touch = {}
    for l in n.links.index:
        if l in internal_links:
            continue
        for i in terms:
            b = n.links.at[l, f"bus{i}"]
            if isinstance(b, str) and b in bus_set:
                touch.setdefault(n.links.at[l, "carrier"], set()).add(l)
    p0df = dyn(n.links_t, "p0", snaps)
    audit_rows, max_eff_err = [], 0.0
    for lc in sorted(touch):
        ids = sorted(touch[lc])
        e0 = float(p0df[ids].mul(w, axis=0).sum().sum()) if not p0df.empty else 0.0
        for i in terms:
            if i == 0:
                continue
            ok_ids = [l for l in ids if isinstance(n.links.at[l, f"bus{i}"], str)
                      and n.links.at[l, f"bus{i}"] in n.buses.index]
            if not ok_ids:
                continue
            pi = dyn(n.links_t, f"p{i}", snaps)
            if pi.empty or not set(ok_ids) <= set(pi.columns):
                continue
            ename = "efficiency" if i == 1 else f"efficiency{i}"
            tdyn = dyn(n.links_t, ename, snaps)
            eff = pd.DataFrame(
                {l: (tdyn[l] if l in tdyn.columns else
                     (n.links.at[l, ename] if ename in n.links.columns else 1.0)) for l in ok_ids},
                index=snaps).astype(float).fillna(1.0)
            err = float((pi[ok_ids] + p0df[ok_ids] * eff).abs().max().max())
            max_eff_err = max(max_eff_err, err)
            ei = float(-pi[ok_ids].mul(w, axis=0).sum().sum())
            tcar = n.buses.loc[n.links.loc[ok_ids, f"bus{i}"], "carrier"].mode().iat[0]
            audit_rows.append({
                "link carrier": lc, "#": len(ids), "terminal": f"bus{i}",
                "terminal carrier": tcar,
                "input @bus0": F(e0), "output @terminal": fmt_energy(ei, tcar.lower() in MASS_CARRIERS),
                "eff (min-max)": f"{eff.min().min():.4f}-{eff.max().max():.4f}",
                "max|p_i+eff*p0| MW": f"{err:.2e}"})
    if audit_rows:
        out(pd.DataFrame(audit_rows).to_string(index=False))
    ok_eff = max_eff_err <= max(tol_mw, 1e-6)
    out(f"\n Max |p_i + eff_i*p0| over all audited terminals: {max_eff_err:.3e} MW")
    flags.append(("PASS" if ok_eff else "WARN", f"Link efficiency consistency: max err {max_eff_err:.2e} MW"))

    # ============================================ 6. CARRIER ELEMENTS DISTRIBUTION PER NODE
    out("\n" + "#" * 100)
    out(" 6. CARRIER ELEMENTS DISTRIBUTION PER NODE")
    out("#" * 100)
    if not rec.empty:
        rec_nodal = rec.copy()
        rec_nodal["net"] = rec_nodal["sup"] - rec_nodal["dem"]
        pivot_elem = rec_nodal.pivot_table(index="bus", columns="label", values="net", fill_value=0.0)
        show_elem = pivot_elem.copy()
        for col in show_elem.columns:
            show_elem[col] = show_elem[col].map(lambda x: F(x))
        out(show_elem.to_string())
    else:
        out(" No discrete carrier elements registered for this carrier.")

    # ======================================================= 7. SANITY FLAGS
    out("\n" + "#" * 100)
    out(" 7. SANITY FLAGS")
    out("#" * 100)
    shed = rec[rec["label"].str.contains("load shedding", case=False, na=False)]["sup"].sum()
    load_total = rec[rec["label"].str.startswith("Load:")]["dem"].sum()
    out(f" Load shedding energy       : {F(float(shed))}  "
        f"({float(shed) / max(load_total, 1e-9) * 100:.4f} % of loads on this carrier)")
    flags.append(("PASS" if shed <= 1e-6 * max(load_total, 1) else "WARN",
                  f"Load shedding: {F(float(shed))}"))
    neg_gen = 0.0
    if not gp.empty:
        gcols = [g for g in gp.columns if n.generators.at[g, "bus"] in bus_set]
        neg_gen = float((-gp[gcols]).clip(lower=0).mul(w, axis=0).sum().sum())
    out(f" Negative generator dispatch: {F(neg_gen)}")
    try:
        pmax = n.get_switchable_as_dense("Generator", "p_max_pu")
        gens = [g for g in n.generators.index if n.generators.at[g, "bus"] in bus_set
                and n.generators.at[g, "carrier"] in ("solar", "onwind", "offwind", "solar rooftop", "ror", "hydro")]
        if gens and not gp.empty:
            cap = n.generators.loc[gens, "p_nom_opt"] if "p_nom_opt" in n.generators else n.generators.loc[gens, "p_nom"]
            pot = pmax[gens].mul(cap, axis=1)
            cur = (pot - gp[gens]).clip(lower=0)
            out(" Renewable curtailment    :")
            for gc, ids in n.generators.loc[gens].groupby("carrier").groups.items():
                ids = list(ids)
                pot_e = float(pot[ids].mul(w, axis=0).sum().sum())
                cur_e = float(cur[ids].mul(w, axis=0).sum().sum())
                out(f"    {gc:<14} potential {F(pot_e):>14} | used {F(pot_e - cur_e):>14} | "
                    f"curtailed {F(cur_e):>14} ({cur_e / max(pot_e, 1e-9) * 100:.2f} %)")
    except Exception:
        out(" Renewable curtailment    : (skipped - p_max_pu not available)")
    if len(undef):
        e_undef = float(lp[undef.index].mul(w, axis=0).sum().sum())
        by_c = undef.groupby("carrier").apply(lambda d: float(lp[d.index].mul(w, axis=0).sum().sum()))
        out(f" WARNING: {len(undef)} load(s) are attached to buses that DO NOT EXIST "
            f"(their demand is IGNORED by the optimisation): {F(e_undef)}")
        for c, v in by_c.items():
            out(f"    - {c}: {F(float(v))}")
        flags.append(("WARN", f"{len(undef)} loads on undefined buses ({F(e_undef)}) are silently ignored"))
    else:
        out(" Loads on undefined buses  : none")

    # ===================================================== final verdict
    out("\n" + "=" * 100)
    out(" VERDICT")
    out("=" * 100)
    for st, tx in flags:
        out(f"  [{st:^4}] {tx}")
    overall = "PASSED" if all(s != "FAIL" for s, _ in flags) else "FAILED"
    out(f"\n  OVERALL ENERGY CONSERVATION FOR '{actual_carrier.upper()}': {overall}"
        + ("  (with warnings)" if overall == "PASSED" and any(s == "WARN" for s, _ in flags) else ""))
    out("=" * 100)
    return "\n".join(rep)


# =============================================================================
# MAIN AUTOMATION BATCH LOOP OVER ALL SCENARIOS
# =============================================================================
if __name__ == "__main__":
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        HAS_CARTOPY = True
    except ImportError:
        HAS_CARTOPY = False
        print("Warning: Cartopy not installed. Nodal dispatch map will use standard matplotlib axes.")

    for sc in SCENARIOS:
        # Automatically construct paths using the prefix and base directory
        txt_out = os.path.join(OUTPUT_DIR_BASE, f"{sc['prefix']}_energy_conservation_all_carriers.txt")
        png_out = os.path.join(OUTPUT_DIR_BASE, f"{sc['prefix']}_pypsa_libya_nodal_dispatch_map.png")
        bar_out = os.path.join(OUTPUT_DIR_BASE, f"{sc['prefix']}_pypsa_libya_nodal_bar_chart.png")

        print("\n" + "=" * 110)
        print(f"PROCESSING SCENARIO: {sc['name']}")
        print(f"File path: {sc['nc_path']}")
        print("=" * 110)

        if not os.path.exists(sc["nc_path"]):
            print(f"ERROR: Network file not found: {sc['nc_path']}. Skipping scenario...")
            continue

        print(f"Loading network file...")
        n = pypsa.Network(sc["nc_path"])

        available_carriers = list(n.buses["carrier"].unique())
        carriers_to_run = [c for c in CARRIER if any(str(ac).lower() == str(c).lower() for ac in available_carriers)]
        for ac in available_carriers:
            if not any(str(ac).lower() == str(c).lower() for c in carriers_to_run):
                carriers_to_run.append(ac)

        # 1. Run Energy Conservation Validation across all carriers
        all_reports = []
        for c in carriers_to_run:
            print(f"  -> Validating carrier: {c} ...")
            try:
                report_text = validate_carrier(n, carrier=c, tol_mw=TOL_MW, rel_tol=REL_TOL)
                all_reports.append(report_text)
            except Exception as e:
                err_msg = f"\nERROR validating carrier '{c}': {e}\n"
                print(err_msg)
                all_reports.append(err_msg)

        separator = "\n\n" + "=" * 100 + "\n" + " " * 35 + "NEXT CARRIER REPORT SECTION" + "\n" + "=" * 100 + "\n\n"
        combined_content = separator.join(all_reports)

        # Ensure text report output directory exists
        txt_dir = os.path.dirname(txt_out)
        if txt_dir:
            os.makedirs(txt_dir, exist_ok=True)

        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(combined_content)
        print(f"Saved text validation report to: {txt_out}")
        
        # Also copy/save text validation report to Paper_Figures folder
        txt_paper_out = os.path.join(PAPER_DIR, f"{sc['prefix']}_energy_conservation_all_carriers.txt")
        with open(txt_paper_out, "w", encoding="utf-8") as f:
            f.write(combined_content)
        print(f"Saved text validation report to Paper_Figures: {txt_paper_out}")
        
# =====================================================================
# NODAL DISPATCH & ENERGY MIX PIE CHART MAP
# =====================================================================

        # 2. Generate and Save Nodal Dispatch & Energy Mix Map
        print(f"  -> Generating Nodal Dispatch & Energy Mix Pie Chart Map...")
        
        CHART_COLORS = {
            "Onshore Wind": "#2ecc71",          # Green
            "Solar PV": "#ffbf00",              # Amber (Yellow-Orange blend)
            "Oil": "#8b4513",                   # Brown
            "CCGT": "#e377c2",                  # Pink
            "Gas": "#ff00ff",                   # Magenta
            "H2 Turbine": "#9467bd",            # Purple
            "H2 Fuel Cell": "#00ffff",          # Cyan
            "H2 Electrolyser": "#8da35d",       # Olive green (alias)
            "H2 Storage": "#87ceeb",            # Light blue (Sky blue)
            "Battery": "#8e44ad",               # Purple
            "SMR": "#95a5a6",                   # Grey
            "SMR CC": "#1f77b4",                # Blue
        }

        ac_buses = [b for b in n.buses.index if n.buses.loc[b, "carrier"] == "AC"]
        weights = n.snapshot_weightings.generators
        terms = link_terminals(n)

        all_nodal_records = []

        def get_base_ac_bus(bus_name):
            b_str = str(bus_name)
            for suffix in [" H2", " gas", " low voltage", " home battery", " battery", " co2 stored", " NH3", " co2", " process emissions", " solid biomass for industry", " gas for industry"]:
                if b_str.endswith(suffix):
                    b_str = b_str[:-len(suffix)]
            return b_str if b_str in ac_buses else None

        def collect_component_records(inj, bus_of, label_of):
            if inj.empty:
                return
            cols = [c for c in inj.columns if c in bus_of.index]
            if not cols:
                return
            sub = inj[cols]
            for c in cols:
                bus = bus_of.loc[c]
                base_bus = get_base_ac_bus(bus)
                if base_bus is not None:
                    lbl = label_of.loc[c] if hasattr(label_of, "loc") else label_of[c]
                    val = float(sub[c].clip(lower=0).mul(weights, axis=0).sum())
                    if val > 0:
                        all_nodal_records.append({"bus": base_bus, "label": lbl, "energy": val})

        # Generators
        if not n.generators.empty and not n.generators_t.p.empty:
            collect_component_records(n.generators_t.p, n.generators["bus"], "Generator: " + n.generators["carrier"].astype(str))

        # Storage Units
        if not n.storage_units.empty and not n.storage_units_t.p.empty:
            collect_component_records(n.storage_units_t.p, n.storage_units["bus"], "StorageUnit: " + n.storage_units["carrier"].astype(str))

        # Stores
        if not n.stores.empty and not n.stores_t.p.empty:
            collect_component_records(n.stores_t.p, n.stores["bus"], "Store: " + n.stores["carrier"].astype(str))

        # Links
        if not n.links.empty and not n.links_t.p0.empty:
            for l_id, l_row in n.links.iterrows():
                carrier = l_row["carrier"]
                for i in terms:
                    flow_col = f"p{i}" if i > 0 else "p0"
                    bus_col = f"bus{i}" if i > 0 else "bus0"
                    if flow_col in n.links_t and bus_col in n.links.columns:
                        bus = l_row[bus_col]
                        base_bus = get_base_ac_bus(bus)
                        if base_bus is not None:
                            eff_col = f"efficiency{i}" if i > 0 else "efficiency"
                            eff = l_row.get(eff_col, 1.0) or 1.0
                            flow_series = n.links_t[flow_col][l_id]
                            val = float((-flow_series).clip(lower=0).mul(weights, axis=0).sum() * eff)
                            if val > 0:
                                lbl = f"Link: {carrier}"
                                all_nodal_records.append({"bus": base_bus, "label": lbl, "energy": val})

        if all_nodal_records:
            df_raw = pd.DataFrame(all_nodal_records)
            target_keywords = ["solar", "wind", "ccgt", "cogt", "oil", "gas", "battery", "fuel cell", "electrolysis", "h2-turbine", "h2 turbine", "smr", "smr-cc"]
            exclude_keywords = ["load", "grid", "distribution", "charger", "h2 uhs", "h2 store tank", "vent", "cracker", "haber-bosch", "helmeth", "fischer-tropsch", "sabatier"]
            
            def is_target_element(label):
                l_lower = str(label).lower()
                if any(ex in l_lower for ex in exclude_keywords):
                    return False
                return any(kw in l_lower for kw in target_keywords)

            df_filtered = df_raw[df_raw["label"].apply(is_target_element)].copy()
            
            def normalize_carrier_name(label):
                l_lower = str(label).lower()
                if "solar" in l_lower:
                    return "Solar PV"
                elif "wind" in l_lower:
                    return "Onshore Wind"
                elif "battery" in l_lower:
                    return "Battery"
                elif "ccgt" in l_lower:
                    return "CCGT"
                elif "oil" in l_lower:
                    return "Oil"
                elif "gas" in l_lower:
                    return "Gas"
                elif "h2 turbine" in l_lower or "h2-turbine" in l_lower:
                    return "H2 Turbine"
                elif "fuel cell" in l_lower:
                    return "H2 Fuel Cell"
                elif "electrolyser" in l_lower or "electrolysis" in l_lower:
                    return "H2 Electrolyser"
                elif "h2 uhs" in l_lower or "h2 store tank" in l_lower or "h2 storage" in l_lower or ("store" in l_lower and "h2" in l_lower):
                    return "H2 Storage"
                elif "smr cc" in l_lower:
                    return "SMR CC"
                elif "smr" in l_lower:
                    return "SMR"
                else:
                    return label.split(":")[-1].strip().title()

            df_filtered["carrier_key"] = df_filtered["label"].apply(normalize_carrier_name)
            nodal_mix = df_filtered.pivot_table(index="bus", columns="carrier_key", values="energy", aggfunc="sum", fill_value=0.0)
        else:
            nodal_mix = pd.DataFrame(index=ac_buses)

        total_energy_all = nodal_mix.sum().sum() if not nodal_mix.empty else 0.0
        if not nodal_mix.empty and total_energy_all > 0:
            global_percentages = (nodal_mix.sum() / total_energy_all) * 100.0
            valid_carriers = global_percentages[global_percentages >= CARRIER_PERCENTAGE].index
            nodal_mix = nodal_mix[[c for c in nodal_mix.columns if c in valid_carriers]]

        rounded_percentages = {}
        if not nodal_mix.empty and total_energy_all > 0:
            active_cols = sorted(list(nodal_mix.columns))
            raw_pcts = {c: (nodal_mix[c].sum() / total_energy_all) * 100.0 for c in active_cols}
            int_pcts = {c: int(round(p)) for c, p in raw_pcts.items()}
            diff = 100 - sum(int_pcts.values())
            if diff != 0:
                remainders = {c: raw_pcts[c] - int_pcts[c] for c in active_cols}
                sorted_by_rem = sorted(remainders, key=remainders.get, reverse=True)
                for i in range(abs(diff)):
                    c_target = sorted_by_rem[i % len(sorted_by_rem)]
                    int_pcts[c_target] += 1 if diff > 0 else -1
            rounded_percentages = int_pcts

        fig = plt.figure(figsize=(11, 8))
        if HAS_CARTOPY:
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.set_extent([9.0, 25.5, 19.0, 33.8], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor="#ffffff", zorder=0)  # Pure white background for high contrast
            ax.add_feature(cfeature.OCEAN, facecolor="#e0f2fe", zorder=0)
            ax.add_feature(cfeature.BORDERS, linestyle="--", edgecolor="#666666", linewidth=0.8, zorder=1)
            ax.add_feature(cfeature.COASTLINE, edgecolor="#333333", linewidth=1.0, zorder=1)
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle=":")
            gl.top_labels = False
            gl.right_labels = False
        else:
            ax = fig.add_subplot(1, 1, 1)

        n.plot(ax=ax, bus_sizes=0, line_widths=0, link_widths=0, margin=0)

        if HAS_CARTOPY:
            ax.set_extent([9.0, 25.5, 19.0, 33.8], crs=ccrs.PlateCarree())

        plot_buses = n.buses.copy()
        mask_sea_node = (plot_buses["x"] >= 19.4) & (plot_buses["x"] <= 20.3) & (plot_buses["y"] >= 30.0) & (plot_buses["y"] <= 31.5)
        plot_buses.loc[mask_sea_node, "y"] = 30.15
        plot_buses.loc[mask_sea_node, "x"] = 19.80

        COAST_LONS = np.array([9.0, 11.5, 12.08, 12.73, 13.19, 14.26, 14.57, 15.09, 15.25, 15.50, 15.80, 16.20, 16.59, 17.50, 18.56, 19.20, 19.57, 19.80, 20.00, 20.10, 20.20, 20.30, 20.50, 20.83, 21.75, 22.64, 23.96, 25.09, 26.0])
        COAST_LATS = np.array([33.0, 33.1, 32.93, 32.76, 32.89, 32.65, 32.47, 32.38, 31.40, 31.20, 31.12, 31.15, 31.21, 30.80, 30.51, 30.35, 30.41, 30.42, 30.50, 30.60, 35.20, 32.00, 32.10, 32.49, 32.76, 32.77, 32.08, 31.76, 31.50])


        def get_coast_lat(lon):
            return np.interp(lon, COAST_LONS, COAST_LATS)

        def route_line_land(x0, y0, x1, y1, land_offset=0.13, num_pts=50, allow_waypoint=True):
            if x0 == x1 or (abs(x0 - x1) < 1e-5 and abs(y0 - y1) < 1e-5):
                return [x0, x1], [y0, y1]
            min_x, max_x = min(x0, x1), max(x0, x1)
            min_y, max_y = min(y0, y1), max(y0, y1)

            if ((min_x < 20.15 and max_x > 19.8) and (min_y < 30.8 and max_y > 31.3)) and allow_waypoint:
                x_mid, y_mid = 20.32, 30.70
                xs1, ys1 = route_line_land(x0, y0, x_mid, y_mid, land_offset=land_offset, num_pts=num_pts//2, allow_waypoint=False)
                xs2, ys2 = route_line_land(x_mid, y_mid, x1, y1, land_offset=land_offset, num_pts=num_pts//2, allow_waypoint=False)
                return np.concatenate([xs1, xs2]), np.concatenate([ys1, ys2])

            xs = np.linspace(x0, x1, num_pts)
            ys_linear = np.linspace(y0, y1, num_pts)
            coast_lats = get_coast_lat(xs) - land_offset
            ys_land = np.minimum(ys_linear, coast_lats)
            ys_land[0], ys_land[-1] = y0, y1

            if num_pts > 5 and np.any(ys_linear > coast_lats):
                kernel = np.ones(5) / 5.0
                ys_smooth = np.convolve(ys_land, kernel, mode='same')
                ys_smooth[0], ys_smooth[1], ys_smooth[-2], ys_smooth[-1] = y0, ys_land[1], ys_land[-2], y1
                ys_land = np.minimum(ys_smooth, coast_lats)
                ys_land[0], ys_land[-1] = y0, y1
            return xs, ys_land

        # ===================================================================
        # TRANSMISSION LINE CAPACITY LOADING & COLOR BAR MAPPING
        # ===================================================================
        import matplotlib.colors as mcolors
        from matplotlib.colors import LinearSegmentedColormap

        line_loading_series = pd.Series(0.0, index=n.lines.index)
        if not n.lines.empty and not n.lines_t.p0.empty:
            flows = n.lines_t.p0.abs()
            caps = n.lines["s_nom_opt"] if "s_nom_opt" in n.lines.columns else n.lines["s_nom"]
            caps = caps.replace(0, 1e-6)
            loading_ratios = flows.div(caps, axis=1) * 100.0
            line_loading_series = loading_ratios.mean(axis=0)

        min_val, max_val = 0.0, max(line_loading_series.max() if not line_loading_series.empty else 1.0, 1.0)
        norm_metric = mcolors.Normalize(vmin=min_val, vmax=max_val)
        
        # Truncate Reds colormap to start at 0.20 instead of 0.0 so low-loading lines are clearly visible
        base_cmap = plt.colormaps.get_cmap("Reds") if hasattr(plt, "colormaps") else plt.cm.get_cmap("Reds")
        cmap_lines = LinearSegmentedColormap.from_list(
            "truncated_reds", base_cmap(np.linspace(0.20, 1.0, 100)))

        if not n.lines.empty:
            for idx, row in n.lines.iterrows():
                b0, b1 = row["bus0"], row["bus1"]
                if b0 in plot_buses.index and b1 in plot_buses.index:
                    x0, y0 = plot_buses.loc[b0, ["x", "y"]]
                    x1, y1 = plot_buses.loc[b1, ["x", "y"]]
                    if y0 > 33.0 or y1 > 33.0:
                        continue
                    
                    val = line_loading_series.get(idx, 0.0)
                    line_color = cmap_lines(norm_metric(val))
                    linewidth = 1.0 + 3.0 * norm_metric(val)
                    
                    xs, ys = route_line_land(x0, y0, x1, y1)
                    ax.plot(xs, ys, color=line_color, lw=linewidth, zorder=3, alpha=0.85)

        # Attach Vertical Color Bar for Transmission Line Capacity Loading
        sm_line = plt.cm.ScalarMappable(cmap=cmap_lines, norm=norm_metric)
        sm_line.set_array([])
        cbar = fig.colorbar(sm_line, ax=ax, orientation="vertical", shrink=0.7, pad=0.03)
        cbar.set_label("Mean Capacity Loading (%)", fontsize=10)

        max_energy = nodal_mix.sum(axis=1).max() if not nodal_mix.empty and nodal_mix.sum(axis=1).max() > 0 else 1.0
        
        base_pie_size = 0.06 * PIE_SCALE_FACTOR
        min_pie_size = 0.025 * PIE_SCALE_FACTOR

        fig.canvas.draw()
        present_carriers = set()

        node_totals = nodal_mix.sum(axis=1) if not nodal_mix.empty else pd.Series(dtype=float)
        sorted_buses = node_totals.sort_values(ascending=False).index.tolist()
        for b in ac_buses:
            if b not in sorted_buses:
                sorted_buses.append(b)
        bus_ranks = {b: idx + 1 for idx, b in enumerate(sorted_buses)}
        
        is_elec_only = len(ac_buses) > 12

        pending_annotations = []

        for bus in ac_buses:
            if bus not in nodal_mix.index:
                continue
            slices = nodal_mix.loc[bus]
            total_bus_energy = slices.sum()
            if total_bus_energy <= 0:
                continue
            filtered_slices = slices[slices > 0]
            if filtered_slices.empty:
                continue
            for c in filtered_slices.index:
                present_carriers.add(c)

            pie_size = max(min_pie_size, base_pie_size * np.sqrt(total_bus_energy / max_energy))
            colors = [CHART_COLORS.get(c, "#7f7f7f") for c in filtered_slices.index]
            x_coord, y_coord = plot_buses.loc[bus, ["x", "y"]]

            disp_pt = ax.transData.transform((x_coord, y_coord))
            fig_w, fig_h = fig.bbox.width, fig.bbox.height
            if fig_w > 0 and fig_h > 0:
                x_frac, y_frac = disp_pt[0] / fig_w, disp_pt[1] / fig_h
                ax_pie = fig.add_axes([x_frac - pie_size / 2, y_frac - pie_size / 2, pie_size, pie_size])
                ax_pie.set_zorder(5)
                ax_pie.pie(filtered_slices.values, colors=colors, startangle=90)
                ax_pie.set_aspect("equal")

            rank = bus_ranks.get(bus, 1)
            pending_annotations.append({
                "text": str(rank),
                "x": x_coord,
                "y": y_coord,
                "rank": rank
            })

        fig.canvas.draw()
        for ann in pending_annotations:
            if is_elec_only and ann["rank"] > 10:
                continue

            
            if is_elec_only:
                label_x, label_y = ann["x"] + MAP_LABEL_OFFSET_X, ann["y"] + MAP_LABEL_OFFSET_Y
                ha, va = "center", "center"
            else:
                node_px = ax.transData.transform((ann["x"], ann["y"]))
                label_px = (node_px[0], node_px[1] + MAP_PIXEL_OFFSET_Y * fig.dpi / 72)
                label_x, label_y = ax.transData.inverted().transform(label_px)
                ha, va = "center", "bottom"

            label_px = ax.transData.transform((label_x, label_y))
            label_fig = fig.transFigure.inverted().transform(label_px)
            fig.text(
                label_fig[0], label_fig[1], ann["text"],
                ha=ha, va=va, fontsize=MAP_NODE_LABEL_SIZE, color="black",
                weight=MAP_NODE_LABEL_WEIGHT, zorder=1000, clip_on=False
            )
            
        def format_legend_label(carrier_key):
            if carrier_key == "SMR CC":
                name = "SMR+CC"
            elif carrier_key == "SMR":
                name = "SMR"
            elif carrier_key == "CCGT":
                name = "CCGT"
            elif carrier_key == "Solar PV":
                name = "Solar PV"
            else:
                name = str(carrier_key).title()
                
            energy_mwh = nodal_mix[carrier_key].sum() if not nodal_mix.empty and carrier_key in nodal_mix.columns else 0.0
            energy_twh = energy_mwh / 1e6
            pct = rounded_percentages.get(carrier_key, 0)
            return f"{name}: {energy_twh:,.2f} TWh ({pct}%)"
        

        # Sort carriers from highest to lowest total energy share
        sorted_present_carriers = sorted(
            present_carriers, 
            key=lambda c: nodal_mix[c].sum() if not nodal_mix.empty and c in nodal_mix.columns else 0, 
            reverse=True
        )

        legend_handles = [
            mpatches.Patch(color=CHART_COLORS.get(c, "#7f7f7f"), label=format_legend_label(c))
            for c in sorted_present_carriers
        ]
        
        if legend_handles:
            ax.legend(
                handles=legend_handles, 
                loc="lower left", 
                fontsize=MAP_LEGEND_FONT_SIZE,        
                title_fontsize=MAP_LEGEND_FONT_SIZE,  
                title="Energy Carriers (Annual TWh & Share)",
                framealpha=1.0,   
                facecolor="white"
            )
                
        ax.set_title(
            f"PyPSA-Libya 2030: Nodal Dispatch and Energy Mix ({sc['name']})\n"
            "Pie Size = Proportional Node Energy; Slices = Carriers",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Longitude", fontsize=MAP_AXIS_LABEL_FONT_SIZE)
        ax.set_ylabel("Latitude", fontsize=MAP_AXIS_LABEL_FONT_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=MAP_AXIS_TICK_FONT_SIZE)

        # Ensure map output directory exists
        png_dir = os.path.dirname(png_out)
        if png_dir:
            os.makedirs(png_dir, exist_ok=True)

        fig.savefig(png_out, dpi=300, bbox_inches="tight")
        # Save high-res paper PDF version with title removed
        save_paper_pdf(fig, ax, f"{sc['prefix']}_nodal_dispatch_map")
        plt.close(fig)
        print(f"Saved nodal dispatch map to: {png_out}\n")

# =====================================================================
# NODAL DISPATCH & ENERGY MIX STACKED HORIZONTAL BAR CHART
# =====================================================================

        print(f"  -> Generating Nodal Dispatch Stacked Horizontal Bar Chart...")
        
        if not nodal_mix.empty:
            node_totals_bar = nodal_mix.sum(axis=1)
            sorted_nodes = node_totals_bar.sort_values(ascending=True).index.tolist()
            
            df_bar = nodal_mix.loc[sorted_nodes] / 1e6  # Convert MWh to TWh

            if is_elec_only and len(sorted_nodes) > 10:
                top_10_nodes = node_totals_bar.nlargest(10).sort_values(ascending=True).index.tolist()
                df_bar = df_bar.loc[top_10_nodes]

            bar_y_labels = [str(bus_ranks[b]) for b in df_bar.index]

            
            # --- UNIFIED FIGSIZE TO MATCH THE MAP FIGURE (11x8) ---
            
            # Use the identical sorted carriers list for unified ordering
            active_bar_carriers = sorted_present_carriers

            fig_bar, ax_bar = plt.subplots(figsize=(11, 8))
            
            bottom_val = np.zeros(len(df_bar))

            for carrier in active_bar_carriers:
                if carrier not in df_bar.columns:
                    continue
                values = df_bar[carrier].values
                color = CHART_COLORS.get(carrier, "#7f7f7f")
                ax_bar.barh(bar_y_labels, values, left=bottom_val, color=color, label=carrier, edgecolor="none")
                bottom_val += values

            ax_bar.set_title(
                f"PyPSA-Libya 2030: Nodal Energy Dispatch & Mix ({sc['name']})\n"
                f"{'Top 10 Nodes' if is_elec_only else 'All Nodes'} Ranked by Total Energy Capacity",
                fontsize=11,
                fontweight="bold"
            )
            ax_bar.set_xlabel("Energy Dispatch (TWh)", fontsize=BAR_LABEL_FONT_SIZE)
            ax_bar.set_ylabel("Cluster Rank", fontsize=BAR_LABEL_FONT_SIZE)
            ax_bar.tick_params(axis='both', which='major', labelsize=BAR_TICK_FONT_SIZE)
            ax_bar.grid(axis="x", linestyle="--", alpha=0.5)

            # Match handles to the sorted list and fix the title typo
            handles = [mpatches.Patch(color=CHART_COLORS.get(c, "#7f7f7f"), label=c) for c in active_bar_carriers if c in df_bar.columns]
            labels = [c for c in active_bar_carriers if c in df_bar.columns]

            if handles:
                ax_bar.legend(
                    handles, 
                    labels, 
                    loc="lower right", 
                    fontsize=BAR_LEGEND_FONT_SIZE,        
                    title_fontsize=BAR_LEGEND_FONT_SIZE,  
                    title="Energy Carriers)" # Fixed closing parenthesis typo here too
                )
                
            fig_bar.tight_layout()
            bar_dir = os.path.dirname(bar_out)
            if bar_dir:
                os.makedirs(bar_dir, exist_ok=True)
            fig_bar.savefig(bar_out, dpi=300, bbox_inches="tight")
            # Save high-res paper PDF version with title removed
            save_paper_pdf(fig_bar, ax_bar, f"{sc['prefix']}_nodal_bar_chart")
            plt.close(fig_bar)
            print(f"Saved nodal bar chart to: {bar_out}\n")

    print("=" * 110)
    print("ALL SCENARIOS PROCESSED AND SAVED SUCCESSFULLY!")
    print("=" * 110)