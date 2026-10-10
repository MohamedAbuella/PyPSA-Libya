#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_carrier_energy_conservation.py
=======================================

Multi-carrier ENERGY CONSERVATION & NODAL BALANCE validation for a solved
PyPSA / PyPSA-Earth network (.nc). Iterates through all network carriers
in a defined, organized presentation order.

What it checks (for ALL buses of the chosen carrier):
---------------------------------------------------------------------------
 1. NODAL BALANCE (Kirchhoff current law, per bus AND per snapshot)
        sum(injections) - sum(withdrawals) = 0
 2. CARRIER-LEVEL ENERGY BALANCE  (Supply vs Demand, with full breakdown)
        Supply : generators, link outputs, store/storage discharge, negative loads
        Demand : loads, link inputs, store/storage charge, transmission losses
 3. PER-SNAPSHOT SYSTEM CLOSURE   (worst snapshot of the whole carrier system)
 4. TRANSMISSION AUDIT            (line / internal-link losses, sending vs receiving)
 5. CONVERSION-LINK AUDIT         (p_i = -eff_i * p0 consistency + implied losses)
 6. CARRIER ELEMENTS DISTRIBUTION PER NODE (nodal breakdown of carrier elements)
 7. SANITY FLAGS                  (load shedding, curtailment, loads on undefined
                                   buses, negative generation, ...)

PyPSA sign conventions used (identical to PyPSA itself)
-------------------------------------------------------
 generators_t.p      > 0 : injection into the bus
 loads_t.p           > 0 : withdrawal from the bus  (negative load = fixed source)
 stores_t.p          > 0 : dispatch (injection)     < 0 : charging (withdrawal)
 storage_units_t.p   > 0 : dispatch (injection)     < 0 : charging (withdrawal)
 links_t.p{i}        > 0 : WITHDRAWAL at terminal i (outputs are usually < 0)
 lines_t.p{0,1}      > 0 : WITHDRAWAL at that end

Energies are weighted with n.snapshot_weightings.generators, i.e. they are
ANNUALISED exactly as the optimisation's objective is.

Units: values >= 1000 GWh are automatically printed in TWh
       (and < 1 GWh in MWh). Mass carriers (co2) printed in t / kt / Mt.

Usage
-----
    python validate_carrier_energy_conservation.py path/to/result.nc
"""

import os
import sys
import argparse
import logging
import warnings

import numpy as np
import pandas as pd
import pypsa

warnings.filterwarnings("ignore")
logging.getLogger("pypsa").setLevel(logging.ERROR)

# ----------------------------------------------------------------------------
# USER SETTINGS (used when no command-line arguments are given, e.g. Spyder)
# ----------------------------------------------------------------------------

####### Electricity-only ##########

# NC_PATH = r"Output\libya-year-36\LCO2\3h\networks\elec_s_36_ec_lcopt_Co2L-3h.nc"

# NC_PATH = r"Output\libya-year-36\No_LCO2\3h\networks\elec_s_36_ec_lcopt_Co2L-3h.nc"

######### Sector-Coupling ##########

# NC_PATH = r"Output\libya_h2export_test\scenario_a\3hrs\postnetworks\elec_s_12_ec_lcopt_Co2L-3h__2030_0.071_AB_70export.nc"

# NC_PATH = r"Output\libya_h2export_test\scenario_b\3hrs\postnetworks\elec_s_12_ec_lcopt_Co2L-3h__2030_0.071_AB_70export.nc"

# NC_PATH = r"Output\libya_h2export_test\scenario_c\3hrs\postnetworks\elec_s_12_ec_lcopt_Co2L-3h__2030_0.071_AB_70export.nc"

######### Hybrid Pathway ##########


### NC_PATH = r"Output/libya_h2export_hybrid_emispr_100/3hrs/postnetworks/elec_s_12_ec_lcopt_Co2L-Ep-3h__2030_0.071_AB_70export.nc"

### NC_PATH = r"Output/libya_h2export_hybrid_emispr_100/3hrs/postnetworks/Week_hybrid_elec_s_12_ec_lcopt_Co2L-Ep-3h__2030_0.071_AB_70export.nc"

### NC_PATH = r"Output/libya_h2export_hybrid_emispr_100/3hrs/postnetworks/Month_Hybrid_Solar_H2_turbine_elec_s_12_ec_lcopt_Co2L-Ep-3h__2030_0.071_AB_70export.nc"

### NC_PATH = r"Output/libya_h2export_hybrid_emispr_100/3hrs/postnetworks/Month_Hybrid_Solar_Wind_H2_turbine_elec_s_12_ec_lcopt_Co2L-Ep-3h__2030_0.071_AB_70export.nc"

### NC_PATH = r"Output/libya_h2export_hybrid_emispr_100/3hrs/postnetworks/Month_Hybrid_Solar_Battery_energy_conservation_all_carriers.nc"



NC_PATH = r"Output/libya_h2export_hybrid_emispr_100/3hrs/postnetworks/orig_elec_s_12_ec_lcopt_Co2L-Ep-3h__2030_0.071_AB_70export.nc"



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

    # Flexible case-insensitive carrier matching
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

    # ---------------------------------------------------------------- header
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

    # -------------------------------------------- collect injections by group
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


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Multi-carrier energy conservation validation into a single file")
    ap.add_argument("nc_path", nargs="?", default=NC_PATH)
    ap.add_argument("--carrier", default=None, help="Carrier name or comma-separated list (optional)")
    ap.add_argument("--tol-mw", type=float, default=TOL_MW)
    ap.add_argument("--rel-tol", type=float, default=REL_TOL)
    ap.add_argument("--out", default=None, help="Path for the combined output .txt file")
    ap.add_argument("--no-save", action="store_true")
    args, _ = ap.parse_known_args()

    if not os.path.exists(args.nc_path):
        sys.exit(f"Network file not found: {args.nc_path}")

    print(f"Loading network file: {os.path.basename(args.nc_path)} ...")
    n = pypsa.Network(args.nc_path)

    available_carriers = list(n.buses["carrier"].unique())

    if args.carrier:
        carriers_to_run = [c.strip() for c in args.carrier.split(",")]
    elif isinstance(CARRIER, list) and CARRIER:
        carriers_to_run = [c for c in CARRIER if any(str(ac).lower() == str(c).lower() for ac in available_carriers)]
        for ac in available_carriers:
            if not any(str(ac).lower() == str(c).lower() for c in carriers_to_run):
                carriers_to_run.append(ac)
    else:
        carriers_to_run = available_carriers

    output_dir = os.path.dirname(os.path.abspath(args.nc_path))
    combined_report_path = args.out or os.path.join(output_dir, "energy_conservation_all_carriers.txt")

    all_reports = []
    
    for c in carriers_to_run:
        print(f"Running validation for carrier: {c} ...")
        try:
            report_text = validate_carrier(n, carrier=c, tol_mw=args.tol_mw, rel_tol=args.rel_tol)
            all_reports.append(report_text)
        except Exception as e:
            err_msg = f"\nERROR validating carrier '{c}': {e}\n"
            print(err_msg)
            all_reports.append(err_msg)

    separator = "\n\n" + "=" * 100 + "\n" + " " * 35 + "NEXT CARRIER REPORT SECTION" + "\n" + "=" * 100 + "\n\n"
    combined_content = separator.join(all_reports)

    if not args.no_save:
        with open(combined_report_path, "w", encoding="utf-8") as f:
            f.write(combined_content)
        print(f"\nCombined report successfully saved to: {combined_report_path}")