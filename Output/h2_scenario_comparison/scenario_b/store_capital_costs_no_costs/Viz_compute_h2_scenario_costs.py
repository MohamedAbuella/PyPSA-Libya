"""
compute_h2_scenario_costs.py

Plots and compares total system costs and spatial investments directly across 
the 3 exported hydrogen scenarios (a, b, and c) for PyPSA-Libya.

Outputs generated in `viz_outputs/h2_scenario_comparison/`:
  1. figure9a_style_cost_by_scenario.png (Stacked bar chart comparing Scenarios a, b, c)
  2. figure9b_style_investment_map_scenario_a.png (Spatial investment map for Scenario a)
  3. figure9b_style_investment_map_scenario_b.png (Spatial investment map for Scenario b)
  4. figure9b_style_investment_map_scenario_c.png (Spatial investment map for Scenario c)

Run from pypsa-libya folder in pypsa-earth conda environment:
    python compute_h2_scenario_costs.py
"""


import os
import re
import math
import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D


# =====================================================================
# PATH CONFIGURATION
# =====================================================================

#### H2 export, week

# RESULTS = {
#     "Scenario A": r"Output\libya_h2export_test\scenario_a\postnetworks\elec_s_36_ec_lcopt_Co2L-3h_144h_2030_0.071_AB_0.25export.nc",
#     "Scenario B": r"Output\libya_h2export_test\scenario_b\postnetworks\elec_s_36_ec_lcopt_Co2L-3h_144h_2030_0.071_AB_0.25export.nc",
#     "Scenario C": r"Output\libya_h2export_test\scenario_c\postnetworks\elec_s_36_ec_lcopt_Co2L-3h_144h_2030_0.071_AB_0.25export.nc",
#           }


#### H2 export, year, 6-hrs 

RESULTS = {
    "Scenario A": r"Output\libya_h2export_test\scenario_a\postnetworks\elec_s_36_ec_lcopt_Co2L-6h_144h_2030_0.071_AB_70export.nc",
    "Scenario B": r"Output\libya_h2export_test\scenario_b\postnetworks\elec_s_36_ec_lcopt_Co2L-6h_144h_2030_0.071_AB_70export.nc",
    "Scenario C": r"Output\libya_h2export_test\scenario_c\postnetworks\elec_s_36_ec_lcopt_Co2L-6h_144h_2030_0.071_AB_70export.nc",
          }


#### H2 export, year, 3-hrs 

# RESULTS = {
#     "Scenario A": r"Output\libya_h2export_test\scenario_a\3hrs\postnetworks\elec_s_36_ec_lcopt_Co2L-3h_144h_2030_0.071_AB_70export.nc",
#     "Scenario B": r"Output\libya_h2export_test\scenario_b\3hrs\postnetworks\elec_s_36_ec_lcopt_Co2L-3h_144h_2030_0.071_AB_70export.nc",
#     "Scenario C": r"Output\libya_h2export_test\scenario_c\3hrs\postnetworks\elec_s_36_ec_lcopt_Co2L-3h_144h_2030_0.071_AB_70export.nc",
#           }


#### No H2 export, year

# RESULTS = {
#     "Scenario A": r"Output\libya_h2export_test\scenario_a\No_Hydrogen_H2_0\postnetworks\elec_s_36_ec_lcopt_Co2L-6h_144h_2030_0.071_AB_0export.nc",
#     "Scenario B": r"Output\libya_h2export_test\scenario_b\No_Hydrogen_H2_0\postnetworks\elec_s_36_ec_lcopt_Co2L-6h_144h_2030_0.071_AB_0export.nc",
#     "Scenario C": r"Output\libya_h2export_test\scenario_c\No_Hydrogen_H2_0\postnetworks\elec_s_36_ec_lcopt_Co2L-6h_144h_2030_0.071_AB_0export.nc",
#           }


OUT_DIR = "viz_outputs/h2_scenario_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

# Map raw PyPSA carrier names -> display categories
CAPEX_CARRIER_MAP = {
    "H2 Electrolysis": "Electrolysis",
    "solar": "Solar PV",
    "onwind": "Onshore Wind",
    "offwind": "Offshore Wind",
    "SMR": "SMR",
    "SMR CC": "SMR with CC",
    "co2 sequestered": "CO2 sequestration",
    "co2 sequestration": "CO2 sequestration",
    "H2 pipeline": "H2 pipeline",
    "H2 Store": "H2 Storage",
    "H2 Store Underground": "H2 Storage",
    "H2 UHS": "H2 Storage",
    "H2 Fuel Cell": "H2 Fuel Cell",
}


CHART_COLORS = {
    "Electrolysis": "#7b2d8e",
    "Solar PV": "#f5a623",
    "Onshore Wind": "#2ecc71",
    "Offshore Wind": "#1abc9c",
    "CO2 sequestration": "#0b1f3a",
    "SMR": "#4F4F2F",
    "SMR with CC": "#1a5276",
    "H2 pipeline": "#c0392b",
    "H2 Storage": "#3498db",
    "H2 Fuel Cell": "#e67e22",
    "Operating Cost": "#d2b48c",
}


def capex_by_category(n: pypsa.Network) -> pd.Series:
    """Annualized capital cost (capital_cost x optimal capacity) grouped by category."""
    parts = []

    if not n.generators.empty:
        g = n.generators
        parts.append((g.carrier, g.capital_cost * g.p_nom_opt))

    if not n.links.empty:
        l = n.links
        parts.append((l.carrier, l.capital_cost * l.p_nom_opt))

    if not n.stores.empty:
        s = n.stores
        parts.append((s.carrier, s.capital_cost * s.e_nom_opt))

    if not n.storage_units.empty:
        su = n.storage_units
        parts.append((su.carrier, su.capital_cost * su.p_nom_opt))

    if not n.lines.empty and "capital_cost" in n.lines:
        ln = n.lines
        parts.append((pd.Series("H2 pipeline", index=ln.index),
                      ln.capital_cost * (ln.s_nom_opt - ln.s_nom).clip(lower=0)))

    if not parts:
        return pd.Series(dtype=float)

    combined = pd.concat(
        [pd.Series(cost.values, index=carrier.values) for carrier, cost in parts]
    )
    combined = combined.groupby(level=0).sum()

    mapped = combined.rename(index=lambda c: CAPEX_CARRIER_MAP.get(c, None))
    mapped = mapped[mapped.index.notna()]
    return mapped.groupby(level=0).sum()


def operating_cost(n: pypsa.Network) -> float:
    """Total operating cost (marginal_cost x dispatch x snapshot weighting)."""
    weights = n.snapshot_weightings.generators
    total = 0.0

    if not n.generators_t.p.empty:
        mc = n.generators.marginal_cost
        total += (n.generators_t.p.mul(mc, axis=1).mul(weights, axis=0)).sum().sum()

    if not n.links_t.p0.empty:
        mc = n.links.marginal_cost
        total += (n.links_t.p0.mul(mc, axis=1).mul(weights, axis=0)).sum().sum()

    return total


print("=" * 70)
print("Computing Annualized Total System Costs across Export Scenarios (A, B, C)")
print("=" * 70)

scenario_costs = {}

for scenario_name, path in RESULTS.items():
    if not os.path.exists(path):
        print(f"\n[SKIP] {scenario_name}: File not found at '{path}'")
        continue

    print(f"Loading network for {scenario_name}...")
    n_export = pypsa.Network(path)

    capex = capex_by_category(n_export)
    opex = operating_cost(n_export)

    combined = capex.copy()
    combined["Operating Cost"] = opex

    scenario_costs[scenario_name] = combined
    print(f"\n{scenario_name} Breakdown:")
    print((combined / 1e9).round(3).to_string())
    print(f"  TOTAL System Cost: {combined.sum() / 1e9:,.3f} B EUR/yr")

if not scenario_costs:
    print("\nNo scenario output files were found. Please check network run paths.")
else:
    # ------------------------------------------------------------------
    # 1. Total System Cost Stacked Bar Chart Comparison
    # ------------------------------------------------------------------
    df = pd.DataFrame(scenario_costs).T / 1e9  # Convert to billions
    df = df.fillna(0.0)

    category_order = [c for c in
                      ["Solar PV", "Onshore Wind", "Offshore Wind", "Electrolysis", 
                       "H2 Storage", "H2 Fuel Cell", "SMR", "SMR with CC", 
                       "CO2 sequestration", "H2 pipeline", "Operating Cost"]
                      if c in df.columns]
    df = df[category_order]

    fig, ax = plt.subplots(figsize=(9, 6))
    bottom = pd.Series(0.0, index=df.index)

    for cat in category_order:
        ax.bar(df.index, df[cat], bottom=bottom,
               color=CHART_COLORS.get(cat, "#95a5a6"), label=cat,
               edgecolor="black", linewidth=0.5)
        bottom += df[cat]

    ax.set_ylabel("Annualized System Cost [Billion EUR]", fontsize=11)
    ax.set_title("PyPSA-Libya: Export Hydrogen System Cost Comparison\nScenarios (a), (b), and (c)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "figure9a_style_cost_by_scenario.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved cost comparison chart -> {out_path}")



    def per_bus_capex(net: pypsa.Network) -> pd.DataFrame:
        """Extracts capital costs per bus across all network components, including H2 stores."""
        frames = []
        
        if not net.generators.empty:
            g = net.generators
            cost = g.capital_cost * g.p_nom_opt
            cat = g.carrier.map(lambda c: CAPEX_CARRIER_MAP.get(c, None))
            frames.append(pd.DataFrame({"bus": g.bus, "category": cat, "cost": cost}))
    
        if not net.links.empty:
            l = net.links
            cost = l.capital_cost * l.p_nom_opt
            cat = l.carrier.map(lambda c: CAPEX_CARRIER_MAP.get(c, None))
            frames.append(pd.DataFrame({"bus": l.bus0, "category": cat, "cost": cost}))
    
        if not net.stores.empty:
            s = net.stores
            cost = s.capital_cost * s.e_nom_opt
            cat = s.carrier.map(lambda c: CAPEX_CARRIER_MAP.get(c, None))
            frames.append(pd.DataFrame({"bus": s.bus, "category": cat, "cost": cost}))
    
        if not net.storage_units.empty:
            su = net.storage_units
            cost = su.capital_cost * su.p_nom_opt
            cat = su.carrier.map(lambda c: CAPEX_CARRIER_MAP.get(c, None))
            frames.append(pd.DataFrame({"bus": su.bus, "category": cat, "cost": cost}))
    
        if not frames:
            return pd.DataFrame()
    
        res = pd.concat(frames, ignore_index=True)
        res = res.dropna(subset=["category"])
        
        # Normalize bus names (e.g., 'LY0 32 H2' -> 'LY0 32')
        res["bus"] = res["bus"].apply(lambda b: b[:-3] if str(b).endswith(" H2") else str(b))
        return res
        


    def strip_h2_suffix(b):
        return b[:-3] if b.endswith(" H2") else b

    print("\nGenerating Spatial Investment (CAPEX) Maps...")

    for scenario_name, path in RESULTS.items():
        if not os.path.exists(path):
            print(f"[SKIP Map] {scenario_name}: File not found at '{path}'")
            continue

        n = pypsa.Network(path)
        bus_cost = per_bus_capex(n)

        if bus_cost.empty:
            print(f"[SKIP Map] {scenario_name}: No mapped capex components found.")
            continue

        pivot = bus_cost.pivot_table(index="bus", columns="category",
                                      values="cost", aggfunc="sum", fill_value=0.0)

        pivot.index = [strip_h2_suffix(b) for b in pivot.index]
        pivot = pivot.groupby(level=0).sum()

        buses = n.buses.loc[n.buses.index.intersection(pivot.index)]
        fig, ax = plt.subplots(figsize=(10, 8))
        total_by_bus = pivot.sum(axis=1)
        max_cost = total_by_bus.max() if total_by_bus.max() > 0 else 1.0

        for _, row in n.lines.iterrows():
            if row.bus0 in buses.index and row.bus1 in buses.index:
                x0, y0 = buses.loc[row.bus0, ["x", "y"]]
                x1, y1 = buses.loc[row.bus1, ["x", "y"]]
                ax.plot([x0, x1], [y0, y1], c="gray", lw=0.5, alpha=0.5, zorder=1)

        for bus in buses.index:
            x, y = buses.loc[bus, ["x", "y"]]
            total = total_by_bus.get(bus, 0.0)
            
            if total < 0.5 * 1e6:
                ax.scatter(x, y, s=20, c="lightgray", edgecolors="gray", linewidths=0.5, zorder=2)
                continue

            size = 150 + 2500 * (total / max_cost)
            dominant = pivot.loc[bus].idxmax()
            ax.scatter(x, y, s=size, c=CHART_COLORS.get(dominant, "gray"),
                       edgecolors="black", linewidths=0.6, alpha=0.85, zorder=3)
            
            label_text = f"{total/1e9:.2f}B" if total >= 1e9 else f"{total/1e6:.0f}M"
            ax.annotate(label_text, (x, y), fontsize=7, fontweight="bold",
                        textcoords="offset points", xytext=(4, 4), zorder=4)

        legend_handles = [mpatches.Patch(color=c, label=k) for k, c in CHART_COLORS.items() if k in pivot.columns]
        ax.legend(handles=legend_handles, loc="lower left", fontsize=8)
        ax.set_title(f"PyPSA-Libya — {scenario_name}: Spatial Financial Investment Allocation\n"
                     "Marker size = Total Capex at node; Color = Dominant Technology", fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle=":", alpha=0.6)

        safe_name = scenario_name.lower().replace(" ", "_")
        out_path = os.path.join(OUT_DIR, f"figure9b_style_investment_map_{safe_name}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved financial map -> {out_path}")


# ------------------------------------------------------------------
    # 2.1 Spatial Investment (Financial CAPEX) Maps for All Scenarios (A, B, C)
    # ------------------------------------------------------------------
    print("\nGenerating Spatial Investment (CAPEX) Maps...")

    for scenario_name, path in RESULTS.items():
        if not os.path.exists(path):
            print(f"[SKIP Map] {scenario_name}: File not found at '{path}'")
            continue

        n = pypsa.Network(path)
        bus_cost = per_bus_capex(n)

        if bus_cost.empty:
            print(f"[SKIP Map] {scenario_name}: No mapped capex components found.")
            continue

        pivot = bus_cost.pivot_table(
            index="bus", columns="category", values="cost", aggfunc="sum", fill_value=0.0
        )

        buses = n.buses.loc[n.buses.index.intersection(pivot.index)]
        fig, ax = plt.subplots(figsize=(10, 8))
        total_by_bus = pivot.sum(axis=1)
        max_cost = total_by_bus.max() if total_by_bus.max() > 0 else 1.0

        for _, row in n.lines.iterrows():
            b0 = strip_h2_suffix(row.bus0)
            b1 = strip_h2_suffix(row.bus1)
            if b0 in buses.index and b1 in buses.index:
                x0, y0 = buses.loc[b0, ["x", "y"]]
                x1, y1 = buses.loc[b1, ["x", "y"]]
                ax.plot([x0, x1], [y0, y1], c="gray", lw=0.5, alpha=0.5, zorder=1)

        for bus in buses.index:
            x, y = buses.loc[bus, ["x", "y"]]
            total = total_by_bus.get(bus, 0.0)
            
            if total < 0.5 * 1e6:
                ax.scatter(x, y, s=20, c="lightgray", edgecolors="gray", linewidths=0.5, zorder=2)
                continue

            size = 150 + 2500 * (total / max_cost)
            dominant = pivot.loc[bus].idxmax()
            ax.scatter(
                x, y, s=size, c=CHART_COLORS.get(dominant, "gray"),
                edgecolors="black", linewidths=0.6, alpha=0.85, zorder=3
            )
            
            label_text = f"{total/1e9:.2f}B" if total >= 1e9 else f"{total/1e6:.0f}M"
            ax.annotate(
                label_text, (x, y), fontsize=7, fontweight="bold",
                textcoords="offset points", xytext=(4, 4), zorder=4
            )

        legend_handles = [
            mpatches.Patch(color=c, label=k) 
            for k, c in CHART_COLORS.items() 
            if k in pivot.columns
        ]
        ax.legend(handles=legend_handles, loc="lower left", fontsize=8)
        ax.set_title(
            f"PyPSA-Libya — {scenario_name}: Spatial Financial Investment Allocation\n"
            "Marker size = Total Capex at node; Color = Dominant Technology", fontsize=11
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle=":", alpha=0.6)

        safe_name = scenario_name.lower().replace(" ", "_")
        out_path = os.path.join(OUT_DIR, f"figure9b_style_investment_map_{safe_name}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved financial map -> {out_path}")
        
        
        

    # ------------------------------------------------------------------
    # 2.2 Spatial Capacity Investment Maps for All Scenarios (A, B, C)
    # ------------------------------------------------------------------
    def per_bus_capacity(net: pypsa.Network) -> pd.DataFrame:
        """Extracts optimal installed capacities per bus (MW for generators/links, MWh for storage)."""
        frames = []
        
        if not net.generators.empty:
            g = net.generators
            cat = g.carrier.map(lambda c: CAPEX_CARRIER_MAP.get(c, None))
            frames.append(pd.DataFrame({"bus": g.bus, "category": cat, "capacity": g.p_nom_opt}))
    
        if not net.links.empty:
            l = net.links
            cat = l.carrier.map(lambda c: CAPEX_CARRIER_MAP.get(c, None))
            frames.append(pd.DataFrame({"bus": l.bus0, "category": cat, "capacity": l.p_nom_opt}))
    
        if not net.stores.empty:
            s = net.stores
            cat = s.carrier.map(lambda c: CAPEX_CARRIER_MAP.get(c, None))
            frames.append(pd.DataFrame({"bus": s.bus, "category": cat, "capacity": s.e_nom_opt}))
    
        if not net.storage_units.empty:
            su = net.storage_units
            cat = su.carrier.map(lambda c: CAPEX_CARRIER_MAP.get(c, None))
            frames.append(pd.DataFrame({"bus": su.bus, "category": cat, "capacity": su.p_nom_opt}))
    
        if not frames:
            return pd.DataFrame()
    
        res = pd.concat(frames, ignore_index=True)
        res = res.dropna(subset=["category"])
        
        # Normalize bus names to align with network spatial coordinates
        res["bus"] = res["bus"].apply(lambda b: b[:-3] if str(b).endswith(" H2") else str(b))
        return res

    print("\nGenerating Spatial Capacity Investment Maps...")

    for scenario_name, path in RESULTS.items():
        if not os.path.exists(path):
            print(f"[SKIP Capacity Map] {scenario_name}: File not found at '{path}'")
            continue

        n = pypsa.Network(path)
        bus_cap = per_bus_capacity(n)

        if bus_cap.empty:
            print(f"[SKIP Capacity Map] {scenario_name}: No capacity components found.")
            continue

        pivot_cap = bus_cap.pivot_table(index="bus", columns="category",
                                         values="capacity", aggfunc="sum", fill_value=0.0)

        pivot_cap.index = [strip_h2_suffix(b) for b in pivot_cap.index]
        pivot_cap = pivot_cap.groupby(level=0).sum()

        buses = n.buses.loc[n.buses.index.intersection(pivot_cap.index)]
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate total power generation & conversion capacity in MW (excluding MWh storage for sizing scale)
        gen_cols = [c for c in pivot_cap.columns if c != "H2 Storage"]
        total_mw_by_bus = pivot_cap[gen_cols].sum(axis=1) if gen_cols else pivot_cap.sum(axis=1)
        max_mw = total_mw_by_bus.max() if total_mw_by_bus.max() > 0 else 1.0

        for _, row in n.lines.iterrows():
            if row.bus0 in buses.index and row.bus1 in buses.index:
                x0, y0 = buses.loc[row.bus0, ["x", "y"]]
                x1, y1 = buses.loc[row.bus1, ["x", "y"]]
                ax.plot([x0, x1], [y0, y1], c="gray", lw=0.5, alpha=0.5, zorder=1)

        for bus in buses.index:
            x, y = buses.loc[bus, ["x", "y"]]
            total_mw = total_mw_by_bus.get(bus, 0.0)
            
            # Filter out zero / minimal capacity buses (< 50 MW)
            if total_mw < 50.0:
                ax.scatter(x, y, s=20, c="lightgray", edgecolors="gray", linewidths=0.5, zorder=2)
                continue

            size = 150 + 2500 * (total_mw / max_mw)
            dominant = pivot_cap.loc[bus, gen_cols].idxmax() if gen_cols else pivot_cap.loc[bus].idxmax()
            
            ax.scatter(x, y, s=size, c=CHART_COLORS.get(dominant, "gray"),
                       edgecolors="black", linewidths=0.6, alpha=0.85, zorder=3)
            
            # Label node in GW or MW
            label_text = f"{total_mw/1e3:.1f}GW" if total_mw >= 1000.0 else f"{total_mw:.0f}MW"
            ax.annotate(label_text, (x, y), fontsize=7, fontweight="bold",
                        textcoords="offset points", xytext=(4, 4), zorder=4)

        legend_handles = [mpatches.Patch(color=c, label=k) for k, c in CHART_COLORS.items() if k in pivot_cap.columns]
        ax.legend(handles=legend_handles, loc="lower left", fontsize=8)
        ax.set_title(f"PyPSA-Libya — {scenario_name}: Spatial Installed Capacity Allocation\n"
                     "Marker size = Total Power/Conversion Capacity [MW/GW]; Color = Dominant Technology", fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle=":", alpha=0.6)

        safe_name = scenario_name.lower().replace(" ", "_")
        out_path = os.path.join(OUT_DIR, f"figure9c_style_capacity_map_{safe_name}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved physical capacity map -> {out_path}")
        
        
        
        
# =====================================================================
    # 2.3 Spatial & Graphical Transmission Line Bottlenecks for Scenarios
    # =====================================================================
    
    import numpy as np

    print("\n" + "=" * 70)
    print("Generating Transmission Line Bottlenecks (Reports & Maps) across Scenarios")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Helper Functions for Routing & Line Identification
    # ------------------------------------------------------------------
    COAST_LONS = np.array([
        9.0, 11.5, 12.08, 12.73, 13.19, 14.26, 14.57, 15.09, 
        15.25, 15.50, 15.80, 16.20, 16.59, 17.50, 18.56, 19.20,
        19.57, 19.80, 20.00, 20.10, 20.20, 20.30, 20.50, 20.83, 
        21.75, 22.64, 23.96, 25.09, 26.0
    ])
    COAST_LATS = np.array([
        33.0, 33.1, 32.93, 32.76, 32.89, 32.65, 32.47, 32.38, 
        31.40, 31.20, 31.12, 31.15, 31.21, 30.80, 30.51, 30.35,
        30.41, 30.42, 30.50, 30.60, 31.20, 32.00, 32.10, 32.49, 
        32.76, 32.77, 32.08, 31.76, 31.50
    ])

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

    def shorten_line_id(line_idx, bus0, bus1):
        idx_str = str(line_idx).strip()
        if "lines new" in idx_str or "new" in idx_str.lower() or idx_str.startswith("N"):
            m = re.search(r'LY\s*([1-9]\d*)\s*(\d*)', str(bus1)) or re.search(r'LY\s*([1-9]\d*)\s*(\d*)', str(bus0))
            if m:
                return f"NLY{m.group(1)} {m.group(2)}".strip() if m.group(2) else f"NLY{m.group(1)}"
            clean_idx = idx_str.replace("lines new ", "NL_").replace(" <-> ", "_").replace(" AC", "")
            return re.sub(r'\s+', '', clean_idx)[:10]
        return idx_str

    # ------------------------------------------------------------------
    # Data Processing & Analysis
    # ------------------------------------------------------------------
    scenario_line_data = {}
    
    with open(os.path.join(OUT_DIR, "transmission_line_bottlenecks.txt"), "w", encoding="utf-8") as f_out:
        f_out.write("=" * 135 + "\n")
        f_out.write(" PyPSA-Libya — Scenario Comparison: Transmission Line Bottlenecks Report\n")
        f_out.write("=" * 135 + "\n\n")

        for scenario_name, path in RESULTS.items():
            if not os.path.exists(path):
                continue
            
            n_scen = pypsa.Network(path)
            if n_scen.lines.empty or n_scen.lines_t.p0.empty:
                continue

            flows = n_scen.lines_t.p0.abs()
            caps = n_scen.lines["s_nom_opt"] if "s_nom_opt" in n_scen.lines.columns else n_scen.lines["s_nom"]
            caps = caps.replace(0, 1e-6)
            loading_ratios = flows.div(caps, axis=1) * 100.0

            short_ids = [shorten_line_id(idx, row.bus0, row.bus1) for idx, row in n_scen.lines.iterrows()]

            line_df = pd.DataFrame({
                "short_line_id": short_ids,
                "bus0": n_scen.lines["bus0"],
                "bus1": n_scen.lines["bus1"],
                "capacity_MW": caps,
                "max_flow_MW": flows.max(axis=0),
                "max_loading_pct": loading_ratios.max(axis=0),
                "mean_loading_pct": loading_ratios.mean(axis=0),
                "hours_gt_50pct": (loading_ratios >= 50.0).sum(axis=0),
                "hours_gt_70pct": (loading_ratios >= 70.0).sum(axis=0),
            }, index=n_scen.lines.index).sort_values(by="mean_loading_pct", ascending=False)

            scenario_line_data[scenario_name] = (n_scen, line_df)

            # Write scenario section to master txt report
            f_out.write(f"--- SCENARIO: {scenario_name} ---\n")
            header = f"{'LINE ID':12s} | {'From BUS':10s} | {'To BUS':10s} | {'CAP (MW)':12s} | {'MAX FLOW (MW)':14s} | {'MAX LOAD %':11s} | {'MEAN LOAD %':11s} | {'HRS >= 50%':10s} | {'HRS >= 70%':10s}"
            f_out.write(header + "\n" + "-" * 135 + "\n")

            for _, row in line_df.iterrows():
                f_out.write(f"{row['short_line_id']:12s} | {str(row['bus0']):10s} | {str(row['bus1']):10s} | "
                            f"{row['capacity_MW']:12.1f} | {row['max_flow_MW']:14.1f} | {row['max_loading_pct']:10.2f}% | "
                            f"{row['mean_loading_pct']:10.2f}% | {row['hours_gt_50pct']:10d} | {row['hours_gt_70pct']:10d}\n")
            f_out.write("\n\n")

    print(f"Saved transmission bottlenecks summary text file -> {os.path.join(OUT_DIR, 'transmission_line_bottlenecks.txt')}")

    # ------------------------------------------------------------------
    # Comparative Bar Chart: Top Bottleneck Lines across Scenarios
    # ------------------------------------------------------------------
    if scenario_line_data:
        # Find global top 10 bottleneck lines based on mean loading across scenarios
        combined_means = pd.concat([df["mean_loading_pct"] for _, df in scenario_line_data.values()], axis=1)
        top_line_indices = combined_means.mean(axis=1).nlargest(10).index
        
        # Build comparative dataframe for plotting
        chart_data = {}
        for scen_name, (_, df) in scenario_line_data.items():
            short_labels = df.loc[top_line_indices, "short_line_id"]
            chart_data[scen_name] = df.loc[top_line_indices, "mean_loading_pct"]
        
        plot_df = pd.DataFrame(chart_data, index=top_line_indices)
        plot_df.index = [scenario_line_data[list(scenario_line_data.keys())[0]][1].loc[i, "short_line_id"] for i in top_line_indices]

        fig, ax = plt.subplots(figsize=(12, 6))
        plot_df.plot(kind="bar", ax=ax, width=0.8, colormap="viridis", edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Mean Capacity Loading (%)", fontsize=11)
        ax.set_title("PyPSA-Libya — Transmission Bottlenecks Comparison across Scenarios\n(Top 10 Congested Lines)", fontsize=12, fontweight="bold")
        ax.set_xticklabels(plot_df.index, rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(title="Scenarios", fontsize=9)
        fig.tight_layout()

        out_bar_path = os.path.join(OUT_DIR, "transmission_line_bottlenecks.png")
        fig.savefig(out_bar_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved transmission bottlenecks bar chart -> {out_bar_path}")

    # ------------------------------------------------------------------
    # Spatial Map: Line Bottleneck Locations per Scenario
    # ------------------------------------------------------------------
    for scenario_name, (n_scen, line_df) in scenario_line_data.items():
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(9.0, 25.5)
        ax.set_ylim(19.0, 33.8)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle=":", alpha=0.5)

        metric_series = line_df["mean_loading_pct"]
        min_val, max_val = 0.0, max(metric_series.max(), 1.0)
        norm_metric = mcolors.Normalize(vmin=min_val, vmax=max_val)
        cmap_lines = plt.cm.get_cmap("Reds")

        for idx, row in n_scen.lines.iterrows():
            if row.bus0 in n_scen.buses.index and row.bus1 in n_scen.buses.index:
                x0, y0 = n_scen.buses.loc[row.bus0, ["x", "y"]]
                x1, y1 = n_scen.buses.loc[row.bus1, ["x", "y"]]
                val = metric_series.get(idx, 0.0)
                line_color = cmap_lines(norm_metric(val)) if val > 5.0 else "#a1a1aa"
                linewidth = 1.0 + 3.0 * norm_metric(val)
                xs, ys = route_line_land(x0, y0, x1, y1)
                ax.plot(xs, ys, color=line_color, lw=linewidth, zorder=3, alpha=0.85)

        # Label top 8 congested lines
        top_map_lines = metric_series.head(8)
        for i, idx in enumerate(top_map_lines.index):
            row = n_scen.lines.loc[idx]
            x0, y0 = n_scen.buses.loc[row.bus0, ["x", "y"]]
            x1, y1 = n_scen.buses.loc[row.bus1, ["x", "y"]]
            xs, ys = route_line_land(x0, y0, x1, y1)
            mid_idx = len(xs) // 2
            short_id = line_df.loc[idx, "short_line_id"]
            ax.annotate(short_id, (xs[mid_idx], ys[mid_idx]), fontsize=8, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"), zorder=6)

        macro_buses = n_scen.buses[~n_scen.buses.index.str.contains("low voltage|battery|H2", case=False)]
        ax.scatter(macro_buses.x, macro_buses.y, s=15, c="#1e293b", edgecolors="white", linewidths=0.5, zorder=4)

        sm_line = plt.cm.ScalarMappable(cmap=cmap_lines, norm=norm_metric)
        sm_line.set_array([])
        cbar = fig.colorbar(sm_line, ax=ax, orientation="vertical", shrink=0.7, pad=0.03)
        cbar.set_label("Mean Capacity Loading (%)", fontsize=10)

        ax.set_title(f"PyPSA-Libya — {scenario_name}: Transmission Bottlenecks Map\nLabels indicate Top Congested Corridors", fontsize=11, fontweight="bold")
        
        safe_scen = scenario_name.lower().replace(" ", "_")
        out_map_path = os.path.join(OUT_DIR, f"transmission_line_bottlenecks_map_{safe_scen}.png")
        fig.savefig(out_map_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved transmission map for {scenario_name} -> {out_map_path}")





# =====================================================================
# 3. LCOE & LCOH (Levelized Cost Analysis) Computation & Text Exports
# =====================================================================
print("\n" + "=" * 70)
print("Computing System LCOE and Net-Cost Allocated LCOH across Scenarios")
print("=" * 70)

# Paths to Baseline Reference Networks (0 TWh H2 Export) for Option B Net-Cost Allocation
BASELINE_RESULTS = {
    "Scenario A": r"Output\libya_h2export_test\scenario_a\No_Hydrogen_H2_0\postnetworks\elec_s_36_ec_lcopt_Co2L-6h_144h_2030_0.071_AB_0export.nc",
    "Scenario B": r"Output\libya_h2export_test\scenario_b\No_Hydrogen_H2_0\postnetworks\elec_s_36_ec_lcopt_Co2L-6h_144h_2030_0.071_AB_0export.nc",
    "Scenario C": r"Output\libya_h2export_test\scenario_c\No_Hydrogen_H2_0\postnetworks\elec_s_36_ec_lcopt_Co2L-6h_144h_2030_0.071_AB_0export.nc",
}

DEFAULT_EXPORT_H2_MWH = 70 * 1e6      # 70 TWh export target in MWh
DEFAULT_DOMESTIC_AC_MWH = 52.04 * 1e6 # 52.04 TWh AC domestic demand in MWh

# Hydrogen Heating Values:
# Lower Heating Value (LHV): 33.33 kWh/kg = 0.03333 MWh/kg
# Higher Heating Value (HHV): 39.41 kWh/kg = 0.03941 MWh/kg
H2_LHV_MWH_PER_KG = 0.03333           
H2_HHV_MWH_PER_KG = 0.03941           

lcoe_results = {}
lcoh_results_main = {}
lcoh_results_summary = {}

for scenario_name, export_path in RESULTS.items():
    if not os.path.exists(export_path) or scenario_name not in scenario_costs:
        continue

    # ------------------------------------------------------------------
    # A. Combined System-Wide Levelized Cost of Energy (LCOE) Calculation
    # ------------------------------------------------------------------
    n_export = pypsa.Network(export_path)
    total_system_cost_eur = scenario_costs[scenario_name].sum()
    
    # Extract Domestic AC Load Served
    domestic_ac_mwh = 0.0
    ac_loads = n_export.loads[n_export.loads.carrier == "AC"]
    if not ac_loads.empty and not n_export.loads_t.p.empty:
        weights = n_export.snapshot_weightings.generators
        domestic_ac_mwh = n_export.loads_t.p[ac_loads.index].mul(weights, axis=0).sum().sum()
    if domestic_ac_mwh <= 0:
        domestic_ac_mwh = DEFAULT_DOMESTIC_AC_MWH

    # Extract Exported H2 Energy
    exported_h2_mwh = 0.0
    h2_export_loads = n_export.loads[n_export.loads.carrier.str.contains("H2|h2|export", case=False, na=False)]
    if not h2_export_loads.empty and not n_export.loads_t.p.empty:
        weights = n_export.snapshot_weightings.generators
        exported_h2_mwh = n_export.loads_t.p[h2_export_loads.index].mul(weights, axis=0).sum().sum()
    if exported_h2_mwh <= 0:
        exported_h2_mwh = DEFAULT_EXPORT_H2_MWH

    # Total Multi-Carrier System Energy Demand (Electricity + H2 Export)
    total_system_energy_mwh = domestic_ac_mwh + exported_h2_mwh

    # Calculate Overall System LCOE across all energy served
    lcoe_system_eur_per_mwh = total_system_cost_eur / total_system_energy_mwh
    
    lcoe_results[scenario_name] = {
        "Total System Cost [Billion EUR]": total_system_cost_eur / 1e9,
        "Domestic AC Demand [TWh]": domestic_ac_mwh / 1e6,
        "Exported H2 Energy [TWh]": exported_h2_mwh / 1e6,
        "Total System Energy [TWh]": total_system_energy_mwh / 1e6,
        "System LCOE [EUR/MWh]": lcoe_system_eur_per_mwh,
    }

    # ------------------------------------------------------------------
    # B. Net-Cost Allocated LCOH Calculation (Option B - Table Split)
    # ------------------------------------------------------------------
    base_path = BASELINE_RESULTS.get(scenario_name)
    if base_path and os.path.exists(base_path):
        print(f"Loading 0-Export baseline network for net LCOH: {scenario_name}...")
        n_base = pypsa.Network(base_path)
        
        # Calculate annualized baseline cost for 0 TWh H2 export
        capex_base = capex_by_category(n_base).sum()
        opex_base = operating_cost(n_base)
        baseline_cost_eur = capex_base + opex_base
        
        # Incremental H2 cost = Total Export Run Cost - Baseline Run Cost
        net_h2_cost_eur = total_system_cost_eur - baseline_cost_eur

        # Net LCOH metrics (per MWh, per kg LHV, and per kg HHV)
        lcoh_eur_per_mwh = net_h2_cost_eur / exported_h2_mwh
        lcoh_eur_per_kg_lhv = lcoh_eur_per_mwh * H2_LHV_MWH_PER_KG
        lcoh_eur_per_kg_hhv = lcoh_eur_per_mwh * H2_HHV_MWH_PER_KG

        # Table 1: Cost Breakdown & Export Volume
        lcoh_results_main[scenario_name] = {
            "Export Run Cost [B EUR]": total_system_cost_eur / 1e9,
            "Baseline Cost [B EUR]": baseline_cost_eur / 1e9,
            "Net H2 Cost [B EUR]": net_h2_cost_eur / 1e9,
            "Exported H2 [TWh]": exported_h2_mwh / 1e6,
            "Net LCOH [EUR/MWh]": lcoh_eur_per_mwh,
        }

        # Table 2: Dedicated Unit Conversion Metrics
        lcoh_results_summary[scenario_name] = {
            "Net LCOH [EUR/MWh]": lcoh_eur_per_mwh,
            "Net LCOH LHV [EUR/kg]": lcoh_eur_per_kg_lhv,
            "Net LCOH HHV [EUR/kg]": lcoh_eur_per_kg_hhv,
        }
    else:
        print(f"[WARNING] Baseline network file not found for {scenario_name} at: {base_path}")

# ------------------------------------------------------------------
# Save Reports to File
# ------------------------------------------------------------------
if lcoe_results:
    lcoe_df = pd.DataFrame(lcoe_results).T
    print("\n" + "=" * 50 + "\nSYSTEM LCOE RESULTS\n" + "=" * 50)
    print(lcoe_df.to_string())

    lcoe_txt_path = os.path.join(OUT_DIR, "LCOE_summary_results.txt")
    with open(lcoe_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 85 + "\n")
        f.write("PyPSA-Libya: Combined System-Wide Levelized Cost of Energy (LCOE) Results\n")
        f.write("=" * 85 + "\n\n")
        f.write(lcoe_df.to_string())
        f.write("\n\n" + "=" * 85 + "\n")
        f.write("Notes:\n")
        f.write(" - Total System Cost includes all capital and operational costs for electricity, H2 export, and industrial carriers.\n")
        f.write(" - Total System Energy = Domestic AC Electricity Demand + Exported Hydrogen Energy\n")
        f.write(" - System LCOE [EUR/MWh] = Total System Cost / Total System Energy\n")
        f.write("=" * 85 + "\n")
    print(f"\nSaved LCOE summary report -> {lcoe_txt_path}")

if lcoh_results_main:
    lcoh_main_df = pd.DataFrame(lcoh_results_main).T
    lcoh_summary_df = pd.DataFrame(lcoh_results_summary).T

    print("\n" + "=" * 50 + "\nNET ALLOCATED LCOH RESULTS (OPTION B)\n" + "=" * 50)
    print(lcoh_main_df.to_string())
    print("\n" + lcoh_summary_df.to_string())

    lcoh_txt_path = os.path.join(OUT_DIR, "LCOH_summary_results.txt")
    with open(lcoh_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 95 + "\n")
        f.write("PyPSA-Libya: Net Levelized Cost of Hydrogen (LCOH) Export Results (Option B)\n")
        f.write("=" * 95 + "\n\n")
        f.write("--- Table 1: Cost Allocation & Net LCOH Breakdown ---\n")
        f.write(lcoh_main_df.to_string())
        f.write("\n\n")
        f.write("--- Table 2: Net LCOH Unit Summary (MWh vs. LHV / HHV per kg) ---\n")
        f.write(lcoh_summary_df.to_string())
        f.write("\n\n" + "=" * 95 + "\n")
        f.write("Notes:\n")
        f.write(" - Net H2 Cost = Export System Cost - 0-Export Baseline Cost\n")
        f.write(" - Net LCOH = Net H2 Cost / Exported H2 Energy\n")
        f.write(f" - Assumed H2 Lower Heating Value (LHV) = {H2_LHV_MWH_PER_KG} MWh/kg (33.33 kWh/kg)\n")
        f.write(f" - Assumed H2 Higher Heating Value (HHV) = {H2_HHV_MWH_PER_KG} MWh/kg (39.41 kWh/kg)\n")
        f.write("=" * 95 + "\n")
    print(f"\nSaved Net LCOH summary report -> {lcoh_txt_path}")