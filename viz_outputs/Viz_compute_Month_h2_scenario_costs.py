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


#### H2 export, month, 3-hrs 

# RESULTS = {
   
#     "Scenario A": r"Output\Month_libya_h2export_test\scenario_a\3hrs\postnetworks\elec_s_36_ec_lcopt_Co2L-3h__2030_0.071_AB_70export.nc",
#     "Scenario B": r"Output\Month_libya_h2export_test\scenario_b\3hrs\postnetworks\elec_s_36_ec_lcopt_Co2L-3h__2030_0.071_AB_70export.nc",
#     "Scenario C": r"Output\Month_libya_h2export_test\scenario_c\3hrs\postnetworks\elec_s_36_ec_lcopt_Co2L-3h__2030_0.071_AB_70export.nc",
#           }



#### H2 export, month, 6-hrs 

RESULTS = {
    "Scenario A": r"Output\Month_libya_h2export_test\scenario_a\postnetworks\elec_s_36_ec_lcopt_Co2L-6h__2030_0.071_AB_70export.nc",
    "Scenario B": r"Output\Month_libya_h2export_test\scenario_b\postnetworks\elec_s_36_ec_lcopt_Co2L-6h__2030_0.071_AB_70export.nc",
    "Scenario C": r"Output\Month_libya_h2export_test\scenario_c\postnetworks\elec_s_36_ec_lcopt_Co2L-6h__2030_0.071_AB_70export.nc",
          }



# ## No H2 export, month, 6-hrs 

# RESULTS = {
#     "Scenario A": r"Output\Month_libya_h2export_test\scenario_a\No_Hydrogen_H2_0\postnetworks\elec_s_36_ec_lcopt_Co2L-6h__2030_0.071_AB_0export.nc",
#     "Scenario B": r"Output\Month_libya_h2export_test\scenario_b\No_Hydrogen_H2_0\postnetworks\elec_s_36_ec_lcopt_Co2L-6h__2030_0.071_AB_0export.nc",
#     "Scenario C": r"Output\Month_libya_h2export_test\scenario_c\No_Hydrogen_H2_0\postnetworks\elec_s_36_ec_lcopt_Co2L-6h__2030_0.071_AB_0export.nc",
#           }



OUT_DIR = "viz_outputs/Month_h2_scenario_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

#### Map raw PyPSA carrier names -> display categories

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
    "H2": "H2 Storage",                   
    "H2 store": "H2 Storage",             
    "H2 Store": "H2 Storage",
    "H2 Store Tank": "H2 Storage",
    "H2 Store Underground": "H2 Storage",
    "H2 UHS": "H2 Storage",
    "H2 Fuel Cell": "H2 Fuel Cell",
    # Conventional Thermal Generation (Added to prevent silent exclusion)
    "CCGT": "CCGT",
    "OCGT": "OCGT",
    "oil": "Oil",
    "heavy fuel oil": "Oil",
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
    "CCGT": "#7f8c8d",                     # Added visual color
    "OCGT": "#e74c3c",                     # Added visual color
    "Oil": "#8b4513",                      # Added visual color
    "Operating Cost": "#d2b48c",
}



# =====================================================================
# CARTOPY & MAP AXIS CONFIGURATION
# =====================================================================

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: cartopy not found. Falling back to simple coordinate plotting.")


def setup_map_ax(fig):
    """Initializes a map axis with Libya borders using Cartopy if available."""
    if HAS_CARTOPY:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent([9.0, 25.5, 19.0, 33.8], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#f4f4f0", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="#e0f2fe", zorder=0)
        ax.add_feature(cfeature.BORDERS, linestyle="--", edgecolor="#666666", linewidth=0.8, zorder=1)
        ax.add_feature(cfeature.COASTLINE, edgecolor="#333333", linewidth=1.0, zorder=1)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle=":")
        gl.top_labels = False
        gl.right_labels = False
        return ax
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(9.0, 25.5)
        ax.set_ylim(19.0, 33.8)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle=":", alpha=0.5)
        return ax
    
    
    
    
# =====================================================================
# COST EXTRACTION FUNCTIONS (FIXED)
# =====================================================================

def capex_by_category(n: pypsa.Network) -> pd.Series:
    """Annualized capital cost (capital_cost x optimal capacity) grouped by category.
    Uses DataFrame records to avoid Series index collisions and unmapped key drops.
    """
    records = []

    # Helper to sanitize and map carrier strings cleanly
    def clean_map(carrier_series):
        return carrier_series.astype(str).str.strip().map(lambda c: CAPEX_CARRIER_MAP.get(c, c))

    if not n.generators.empty:
        g = n.generators
        cost = g.capital_cost * g.p_nom_opt
        records.append(pd.DataFrame({"category": clean_map(g.carrier), "cost": cost.values}))

    if not n.links.empty:
        l = n.links
        cost = l.capital_cost * l.p_nom_opt
        records.append(pd.DataFrame({"category": clean_map(l.carrier), "cost": cost.values}))

    if not n.stores.empty:
        s = n.stores
        cost = s.capital_cost * s.e_nom_opt
        records.append(pd.DataFrame({"category": clean_map(s.carrier), "cost": cost.values}))

    if not n.storage_units.empty:
        su = n.storage_units
        cost = su.capital_cost * su.p_nom_opt
        records.append(pd.DataFrame({"category": clean_map(su.carrier), "cost": cost.values}))

    if not n.lines.empty and "capital_cost" in n.lines:
        ln = n.lines
        cost = ln.capital_cost * (ln.s_nom_opt - ln.s_nom).clip(lower=0)
        records.append(pd.DataFrame({"category": "H2 pipeline", "cost": cost.values}))

    if not records:
        return pd.Series(dtype=float)

    # Combine all component records into a single structured DataFrame
    df_all = pd.concat(records, ignore_index=True)
    
    # Filter to valid defined chart categories and aggregate sum per category
    df_mapped = df_all[df_all["category"].isin(CHART_COLORS.keys())]
    return df_mapped.groupby("category")["cost"].sum()


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


# =====================================================================
# COST COMPUTATION & FIGURE 9A GENERATION
# =====================================================================

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
                          ["Solar PV", "Onshore Wind", "Offshore Wind", "CCGT", "OCGT", "Oil",
                           "Electrolysis", "H2 Storage", "H2 Fuel Cell", "SMR", "SMR with CC", 
                           "CO2 sequestration", "H2 pipeline", "Operating Cost"]
                          if c in df.columns]
    
    df = df[category_order]

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = pd.Series(0.0, index=df.index)

    for cat in category_order:
        values = df[cat]
        ax.bar(df.index, values, bottom=bottom,
               color=CHART_COLORS.get(cat, "#95a5a6"), label=cat,
               edgecolor="black", linewidth=0.5)
        bottom += values

    ax.set_ylabel("Annualized System Cost [Billion EUR]", fontsize=11)
    ax.set_title("PyPSA-Libya: Export Hydrogen System Cost Comparison\nScenarios (a), (b), and (c)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
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
        
        # fig, ax = plt.subplots(figsize=(10, 8))
        fig = plt.figure(figsize=(10, 8))
        ax = setup_map_ax(fig)
        
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
        
        # fig, ax = plt.subplots(figsize=(10, 8))
        fig = plt.figure(figsize=(10, 8))
        ax = setup_map_ax(fig)
        
        # Calculate total power generation & conversion capacity in MW (excluding MWh storage AND legacy Oil dummy limits)
        gen_cols = [c for c in pivot_cap.columns if c not in ["H2 Storage", "Oil"]]
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
        
        # fig = plt.figure(figsize=(11, 8))
        # ax = fig.add_subplot(1, 1, 1)
        # ax.set_xlim(9.0, 25.5)
        # ax.set_ylim(19.0, 33.8)
        # ax.set_xlabel("Longitude")
        # ax.set_ylabel("Latitude")
        # ax.grid(True, linestyle=":", alpha=0.5)
        
        fig = plt.figure(figsize=(11, 8))
        ax = setup_map_ax(fig)

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


# Dynamically construct Baseline Reference Networks (0 TWh H2 Export) for Option B Net-Cost Allocation
# derived directly from the main RESULTS paths defined at the top
BASELINE_RESULTS = {}
for scenario_name, path in RESULTS.items():
    # Insert 'No_Hydrogen_H2_0' folder before 'postnetworks' and switch export suffix to '0export.nc'
    path_0export = path.replace(r"\postnetworks", r"\No_Hydrogen_H2_0\postnetworks")
    path_0export = re.sub(r'(\d+)export\.nc$', r'0export.nc', path_0export)
    BASELINE_RESULTS[scenario_name] = path_0export
    
    
    
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
    
    
    


# =====================================================================
# 4. AUGUST MONTHLY DISPATCH TIME-SERIES CHARTS (6-HOUR / 3-HOUR RESOLUTION)
# =====================================================================

print("\n" + "=" * 70)
print("Generating August Monthly Dispatch Profiles for Scenarios (Matching Native Temporal Resolution)")
print("=" * 70)

def extract_total_dispatch(n: pypsa.Network) -> pd.DataFrame:
    """Extracts total dispatch per carrier for all generators and storage units."""
    # Generator dispatch
    gen_dispatch = n.generators_t.p.T.groupby(n.generators.carrier.map(lambda c: CAPEX_CARRIER_MAP.get(str(c).strip(), str(c).strip()))).sum().T / 1e3  # Convert MW to GW
    
    # Storage discharging dispatch
    if hasattr(n, "storage_units") and not n.storage_units.empty and not n.storage_units_t.p.empty:
        su_dispatch = n.storage_units_t.p.clip(lower=0).T.groupby(
            n.storage_units.carrier.map(lambda c: CAPEX_CARRIER_MAP.get(str(c).strip(), str(c).strip()))
        ).sum().T / 1e3
        total_dispatch = pd.concat([gen_dispatch, su_dispatch], axis=1).T.groupby(level=0).sum().T
    else:
        total_dispatch = gen_dispatch

    return total_dispatch


for scenario_name, path in RESULTS.items():
    if not os.path.exists(path):
        continue

    print(f"Generating August dispatch plot for {scenario_name}...")
    n_scen = pypsa.Network(path)

    # Calculate actual temporal resolution (3h or 6h) dynamically from snapshots
    dt_hours = (n_scen.snapshots[1] - n_scen.snapshots[0]).total_seconds() / 3600.0 if len(n_scen.snapshots) > 1 else 6.0

    # Extract dispatch matrix (GW)
    total_dispatch = extract_total_dispatch(n_scen)

    # Slice August period (August 1st to September 1st)
    year = n_scen.snapshots[0].year if isinstance(n_scen.snapshots, pd.DatetimeIndex) else 2030
    start_date = f"{year}-08-01"
    end_date = f"{year}-09-01"
    
    dispatch_aug = total_dispatch.loc[start_date:end_date]

    if dispatch_aug.empty:
        print(f"  [SKIP] No August data found for {scenario_name}.")
        continue

    # Filter active carriers with non-zero dispatch
    active_carriers = [c for c in dispatch_aug.columns if dispatch_aug[c].sum() > 1e-3]
    color_list = [CHART_COLORS.get(c, "gray") for c in active_carriers]

    # Plot Monthly Dispatch Stacked Bar Chart
    fig, ax = plt.subplots(figsize=(14, 6))

    dispatch_aug[active_carriers].plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=color_list,
        width=0.85,
        alpha=0.9
    )

    ax.set_ylabel("Power [GW]", fontsize=11)
    ax.set_xlabel(f"Time (August - {int(dt_hours)}h Resolution)", fontsize=11)
    ax.set_title(f"PyPSA-Libya — {scenario_name}: Hourly Generation & Dispatch Profile ({start_date} to {end_date})", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0), fontsize=9)

    # Format X-ticks to display every 8th step (matching visualize_libya_year.py format)
    n_ticks = max(1, len(dispatch_aug) // 8)
    ticks = range(0, len(dispatch_aug), n_ticks)
    labels = [dispatch_aug.index[i].strftime("%d %b %H:%M") for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=30, ha="right")

    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#cbd5e1")

    fig.tight_layout()

    safe_name = scenario_name.lower().replace(" ", "_")
    out_path = os.path.join(OUT_DIR, f"monthly_dispatch_august_bar_{safe_name}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved August bar chart -> {out_path}")
    
    


# =====================================================================
# 4. PRO-RATA (ENERGY & FUEL ALLOCATED) LCOH — COMPUTATION & EXPORT
# =====================================================================

print("\n" + "=" * 70)
print("Computing CORRECTED PRO-RATA LCOH across Scenarios")
print("=" * 70)

# ---------------------------------------------------------------------
# Explicit Network Component Category Mapping
# ---------------------------------------------------------------------
DEDICATED_H2_LINK_CARRIERS = [
    "H2 Electrolysis", "H2 Fuel Cell", "SMR", "SMR CC"
]
DEDICATED_H2_STORE_CARRIERS = [
    "H2", "H2 store", "H2 Store", "H2 Store Tank", 
    "H2 Store Underground", "H2 UHS"
]
DEDICATED_H2_LINE_CARRIERS = ["H2 pipeline", "H2 pipeline repurposed"]

SHARED_GEN_CARRIERS = ["solar", "solar rooftop", "onwind", "offwind"]
SHARED_LINK_CARRIERS = ["CCGT", "OCGT", "oil"]

GAS_BUS_CARRIERS = ["gas"]


def prorata_h2_cost_corrected(n: pypsa.Network) -> dict:
    """Calculates grounded Pro-Rata LCOH capturing gas fuel inputs, link-based 
    generation, and all dedicated hydrogen chain components.
    """
    weights = n.snapshot_weightings.generators

    # 1. DEDICATED HYDROGEN CHAIN COSTS (100% Attributed to H2)
    ded_links = n.links[n.links.carrier.isin(DEDICATED_H2_LINK_CARRIERS)]
    ded_link_capex = float((ded_links.capital_cost * ded_links.p_nom_opt).sum())
    
    ded_link_opex = 0.0
    if not n.links_t.p0.empty:
        sel_l = [c for c in ded_links.index if c in n.links_t.p0.columns]
        if sel_l:
            mc = n.links.marginal_cost.loc[sel_l]
            ded_link_opex = float((n.links_t.p0[sel_l].mul(mc, axis=1).mul(weights, axis=0)).sum().sum())

    ded_stores = n.stores[n.stores.carrier.isin(DEDICATED_H2_STORE_CARRIERS)] if not n.stores.empty else pd.DataFrame()
    ded_store_capex = float((ded_stores.capital_cost * ded_stores.e_nom_opt).sum()) if not ded_stores.empty else 0.0

    ded_pipe_capex = 0.0
    if not n.lines.empty and "capital_cost" in n.lines.columns:
        pipe_lines = n.lines[n.lines.carrier.isin(DEDICATED_H2_LINE_CARRIERS)] if "carrier" in n.lines.columns else pd.DataFrame()
        if not pipe_lines.empty:
            ded_pipe_capex = float((pipe_lines.capital_cost * (pipe_lines.s_nom_opt - pipe_lines.s_nom).clip(lower=0)).sum())

    total_dedicated_cost = ded_link_capex + ded_link_opex + ded_store_capex + ded_pipe_capex

    # 2. SHARED ELECTRICITY POOL COSTS & ALLOCATION
    shared_gens = n.generators[n.generators.carrier.isin(SHARED_GEN_CARRIERS)]
    shared_gen_capex = float((shared_gens.capital_cost * shared_gens.p_nom_opt).sum())
    
    shared_gen_opex = 0.0
    if not n.generators_t.p.empty:
        sel_g = [c for c in shared_gens.index if c in n.generators_t.p.columns]
        if sel_g:
            mc = n.generators.marginal_cost.loc[sel_g]
            shared_gen_opex = float((n.generators_t.p[sel_g].mul(mc, axis=1).mul(weights, axis=0)).sum().sum())

    shared_links = n.links[n.links.carrier.isin(SHARED_LINK_CARRIERS)]
    shared_link_capex = float((shared_links.capital_cost * shared_links.p_nom_opt).sum())
    
    shared_link_opex = 0.0
    if not n.links_t.p0.empty:
        sel_sl = [c for c in shared_links.index if c in n.links_t.p0.columns]
        if sel_sl:
            mc = n.links.marginal_cost.loc[sel_sl]
            shared_link_opex = float((n.links_t.p0[sel_sl].mul(mc, axis=1).mul(weights, axis=0)).sum().sum())

    total_shared_elec_pool = shared_gen_capex + shared_gen_opex + shared_link_capex + shared_link_opex

    elec_links = n.links[n.links.carrier == "H2 Electrolysis"].index
    elec_consumption_mwh = 0.0
    if not n.links_t.p0.empty:
        sel_el = [c for c in elec_links if c in n.links_t.p0.columns]
        if sel_el:
            elec_consumption_mwh = float(n.links_t.p0[sel_el].mul(weights, axis=0).sum().sum())

    total_elec_gen_mwh = 0.0
    if not n.generators_t.p.empty:
        sel_gen_all = [c for c in shared_gens.index if c in n.generators_t.p.columns]
        if sel_gen_all:
            total_elec_gen_mwh += float(n.generators_t.p[sel_gen_all].mul(weights, axis=0).sum().sum())

    elec_allocation_fraction = (elec_consumption_mwh / total_elec_gen_mwh) if total_elec_gen_mwh > 0 else 0.0
    allocated_elec_cost = elec_allocation_fraction * total_shared_elec_pool

    # 3. SHARED GAS FUEL COSTS & ALLOCATION
    gas_buses = n.buses[n.buses.carrier.isin(GAS_BUS_CARRIERS)].index
    gas_gens = n.generators[n.generators.bus.isin(gas_buses)]
    
    total_gas_cost = 0.0
    if not gas_gens.empty and not n.generators_t.p.empty:
        sel_gg = [c for c in gas_gens.index if c in n.generators_t.p.columns]
        if sel_gg:
            mc = n.generators.marginal_cost.loc[sel_gg]
            gas_capex = (n.generators.capital_cost.loc[sel_gg] * n.generators.p_nom_opt.loc[sel_gg]).sum()
            gas_opex = (n.generators_t.p[sel_gg].mul(mc, axis=1).mul(weights, axis=0)).sum().sum()
            total_gas_cost = float(gas_capex + gas_opex)

    smr_links = n.links[n.links.carrier.isin(["SMR", "SMR CC"])].index
    smr_gas_mwh = 0.0
    if not n.links_t.p0.empty:
        sel_smr = [c for c in smr_links if c in n.links_t.p0.columns]
        if sel_smr:
            smr_gas_mwh = float(n.links_t.p0[sel_smr].mul(weights, axis=0).sum().sum())

    ccgt_gas_mwh = 0.0
    if not n.links_t.p0.empty:
        sel_ccgt = [c for c in n.links[n.links.carrier == "CCGT"].index if c in n.links_t.p0.columns]
        if sel_ccgt:
            ccgt_gas_mwh = float(n.links_t.p0[sel_ccgt].mul(weights, axis=0).sum().sum())

    total_gas_consumed_mwh = smr_gas_mwh + ccgt_gas_mwh
    gas_allocation_fraction = (smr_gas_mwh / total_gas_consumed_mwh) if total_gas_consumed_mwh > 0 else 0.0
    allocated_gas_cost = gas_allocation_fraction * total_gas_cost

    # 4. TOTAL ALLOCATED PRO-RATA COST
    total_prorata_h2_cost = total_dedicated_cost + allocated_elec_cost + allocated_gas_cost

    return {
        "dedicated_cost": total_dedicated_cost,
        "shared_elec_pool_total": total_shared_elec_pool,
        "elec_allocation_fraction": elec_allocation_fraction,
        "allocated_elec_cost": allocated_elec_cost,
        "shared_gas_pool_total": total_gas_cost,
        "gas_allocation_fraction": gas_allocation_fraction,
        "allocated_gas_cost": allocated_gas_cost,
        "total_prorata_h2_cost_eur": total_prorata_h2_cost,
    }


# ---------------------------------------------------------------------
# EXECUTION AND EXPORT MECHANISM (Matches LCOH Summary)
# ---------------------------------------------------------------------
prorata_results_main = {}
prorata_results_summary = {}

for scenario_name, export_path in RESULTS.items():
    if not os.path.exists(export_path):
        continue

    n_export = pypsa.Network(export_path)
    res = prorata_h2_cost_corrected(n_export)

    # Extract Exported H2 Energy
    exported_h2_mwh = 0.0
    h2_export_loads = n_export.loads[n_export.loads.carrier.str.contains("H2|h2|export", case=False, na=False)]
    if not h2_export_loads.empty and not n_export.loads_t.p.empty:
        weights = n_export.snapshot_weightings.generators
        exported_h2_mwh = n_export.loads_t.p[h2_export_loads.index].mul(weights, axis=0).sum().sum()
    if exported_h2_mwh <= 0:
        exported_h2_mwh = DEFAULT_EXPORT_H2_MWH

    total_cost_eur = res["total_prorata_h2_cost_eur"]
    lcoh_mwh = total_cost_eur / exported_h2_mwh
    lcoh_lhv = lcoh_mwh * H2_LHV_MWH_PER_KG
    lcoh_hhv = lcoh_mwh * H2_HHV_MWH_PER_KG

    # Table 1: Cost Breakdown & Allocation Summary
    prorata_results_main[scenario_name] = {
        "Dedicated H2 Cost [B EUR]": res["dedicated_cost"] / 1e9,
        "Allocated Elec Cost [B EUR]": res["allocated_elec_cost"] / 1e9,
        "Allocated Gas Cost [B EUR]": res["allocated_gas_cost"] / 1e9,
        "Total Pro-Rata Cost [B EUR]": total_cost_eur / 1e9,
        "Exported H2 [TWh]": exported_h2_mwh / 1e6,
        "Elec Alloc Share [%]": res["elec_allocation_fraction"] * 100.0,
        "Gas Alloc Share [%]": res["gas_allocation_fraction"] * 100.0,
    }

    # Table 2: Dedicated Unit Metrics
    prorata_results_summary[scenario_name] = {
        "Pro-Rata LCOH [EUR/MWh]": lcoh_mwh,
        "Pro-Rata LCOH LHV [EUR/kg]": lcoh_lhv,
        "Pro-Rata LCOH HHV [EUR/kg]": lcoh_hhv,
    }

if prorata_results_main:
    prorata_main_df = pd.DataFrame(prorata_results_main).T
    prorata_summary_df = pd.DataFrame(prorata_results_summary).T

    print("\n" + "=" * 50 + "\nPRO-RATA ALLOCATED LCOH RESULTS\n" + "=" * 50)
    print(prorata_main_df.to_string())
    print("\n" + prorata_summary_df.to_string())

    # Save report with identical structure to LCOH_summary_results.txt
    prorata_txt_path = os.path.join(OUT_DIR, "Prorata_LCOH_summary_results.txt")
    with open(prorata_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 95 + "\n")
        f.write("PyPSA-Libya: Pro-Rata Levelized Cost of Hydrogen (LCOH) Export Results\n")
        f.write("=" * 95 + "\n\n")
        f.write("--- Table 1: Cost Allocation & Pro-Rata Cost Breakdown ---\n")
        f.write(prorata_main_df.to_string())
        f.write("\n\n")
        f.write("--- Table 2: Pro-Rata LCOH Unit Summary (MWh vs. LHV / HHV per kg) ---\n")
        f.write(prorata_summary_df.to_string())
        f.write("\n\n" + "=" * 95 + "\n")
        f.write("Notes:\n")
        f.write(" - Total Pro-Rata Cost = Dedicated Costs + Allocated Elec Pool Cost + Allocated Gas Fuel Cost\n")
        f.write(" - Pro-Rata LCOH = Total Pro-Rata Cost / Exported H2 Energy\n")
        f.write(f" - Assumed H2 Lower Heating Value (LHV) = {H2_LHV_MWH_PER_KG} MWh/kg (33.33 kWh/kg)\n")
        f.write(f" - Assumed H2 Higher Heating Value (HHV) = {H2_HHV_MWH_PER_KG} MWh/kg (39.41 kWh/kg)\n")
        f.write("=" * 95 + "\n")
    print(f"\nSaved Pro-Rata LCOH summary report -> {prorata_txt_path}")
    
    


# =====================================================================
# EXTRA SECTION: GENERATE 4 DETAILED OUTCOMES PER H2 SCENARIO
# (With Dynamic Threshold Color & Transparency Separation for Small vs Major Bars)
# =====================================================================

print("\n" + "=" * 70)
print("Generating Energy Carriers Analysis Outcomes")
print("=" * 70)

# 1. Target Subfolder Path Configuration
CARRIERS_DIR = os.path.join(OUT_DIR, "Energy Carriers Analysis")
os.makedirs(CARRIERS_DIR, exist_ok=True)

### User-adjustable threshold in GWh defining major vs small bars
# THRESHOLD_GWH = 3000
THRESHOLD_GWH = 1500

# Master Primary Color Palette aligned with Figure 9a (Major Bars)
PRIMARY_CARRIER_COLORS = {
    "Solar PV": "#f5a623",
    "solar": "#f5a623",
    "Onshore Wind": "#2ecc71",
    "onwind": "#2ecc71",
    "Offshore Wind": "#1abc9c",
    "offwind": "#1abc9c",
    "Electrolysis": "#7b2d8e",
    "H2 Electrolysis": "#7b2d8e",
    "H2 Storage": "#3498db",
    "H2": "#3498db",
    "H2 (charge)": "#2980b9",
    "H2 (discharge)": "#3498db",
    "H2 Fuel Cell": "#e67e22",
    "SMR": "#4F4F2F",
    "SMR with CC": "#1a5276",
    "SMR CC": "#1a5276",
    "H2 pipeline": "#c0392b",
    "CCGT": "#b03a2e",
    "OCGT": "#e74c3c",
    "Oil": "#8b4513",
    "oil": "#8b4513",
    "CO2 sequestration": "#0b1f3a",
    "electricity distribution grid": "#f9b7cc",
    "solar rooftop": "#e6007e",
    "gas for industry CC": "#16a085",
    "process emissions": "#606c38",
    "process emissions CC": "#9aa56b",
    "gas": "#7f8c8d",
}

# Distinct light, desaturated pastel palette reserved strictly for minor/small carriers (< THRESHOLD_GWH)
LIGHT_PASTEL_PALETTE = [
    "#e2e8f0", "#cbd5e1", "#f1f5f9", "#fee2e2", "#ffedd5",
    "#fef3c7", "#ecfccb", "#dcfce7", "#e0e7ff", "#f3e8ff"
]

def get_smart_carrier_style(carrier_name, is_major=True):
    """Returns (color, alpha, edgecolor, linewidth) to visually separate major vs minor bars."""
    if is_major:
        color = PRIMARY_CARRIER_COLORS.get(carrier_name)
        if not color:
            hash_val = sum(ord(c) for c in str(carrier_name))
            color = plt.cm.tab10(hash_val % 10)
        return color, 0.95, "black", 0.6
    else:
        # Use desaturated pastel palette and light alpha for minor/small bars
        hash_val = sum(ord(c) for c in str(carrier_name))
        light_color = LIGHT_PASTEL_PALETTE[hash_val % len(LIGHT_PASTEL_PALETTE)]
        return light_color, 0.50, "#64748b", 0.8

def map_outcome_carrier(carrier_str):
    """Normalizes raw PyPSA carrier strings to display categories."""
    c = str(carrier_str).strip()
    return CAPEX_CARRIER_MAP.get(c, c)


# -----------------------------------------------------------------
# 2. COMBINED RENEWABLE CF & CURTAILMENT TEXT REPORT
# -----------------------------------------------------------------
combined_ren_txt_path = os.path.join(CARRIERS_DIR, "Renewable_cf_and_curtailment.txt")

with open(combined_ren_txt_path, "w", encoding="utf-8") as f_comb:
    f_comb.write("=" * 95 + "\n")
    f_comb.write(" PyPSA-Libya — Combined Renewable Capacity Factor (CF) & Curtailment Report\n")
    f_comb.write("=" * 95 + "\n\n")

    for scenario_name, path in RESULTS.items():
        if not os.path.exists(path):
            continue

        n_scen = pypsa.Network(path)
        dt_hours = (n_scen.snapshots[1] - n_scen.snapshots[0]).total_seconds() / 3600.0 if len(n_scen.snapshots) > 1 else 1.0
        total_physical_hours = len(n_scen.snapshots) * dt_hours

        renewable_carriers = [c for c in ["solar", "solar rooftop", "onwind", "offwind"] if c in n_scen.generators.carrier.unique()]
        cf_curtailment_results = []

        for carrier in renewable_carriers:
            gens = n_scen.generators[n_scen.generators.carrier == carrier]
            if gens.empty:
                continue
            
            p_nom = gens["p_nom_opt"].fillna(gens.get("p_nom", 0)) if "p_nom_opt" in gens.columns else gens.get("p_nom", 0)
            total_cap_mw = p_nom.sum()
            actual_gen_gwh = (n_scen.generators_t.p[gens.index].sum().sum() * dt_hours) / 1e3
            
            if hasattr(n_scen.generators_t, "p_max_pu") and not n_scen.generators_t.p_max_pu.empty:
                cols_in_pu = [col for col in gens.index if col in n_scen.generators_t.p_max_pu.columns]
                potential_gen_gwh = (n_scen.generators_t.p_max_pu[cols_in_pu] * p_nom.loc[cols_in_pu]).sum().sum() * dt_hours / 1e3 if cols_in_pu else actual_gen_gwh
            else:
                potential_gen_gwh = actual_gen_gwh
                
            curtailed_gwh = max(0.0, potential_gen_gwh - actual_gen_gwh)
            curtailment_rate_pct = (curtailed_gwh / potential_gen_gwh * 100.0) if potential_gen_gwh > 0 else 0.0
            max_possible_gen_gwh = (total_cap_mw * total_physical_hours) / 1e3
            capacity_factor_pct = (actual_gen_gwh / max_possible_gen_gwh * 100.0) if max_possible_gen_gwh > 0 else 0.0
            
            cf_curtailment_results.append({
                "carrier": map_outcome_carrier(carrier),
                "capacity_mw": total_cap_mw,
                "potential_gwh": potential_gen_gwh,
                "actual_gwh": actual_gen_gwh,
                "curtailed_gwh": curtailed_gwh,
                "capacity_factor_pct": capacity_factor_pct,
                "curtailment_rate_pct": curtailment_rate_pct
            })

        df_renewables = pd.DataFrame(cf_curtailment_results)

        f_comb.write(f"===============================================================================================\n")
        f_comb.write(f" PyPSA-Libya — Renewable Capacity Factor (CF) & Curtailment Report ({scenario_name})\n")
        f_comb.write(f"===============================================================================================\n\n")
        f_comb.write(f"{'CARRIER':15s} | {'CAPACITY (MW)':14s} | {'POTENTIAL (GWh)':16s} | {'ACTUAL (GWh)':14s} | {'CURTAILED (GWh)':16s} | {'CF (%)':8s} | {'CURTAIL %':10s}\n")
        f_comb.write("-" * 95 + "\n")
        for _, row in df_renewables.iterrows():
            f_comb.write(f"{row['carrier']:15s} | {row['capacity_mw']:14,.1f} | {row['potential_gwh']:16,.1f} | {row['actual_gwh']:14,.1f} | {row['curtailed_gwh']:16,.1f} | {row['capacity_factor_pct']:7.2f}% | {row['curtailment_rate_pct']:9.2f}%\n")
        f_comb.write("-" * 95 + "\n\n\n")

print(f"Saved combined renewable report -> {combined_ren_txt_path}")


# -----------------------------------------------------------------
# 3. FIGURE GENERATION PER SCENARIO WITH THRESHOLD COLOR LOGIC
# -----------------------------------------------------------------
for scenario_name, path in RESULTS.items():
    if not os.path.exists(path):
        print(f"[SKIP Outcomes] {scenario_name}: File not found at '{path}'")
        continue

    print(f"\nProcessing Energy Carriers Analysis figures for: {scenario_name}...")
    n_scen = pypsa.Network(path)
    safe_prefix = scenario_name.replace(" ", "_")
    dt_hours = (n_scen.snapshots[1] - n_scen.snapshots[0]).total_seconds() / 3600.0 if len(n_scen.snapshots) > 1 else 1.0

    # -----------------------------------------------------------------
    # A. Physical Generation & Conversion Activity Computation
    # -----------------------------------------------------------------
    phys_gen_dict = {}

    if not n_scen.generators.empty and not n_scen.generators_t.p.empty:
        gen_p = n_scen.generators_t.p.groupby(n_scen.generators.carrier.map(map_outcome_carrier), axis=1).sum()
        for carrier in gen_p.columns:
            energy_gwh = (gen_p[carrier] * dt_hours).sum() / 1e3
            if energy_gwh > 1e-3:
                phys_gen_dict[carrier] = energy_gwh

    if not n_scen.links.empty and not n_scen.links_t.p0.empty:
        link_p = n_scen.links_t.p0.groupby(n_scen.links.carrier.map(map_outcome_carrier), axis=1).sum()
        for carrier in link_p.columns:
            energy_gwh = (link_p[carrier] * dt_hours).sum() / 1e3
            if energy_gwh > 1e-3:
                phys_gen_dict[carrier] = energy_gwh

    if not n_scen.storage_units.empty and not n_scen.storage_units_t.p.empty:
        su_p = n_scen.storage_units_t.p
        su_carrier_mapped = n_scen.storage_units.carrier.map(map_outcome_carrier)
        for carrier in su_carrier_mapped.unique():
            cols = n_scen.storage_units[su_carrier_mapped == carrier].index
            p_carrier = su_p[cols]
            dis_gwh = (p_carrier.clip(lower=0).sum(axis=1) * dt_hours).sum() / 1e3
            if dis_gwh > 1e-3:
                phys_gen_dict[f"{carrier} (discharge)"] = dis_gwh
            chg_gwh = ((-p_carrier.clip(upper=0)).sum(axis=1) * dt_hours).sum() / 1e3
            if chg_gwh > 1e-3:
                phys_gen_dict[f"{carrier} (charge)"] = chg_gwh

    phys_gen_gwh = pd.Series(phys_gen_dict)

    # Classification of major vs minor carriers based on THRESHOLD_GWH
    major_carriers = phys_gen_gwh[phys_gen_gwh >= THRESHOLD_GWH].index.tolist()

    def check_is_major(c_name):
        base_name = str(c_name).replace(" (discharge)", "").replace(" (charge)", "")
        return (c_name in major_carriers) or (base_name in major_carriers)

    # -----------------------------------------------------------------
    # B. Solved Capacity Charts (Full & Filtered)
    # -----------------------------------------------------------------
    cap_records = []
    
    if not n_scen.generators.empty:
        gen_df = n_scen.generators[n_scen.generators.carrier != "load shedding"].copy()
        cap = gen_df["p_nom_opt"].fillna(gen_df.get("p_nom", 0)) if "p_nom_opt" in gen_df.columns else gen_df.get("p_nom", 0)
        for idx, row in gen_df.iterrows():
            if cap.loc[idx] > 1e-3:
                cap_records.append({"carrier": map_outcome_carrier(row.carrier), "capacity": cap.loc[idx]})
    
    if not n_scen.links.empty:
        link_df = n_scen.links.copy()
        cap = link_df["p_nom_opt"].fillna(link_df.get("p_nom", 0)) if "p_nom_opt" in link_df.columns else link_df.get("p_nom", 0)
        for idx, row in link_df.iterrows():
            if cap.loc[idx] > 1e-3:
                cap_records.append({"carrier": map_outcome_carrier(row.carrier), "capacity": cap.loc[idx]})

    if not n_scen.storage_units.empty:
        su_df = n_scen.storage_units.copy()
        cap = su_df["p_nom_opt"].fillna(su_df.get("p_nom", 0)) if "p_nom_opt" in su_df.columns else su_df.get("p_nom", 0)
        for idx, row in su_df.iterrows():
            if cap.loc[idx] > 1e-3:
                cap_records.append({"carrier": map_outcome_carrier(row.carrier), "capacity": cap.loc[idx]})

    df_cap = pd.DataFrame(cap_records)
    
    if not df_cap.empty:
        all_power_cap = df_cap.groupby("carrier")["capacity"].sum().sort_values()
        all_power_cap = all_power_cap[all_power_cap > 1e-2]

        # 1. Full Solved Capacity Chart with dynamic alpha separation
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (carrier, val) in enumerate(all_power_cap.items()):
            is_maj = check_is_major(carrier)
            color, alpha, edge_c, lw = get_smart_carrier_style(carrier, is_major=is_maj)
            ax.barh(carrier, val, color=color, alpha=alpha, edgecolor=edge_c, linewidth=lw)

        ax.set_xlabel("Optimal Installed Power Capacity (MW)")
        ax.set_title(f"PyPSA-Libya — Solved Power Capacity by Carrier ({scenario_name})", fontsize=12, fontweight="bold")
        ax.margins(x=0.20)
        ax.set_axisbelow(True)
        ax.grid(axis='x', linestyle='--', alpha=0.5, color='#cbd5e1')
        
        # Add labels after plotting
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f'{width:,.1f} MW', (width, p.get_y() + p.get_height() / 2.),
                        xytext=(5, 0), textcoords='offset points', ha='left', va='center', fontsize=9)

        fig.tight_layout()
        out_cap_png = os.path.join(CARRIERS_DIR, f"{safe_prefix}_solved_capacity_by_carrier.png")
        fig.savefig(out_cap_png, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 2. Filtered Solved Capacity Chart (Major carriers only)
        filtered_cap = all_power_cap[all_power_cap.index.map(check_is_major)]
        if not filtered_cap.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            for carrier, val in filtered_cap.items():
                color, alpha, edge_c, lw = get_smart_carrier_style(carrier, is_major=True)
                ax.barh(carrier, val, color=color, alpha=alpha, edgecolor=edge_c, linewidth=lw)

            ax.set_xlabel("Optimal Installed Power Capacity (MW)")
            ax.set_title(f"PyPSA-Libya — Solved Capacity by Carrier [>= {THRESHOLD_GWH:,.0f} GWh Gen] ({scenario_name})", fontsize=12, fontweight="bold")
            ax.margins(x=0.20)
            ax.set_axisbelow(True)
            ax.grid(axis='x', linestyle='--', alpha=0.5, color='#cbd5e1')

            for p in ax.patches:
                width = p.get_width()
                ax.annotate(f'{width:,.1f} MW', (width, p.get_y() + p.get_height() / 2.),
                            xytext=(5, 0), textcoords='offset points', ha='left', va='center', fontsize=9)

            fig.tight_layout()
            out_cap_filt_png = os.path.join(CARRIERS_DIR, f"{safe_prefix}_solved_capacity_by_carrier_filtered.png")
            fig.savefig(out_cap_filt_png, dpi=150, bbox_inches="tight")
            plt.close(fig)

    # -----------------------------------------------------------------
    # C. Physical Generation Charts (Full & Filtered)
    # -----------------------------------------------------------------
    if not phys_gen_gwh.empty:
        # 1. Full Physical Generation Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        for carrier, val in phys_gen_gwh.items():
            is_maj = check_is_major(carrier)
            color, alpha, edge_c, lw = get_smart_carrier_style(carrier, is_major=is_maj)
            ax.bar(carrier, val, color=color, alpha=alpha, edgecolor=edge_c, linewidth=lw)

        ax.set_ylabel("Physical Generation / Conversion Activity (GWh)", fontsize=11)
        ax.set_title(f"PyPSA-Libya — Physical Energy Generation & Storage Activity ({scenario_name})", fontsize=12, fontweight="bold")
        plt.xticks(rotation=35, ha='right')
        ax.margins(y=0.18)
        ax.set_axisbelow(True)
        ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cbd5e1')

        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:,.1f}', (p.get_x() + p.get_width() / 2., height),
                        xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontsize=9)

        fig.tight_layout()
        out_gen_png = os.path.join(CARRIERS_DIR, f"{safe_prefix}_physical_generation_by_carrier.png")
        fig.savefig(out_gen_png, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 2. Filtered Physical Generation Chart (Major carriers only)
        filtered_gen = phys_gen_gwh[phys_gen_gwh >= THRESHOLD_GWH]
        if not filtered_gen.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            for carrier, val in filtered_gen.items():
                color, alpha, edge_c, lw = get_smart_carrier_style(carrier, is_major=True)
                ax.bar(carrier, val, color=color, alpha=alpha, edgecolor=edge_c, linewidth=lw)

            ax.set_ylabel("Physical Generation / Conversion Activity (GWh)", fontsize=11)
            ax.set_title(f"PyPSA-Libya — Physical Generation & Activity [>= {THRESHOLD_GWH:,.0f} GWh] ({scenario_name})", fontsize=12, fontweight="bold")
            plt.xticks(rotation=25, ha='right')
            ax.margins(y=0.18)
            ax.set_axisbelow(True)
            ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cbd5e1')

            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{height:,.1f}', (p.get_x() + p.get_width() / 2., height),
                            xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontsize=9)

            fig.tight_layout()
            out_gen_filt_png = os.path.join(CARRIERS_DIR, f"{safe_prefix}_physical_generation_by_carrier_filtered.png")
            fig.savefig(out_gen_filt_png, dpi=150, bbox_inches="tight")
            plt.close(fig)

    # -----------------------------------------------------------------
    # D. Renewable Performance Chart (UNIFIED CARRIER COLORS)
    # -----------------------------------------------------------------
    renewable_carriers = [c for c in ["solar", "solar rooftop", "onwind", "offwind"] if c in n_scen.generators.carrier.unique()]
    cf_curtailment_results = []

    for carrier in renewable_carriers:
        gens = n_scen.generators[n_scen.generators.carrier == carrier]
        if gens.empty:
            continue
        
        p_nom = gens["p_nom_opt"].fillna(gens.get("p_nom", 0)) if "p_nom_opt" in gens.columns else gens.get("p_nom", 0)
        total_cap_mw = p_nom.sum()
        actual_gen_gwh = (n_scen.generators_t.p[gens.index].sum().sum() * dt_hours) / 1e3
        
        if hasattr(n_scen.generators_t, "p_max_pu") and not n_scen.generators_t.p_max_pu.empty:
            cols_in_pu = [col for col in gens.index if col in n_scen.generators_t.p_max_pu.columns]
            potential_gen_gwh = (n_scen.generators_t.p_max_pu[cols_in_pu] * p_nom.loc[cols_in_pu]).sum().sum() * dt_hours / 1e3 if cols_in_pu else actual_gen_gwh
        else:
            potential_gen_gwh = actual_gen_gwh
            
        curtailed_gwh = max(0.0, potential_gen_gwh - actual_gen_gwh)
        curtailment_rate_pct = (curtailed_gwh / potential_gen_gwh * 100.0) if potential_gen_gwh > 0 else 0.0
        max_possible_gen_gwh = (total_cap_mw * len(n_scen.snapshots) * dt_hours) / 1e3
        capacity_factor_pct = (actual_gen_gwh / max_possible_gen_gwh * 100.0) if max_possible_gen_gwh > 0 else 0.0
        
        cf_curtailment_results.append({
            "carrier": map_outcome_carrier(carrier),
            "capacity_factor_pct": capacity_factor_pct,
            "curtailment_rate_pct": curtailment_rate_pct
        })

    df_ren_plot = pd.DataFrame(cf_curtailment_results)

    if not df_ren_plot.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Shared carrier color list for both subplots
        carrier_colors = [get_smart_carrier_style(c, is_major=True)[0] for c in df_ren_plot["carrier"]]

        # Left Subfigure: Capacity Factor
        bars1 = ax1.bar(df_ren_plot["carrier"], df_ren_plot["capacity_factor_pct"], 
                        color=carrier_colors, width=0.5)
        ax1.set_ylabel("Capacity Factor (%)")
        ax1.set_title("Capacity Factor (CF) by Carrier")
        ax1.bar_label(bars1, fmt='%.2f%%', padding=3, fontsize=9)
        ax1.margins(y=0.2)
        ax1.set_axisbelow(True)
        ax1.grid(axis='y', linestyle='--', alpha=0.5, color='#cbd5e1')

        # Right Subfigure: Curtailment Rate (matching technology colors)
        bars2 = ax2.bar(df_ren_plot["carrier"], df_ren_plot["curtailment_rate_pct"], 
                        color=carrier_colors, width=0.5)
        ax2.set_ylabel("Curtailment Rate (%)")
        ax2.set_title("Curtailment Rate (%) by Carrier")
        ax2.bar_label(bars2, fmt='%.2f%%', padding=3, fontsize=9)
        ax2.margins(y=0.2)
        ax2.set_axisbelow(True)
        ax2.grid(axis='y', linestyle='--', alpha=0.5, color='#cbd5e1')

        fig.suptitle(f"PyPSA-Libya — Renewable Energy Performance & Curtailment ({scenario_name})", fontsize=12, fontweight='bold')
        fig.tight_layout()
        ren_png_path = os.path.join(CARRIERS_DIR, f"{safe_prefix}_renewable_cf_and_curtailment.png")
        fig.savefig(ren_png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

print(f"\nAll updated outputs successfully generated in: {CARRIERS_DIR}")


