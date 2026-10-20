"""
compute_h2_scenario_costs.py

Plots and compares total system costs and spatial investments directly across 
the 3 exported hydrogen scenarios (a, b, and c) for PyPSA-Libya.

Outputs generated in `viz_outputs/libya_h2export_hybrid_emispr_100/`:
  1. figure9a_cost_by_scenario.png (Stacked bar chart comparing Scenarios a, b, c)
  2. figure9b_investment_map_scenario_a.png (Spatial investment map for Scenario a)
  3. figure9b_investment_map_scenario_b.png (Spatial investment map for Scenario b)
  4. figure9b_investment_map_scenario_c.png (Spatial investment map for Scenario c)

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


###### H2 export, year, 3-hrs, 36 clusters
  
# RESULTS = {
#     "Hybrid": r"Output\libya_h2export_hybrid_emispr_100\3hrs\postnetworks\elec_s_36_ec_lcopt_Co2L-Ep-3h__2030_0.071_AB_70export.nc",
# }


##### H2 export, year, 3-hrs 
  
RESULTS = {
    "Hybrid": r"Output\libya_h2export_hybrid_emispr_100\3hrs\postnetworks\elec_s_12_ec_lcopt_Co2L-Ep-3h__2030_0.071_AB_70export.nc",
}




##### No H2 export, year, 3-hrs 

# RESULTS = {
    # "Hybrid": r"Output\libya_h2export_hybrid_emispr_100\3hrs\No_Hydrogen_H2_0\postnetworks\elec_s_12_ec_lcopt_Co2L-Ep-3h__2030_0.071_AB_0export.nc",
# }




CO2_TAX_EUR_PER_TON = 100   #ADJUSTABLE: EUR per tonne of vented (atmospheric) CO2

#### CO2_TAX_EUR_PER_TON = 0  
 

YEARS = np.arange(2030, 2051)                  
PROJECT_LIFETIME_YEARS = len(YEARS)  # 21 years inclusive
DEFAULT_WACC = 0.071                 # Baseline Discount Rate: 7.10%
REVENUE_MARGIN = 1.00                # EUR/kg added to LCOH



OUT_DIR = "viz_outputs/libya_h2export_hybrid_emispr_100"
os.makedirs(OUT_DIR, exist_ok=True)

#### --- THIS FOR PAPER PDF SUPPORT ---
PAPER_DIR = os.path.join(OUT_DIR, "Paper_Figures")
os.makedirs(PAPER_DIR, exist_ok=True)

def save_paper_pdf(fig, ax, filename):
    """
    Automatically clears the title and saves a high-res PDF to the Paper_Figures folder.
    Optimizes map file sizes by rasterizing geographic layers while keeping text as vectors.
    """
    original_title = ax.get_title() # Store the title
    ax.set_title("")                # Clear the title for the paper version
    
    # Apply naming logic
    clean_name = re.sub(r'^figure\d+[a-z]?_', '', filename)
    if not clean_name.startswith('fig_'):
        clean_name = 'fig_' + clean_name
    if not clean_name.endswith('.pdf'):
        clean_name += '.pdf'
        
    pdf_path = os.path.join(PAPER_DIR, clean_name)
    
    # --- OPTIMIZATION FOR MAPS ---
    # If the file is a map, we rasterize layers with low z-orders (the map/lines)
    # but keep z-orders >= 3 (the bubbles and text) as vector.
    if "map" in filename.lower():
        # Geographic features in setup_map_ax use zorder 0-1. 
        # Transmission lines in the script use zorder 1.
        # Bubbles start at zorder 3.
        ax.set_rasterization_zorder(2.5) 
    
    # Save with dpi=300 to ensure the rasterized map background looks sharp
    fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches="tight")
    
    ax.set_title(original_title)    # Restore title for the PNG/display
    print(f" Saved Paper PDF (Optimized): {clean_name}")
    
    

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
    "electricity distribution grid": "Grid Expansion",   
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
    "H2 Fuel Cell": "cyan",
    "CCGT": "#7f8c8d",                     
    "OCGT": "#e74c3c",                     
    "Oil": "#8b4513",                      
    "Operating Cost": "#d2b48c",
    # "CO2 Tax": "#c0392b",
    "CO2 Tax": "#000000",
    "Grid Expansion": "#f9b7cc",  
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
# CO2 TAX CONFIGURATION (Applies everywhere CO2 cost impact is shown)
# =====================================================================


def compute_vented_co2_mt(n: pypsa.Network, fallback_mt: float = 0.0) -> float:
    """Net Atmospheric (Vented) CO2 emissions in Mt for any given network.
    Includes BOTH Generator-based emissions (OCGT/CCGT/oil where modeled as
    Generators) AND Link-based emissions (SMR, SMR CC's vented remainder,
    oil-as-Link, gas for industry, etc. -- any Link whose bus2/3/4 is
    'co2 atmosphere'). Mirrors the corrected logic used in Section 7."""
    weights = n.snapshot_weightings.generators
    co2_vented_mt = 0.0

    # Generator-based emissions (electricity-only technologies, if modeled as Generators)
    if not n.generators.empty and not n.generators_t.p.empty:
        gen_emissions = 0.0
        for gen_id, gen in n.generators.iterrows():
            if gen_id in n.generators_t.p.columns:
                p_gen = n.generators_t.p[gen_id].mul(weights, axis=0).sum()
                efficiency = getattr(gen, "efficiency", 1.0) or 1.0
                co2_emissions_attr = getattr(gen, "co2_emissions", 0.0)
                gen_emissions += (p_gen / efficiency) * co2_emissions_attr
        co2_vented_mt += gen_emissions / 1e6

    # Link-based emissions (SMR, SMR CC's vented share, oil, gas for industry, etc.)
    if not n.links.empty and not n.links_t.p0.empty:
        link_vented_mt = 0.0
        for i in [2, 3, 4]:
            bus_col, eff_col = f"bus{i}", f"efficiency{i}"
            if bus_col not in n.links.columns or eff_col not in n.links.columns:
                continue
            mask = n.links[bus_col] == "co2 atmosphere"
            sel = [l for l in n.links[mask].index if l in n.links_t.p0.columns]
            if sel:
                flow = n.links_t.p0[sel].mul(weights, axis=0).sum()
                link_vented_mt += (flow * n.links.loc[sel, eff_col]).sum() / 1e6
        co2_vented_mt += link_vented_mt

    if co2_vented_mt <= 0:
        co2_vented_mt = fallback_mt

    return co2_vented_mt

def co2_tax_cost_eur(n: pypsa.Network, fallback_mt: float = 0.0) -> float:
    """Annual CO2 tax cost in EUR for a given network."""
    return compute_vented_co2_mt(n, fallback_mt) * 1e6 * CO2_TAX_EUR_PER_TON

# Known reference vented emissions (used only if a network has no CO2 data) — export scenarios only
CO2_VENTED_FALLBACK_EXPORT = {"Hybrid": 0.0}



# =====================================================================
# CO2 EMISSIONS & SEQUESTRATION MASS BALANCE ACCOUNTING SUMMARY
# =====================================================================

print("\n" + "=" * 70)
print("Computing CO2 Mass Balance Accounting across Scenarios")
print("=" * 70)

# Mirrors the RUN_CASES pattern used in the cost-computation section below,
# so BOTH the export and no-export networks get their own CO2 mass-balance
# report, instead of only ever computing it for the export network.
CO2_RUN_CASES = {
    "H2_Export": {
        "results": RESULTS,
        "prefix": "",
    },
    "No_H2_Export": {
        "results": {
            "Hybrid": r"Output\libya_h2export_hybrid_emispr_100\3hrs\No_Hydrogen_H2_0\postnetworks\elec_s_12_ec_lcopt_Co2L-Ep-3h__2030_0.071_AB_0export.nc",
        },
        "prefix": "No_H2_export_",
    },
}

for co2_case_key, co2_case_info in CO2_RUN_CASES.items():
    co2_results_dict = co2_case_info["results"]
    co2_prefix = co2_case_info["prefix"]

    co2_summary_results = {}

    for scenario_name, export_path in co2_results_dict.items():
        if not os.path.exists(export_path):
            continue

        n_scen = pypsa.Network(export_path)
        weights = n_scen.snapshot_weightings.generators

        # 1. Geologically Sequestered CO2 (Captured)
        co2_captured_mt = 0.0
    
        # Check Stores for Sequestered CO2
        co2_stores = n_scen.stores[n_scen.stores.carrier.str.contains("co2|sequestered", case=False, na=False)]
        if not co2_stores.empty and not n_scen.stores_t.e.empty:
            sel_stores = [s for s in co2_stores.index if s in n_scen.stores_t.e.columns]
            if sel_stores:
                co2_captured_mt = (n_scen.stores_t.e[sel_stores].iloc[-1] - n_scen.stores_t.e[sel_stores].iloc[0]).sum() / 1e6
                if co2_captured_mt <= 0:
                    co2_captured_mt = n_scen.stores_t.e[sel_stores].max().sum() / 1e6

        # Fallback to Link activity if store tracking isn't explicitly used
        if co2_captured_mt <= 0 and not n_scen.links.empty and not n_scen.links_t.p0.empty:
            co2_links = n_scen.links[n_scen.links.carrier.str.contains("co2|sequestered|cc", case=False, na=False)]
            sel_co2_links = [l for l in co2_links.index if l in n_scen.links_t.p0.columns]
            if sel_co2_links:
                co2_captured_mt = n_scen.links_t.p0[sel_co2_links].mul(weights, axis=0).sum().sum() / 1e6

        # 2. Net Atmospheric CO2 Emissions (Vented)
        co2_vented_mt = 0.0
    
        # Calculate emissions from Global Constraints / Carriers / Generator activity
        if hasattr(n_scen, "global_constraints") and "co2_limit" in n_scen.global_constraints.index:
            if hasattr(n_scen, "emissions_t") and not n_scen.emissions_t.empty:
                co2_vented_mt = n_scen.emissions_t.sum().sum() / 1e6

        if co2_vented_mt <= 0 and not n_scen.generators.empty and not n_scen.generators_t.p.empty:
            gen_emissions = 0.0
            for gen_id, gen in n_scen.generators.iterrows():
                if gen_id in n_scen.generators_t.p.columns:
                    p_gen = n_scen.generators_t.p[gen_id].mul(weights, axis=0).sum()
                    efficiency = getattr(gen, "efficiency", 1.0) or 1.0
                    co2_emissions_attr = getattr(gen, "co2_emissions", 0.0)
                    gen_emissions += (p_gen / efficiency) * co2_emissions_attr
            co2_vented_mt += gen_emissions / 1e6

        # Link-based vented emissions with carrier-level breakdown tracking
        vented_by_carrier = {}
        if not n_scen.links.empty and not n_scen.links_t.p0.empty:
            link_vented_mt = 0.0
            for i in [2, 3, 4]:
                bus_col, eff_col = f"bus{i}", f"efficiency{i}"
                if bus_col not in n_scen.links.columns or eff_col not in n_scen.links.columns:
                    continue
                mask = n_scen.links[bus_col] == "co2 atmosphere"
                sel = [l for l in n_scen.links[mask].index if l in n_scen.links_t.p0.columns]
            
                for l in sel:
                    carrier = n_scen.links.at[l, "carrier"]
                    flow = n_scen.links_t.p0[l].mul(weights, axis=0).sum()
                    eff = n_scen.links.at[l, eff_col]
                    emission_mt = (flow * eff) / 1e6
                
                    vented_by_carrier[carrier] = vented_by_carrier.get(carrier, 0.0) + emission_mt
                    link_vented_mt += emission_mt
                
            co2_vented_mt += link_vented_mt

        # Fallbacks
        if co2_vented_mt <= 0:
            fallback_emissions = {"Hybrid": 0.0}
            co2_vented_mt = fallback_emissions.get(scenario_name, 0.0)

        if co2_captured_mt <= 0:
            fallback_captured = {"Hybrid": 0.0}
            co2_captured_mt = fallback_captured.get(scenario_name, 0.0)

        # 3. Total Gross CO2 Generated
        total_gross_co2_mt = co2_vented_mt + co2_captured_mt

        co2_summary_results[scenario_name] = {
            "vented": co2_vented_mt,
            "captured": co2_captured_mt,
            "total": total_gross_co2_mt,
            "carrier_breakdown": vented_by_carrier
        }

    # =====================================================================
    # SAVE STRUCTURED SUMMARY TO TXT FILE
    # =====================================================================

    co2_txt_filename = f"CO2_emissions_{co2_prefix}summary.txt" if co2_prefix else "CO2_emissions_summary.txt"
    co2_txt_path = os.path.join(OUT_DIR, co2_txt_filename)
    scenarios = list(co2_results_dict.keys())

    with open(co2_txt_path, "w", encoding="utf-8") as f:
        f.write("Recommended Table for Your Results Section\n")
        f.write("To make your report clear and avoid confusion, present both rows under a structured CO2 mass balance header:\n\n")
    
        # 1. OVERALL MASS BALANCE TABLE
        header = f"{'CO2 Accounting Metric':45s} | {'Scenario A':12s}\n"
        f.write(header)
        f.write("-" * 65 + "\n")
    
        vented_row = f"{'Net Atmospheric CO2 Emissions (Vented)':45s} | " + " | ".join(
            [f"{co2_summary_results.get(s, {}).get('vented', 0.0):8.2f} Mt" for s in scenarios]
        ) + "\n"
        f.write(vented_row)
    
        captured_row = f"{'Geologically Sequestered CO2 (Captured)':45s} | " + " | ".join(
            [f"{co2_summary_results.get(s, {}).get('captured', 0.0):8.2f} Mt" for s in scenarios]
        ) + "\n"
        f.write(captured_row)
    
        total_row = f"{'Total Gross CO2 Generated (Vented + Stored)':45s} | " + " | ".join(
            [f"{co2_summary_results.get(s, {}).get('total', 0.0):8.2f} Mt" for s in scenarios]
        ) + "\n"
        f.write(total_row)
        f.write("-" * 65 + "\n\n")

        # 2. DETAILED VENTED BREAKDOWN TABLE
        f.write("Detailed Breakdown of Vented Emissions by Technology/Carrier:\n")
        f.write(f"{'Carrier / Technology':45s} | {'Vented CO2':12s}\n")
        f.write("-" * 65 + "\n")

        for s in scenarios:
            breakdown = co2_summary_results.get(s, {}).get('carrier_breakdown', {})
            sorted_carriers = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
        
            for carrier, val_mt in sorted_carriers:
                if round(val_mt, 2) > 0.0:
                    f.write(f"{carrier:45s} | {val_mt:8.2f} Mt\n")
                
        f.write("-" * 65 + "\n")

    print(f"Saved CO2 emissions summary table -> {co2_txt_path}")


# =====================================================================
# COST COMPUTATION & FIGURE 9A GENERATION (H2 EXPORT & NO H2 EXPORT)
# =====================================================================

RUN_CASES = {
    "H2_Export": {
        "results": RESULTS,
        "prefix": "",
    },
    "No_H2_Export": {
        "results": {
            "Hybrid": r"Output\libya_h2export_hybrid_emispr_100\3hrs\No_Hydrogen_H2_0\postnetworks\elec_s_12_ec_lcopt_Co2L-Ep-3h__2030_0.071_AB_0export.nc",
        },
        "prefix": "No_H2_export_",
    },
}

for case_key, case_info in RUN_CASES.items():
    results_dict = case_info["results"]
    prefix = case_info["prefix"]

    print("=" * 70)
    print(f"Computing Annualized Total System Costs across Scenarios [{case_key}]")
    print("=" * 70)

    scenario_costs = {}

    for scenario_name, path in results_dict.items():
        if not os.path.exists(path):
            print(f"\n[SKIP] {scenario_name}: File not found at '{path}'")
            continue

        print(f"Loading network for {scenario_name}...")
        n_export = pypsa.Network(path)

        capex = capex_by_category(n_export)
        opex = operating_cost(n_export)

        # Compute vented CO2 directly from the network loaded in THIS
        # iteration (correct for both H2_Export and No_H2_Export cases),
        # instead of looking it up by scenario_name from co2_summary_results,
        # which is only ever populated by Section 7's export-side loop.
        co2_vented_mt = compute_vented_co2_mt(n_export)

        # Calculate CO2 Tax based on non-zero CO2_TAX_EUR_PER_TON and actual vented emissions
        co2_tax = co2_vented_mt * 1e6 * CO2_TAX_EUR_PER_TON

        combined = capex.copy()
        combined["Operating Cost"] = opex
        combined["CO2 Tax"] = co2_tax

        scenario_costs[scenario_name] = combined
        print(f"\n{scenario_name} Breakdown:")
        print((combined / 1e9).round(3).to_string())
        print(f"  TOTAL System Cost: {combined.sum() / 1e9:,.3f} B EUR/yr")

    if not scenario_costs:
        print(f"\nNo scenario output files were found for {case_key}. Please check network run paths.")
    else:
        # ------------------------------------------------------------------
        # 1. Total System Cost Stacked Bar Chart Comparison (Figure 9a)
        # ------------------------------------------------------------------
        df = pd.DataFrame(scenario_costs).T / 1e9
        df = df.fillna(0.0)

        category_order = [c for c in
                          ["Solar PV", "Onshore Wind", "Offshore Wind", "CCGT", "OCGT", "Oil",
                           "Electrolysis", "H2 Storage", "H2 Fuel Cell", "SMR", "SMR with CC", 
                           "Grid Expansion", "CO2 sequestration", "H2 pipeline", "Operating Cost", "CO2 Tax"]
                          if c in df.columns]
        
        df = df[category_order]

        # MODIFIED: Reduced figure width from (10, 6) to (5.5, 6) and narrowed bars to width=0.45
        fig, ax = plt.subplots(figsize=(5.5, 6))
        bottom = pd.Series(0.0, index=df.index)

        for cat in category_order:
            values = df[cat]
            ax.bar(df.index, values, bottom=bottom, width=0.45,
                   color=CHART_COLORS.get(cat, "#95a5a6"), label=cat,
                   edgecolor="black", linewidth=0.5)
            bottom += values

        ax.set_ylabel("Annualized System Cost [Billion EUR]", fontsize=11)
        
        title_type = "No Export Hydrogen" if case_key == "No_H2_Export" else "Export Hydrogen"
        ax.set_title(f"PyPSA-Libya: {title_type} System Cost Comparison\nHybrid Scenario", fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Make x-axis labels cleaner for paper presentation
        ax.set_xlim(-0.6, len(df.index) - 0.4)

        fig.tight_layout()

        png_filename = f"figure9a_{prefix}cost_by_scenario.png" if prefix else "figure9a_cost_by_scenario.png"
        out_path = os.path.join(OUT_DIR, png_filename)
        
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        pdf_name_key = f"{prefix}cost_by_scenario" if prefix else "cost_by_scenario"
        save_paper_pdf(fig, ax, pdf_name_key)
        plt.close(fig)
        print(f"\nSaved cost comparison chart -> {out_path}")
        
        # ------------------------------------------------------------------
        # 1.1 Save Itemized Figure 9a Cost Breakdown Summary Text File
        # ------------------------------------------------------------------
        txt_filename = f"Breakdown_{prefix}cost_summary.txt" if prefix else "Breakdown_cost_summary.txt"
        txt_out_path = os.path.join(OUT_DIR, txt_filename)
        
        df_totals = df.sum(axis=1)
        
        with open(txt_out_path, "w", encoding="utf-8") as f:
            f.write("=" * 115 + "\n")
            f.write(f"PyPSA-Libya: Cost Breakdown Summary by Scenario ({title_type}) (Billion EUR/year)\n")
            f.write("=" * 115 + "\n\n")
            f.write("--- Itemized Cost Breakdown (Billion EUR/year) ---\n")
            f.write(df.round(4).to_string())
            f.write("\n\n" + "-" * 115 + "\n")
            f.write("--- Total System Costs Summary ---\n")
            for scenario_k, total_val in df_totals.items():
                f.write(f"  {scenario_k:12s} Total Cost: {total_val:8.3f} Billion EUR/year\n")
            f.write("=" * 115 + "\n")

        print(f"Saved Figure 9a cost breakdown text file -> {txt_out_path}\n")
    
    
    
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
        out_path = os.path.join(OUT_DIR, f"figure9b_investment_map_{safe_name}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        save_paper_pdf(fig, ax, f"investment_map_{safe_name}")
        plt.close(fig)
        print(f"Saved financial map -> {out_path}")


# ------------------------------------------------------------------
# 2.1 Spatial Investment (Financial CAPEX) Maps for All Scenarios (A, B, C)
# ------------------------------------------------------------------
print("\nGenerating Spatial Financial Investment (CAPEX) Maps...")

#### Adjustable Parameters for Bubble Scaling and Gap Distances
CAPEX_BUBBLE_BASE_SIZE = 150       # Minimum bubble area (points^2) for smallest investments
CAPEX_BUBBLE_MAX_ADD_SIZE = 1500  # Dynamic max additional size scaling factor

TOP_GAP_PTS = 2.0             # Clearance above primary outer circle top edge
BOTTOM_GAP_PTS = 3.0          # Clearance below primary outer circle bottom edge

for scenario_name, path in RESULTS.items():
    if not os.path.exists(path):
        print(f"[SKIP Map] {scenario_name}: File not found at '{path}'")
        continue

    n = pypsa.Network(path)
    bus_cost = per_bus_capex(n)

    if bus_cost.empty:
        print(f"[SKIP Map] {scenario_name}: No mapped capex components found.")
        continue

    pivot_cost = bus_cost.pivot_table(index="bus", columns="category",
                                       values="cost", aggfunc="sum", fill_value=0.0)

    pivot_cost.index = [strip_h2_suffix(b) for b in pivot_cost.index]
    pivot_cost = pivot_cost.groupby(level=0).sum()

    buses = n.buses.loc[n.buses.index.intersection(pivot_cost.index)].copy()
    
    # ------------------------------------------------------------------
    # Spatial Adjustment: Push offshore/coastal node (lon ~20, lat ~31) south onto land
    # ------------------------------------------------------------------
    mask_sea_node = (buses["x"] >= 19.4) & (buses["x"] <= 20.3) & (buses["y"] >= 30.0) & (buses["y"] <= 31.5)
    buses.loc[mask_sea_node, "y"] = 30.15  # Shift latitude southwards onto land
    buses.loc[mask_sea_node, "x"] = 19.80  # Optional slight shift inland for visual balance
    
    # Merge spatial coordinates to handle co-located sub-buses at identical (x,y)
    spatial_cost = pivot_cost.copy()
    spatial_cost["x"] = buses["x"]
    spatial_cost["y"] = buses["y"]
    
    # Group strictly by spatial coordinates (x, y) so overlapping nodes aggregate together
    spatial_grouped = spatial_cost.groupby(["x", "y"]).sum()

    fig = plt.figure(figsize=(10, 8))
    ax = setup_map_ax(fig)
    
    # Calculate total financial investment per location (in EUR)
    total_cost_by_loc = spatial_grouped.sum(axis=1)
    max_cost = total_cost_by_loc.max() if total_cost_by_loc.max() > 0 else 1.0

    # Plot transmission topology lines
    for _, row in n.lines.iterrows():
        b0 = strip_h2_suffix(row.bus0)
        b1 = strip_h2_suffix(row.bus1)
        if b0 in buses.index and b1 in buses.index:
            x0, y0 = buses.loc[b0, ["x", "y"]]
            x1, y1 = buses.loc[b1, ["x", "y"]]
            ax.plot([x0, x1], [y0, y1], c="gray", lw=0.5, alpha=0.5, zorder=1)

    for (x, y), row in spatial_grouped.iterrows():
        total_cost = total_cost_by_loc.get((x, y), 0.0)
        
        # Filter out zero / minimal investment buses (< 0.5 Million EUR)
        if total_cost < 0.5 * 1e6:
            ax.scatter(x, y, s=20, c="lightgray", edgecolors="gray", linewidths=0.5, zorder=2)
            continue

        # Select strictly at most 2 carriers with highest investment at this node location
        loc_series = row.sort_values(ascending=False)
        top_carriers = loc_series[loc_series > 0.0].head(2)

        if top_carriers.empty:
            continue

        # 1. Primary Highest Investment Carrier (Outer Larger Circle)
        primary_cat = top_carriers.index[0]
        primary_cost = top_carriers.iloc[0]
        size_primary = CAPEX_BUBBLE_BASE_SIZE + CAPEX_BUBBLE_MAX_ADD_SIZE * (primary_cost / max_cost)
        
        ax.scatter(
            x, y, s=size_primary,
            c=CHART_COLORS.get(primary_cat, "gray"),
            edgecolors="black", linewidths=0.7, alpha=0.9, zorder=3
        )

        primary_val_text = f"{primary_cost/1e9:.2f}B" if primary_cost >= 1e9 else f"{primary_cost/1e6:.0f}M"

        # Calculate radius of primary circle in points for vertical label offsets
        radius_pts_primary = (size_primary ** 0.5) / 2.0
        top_offset_pts = radius_pts_primary + TOP_GAP_PTS
        bottom_offset_pts = radius_pts_primary + BOTTOM_GAP_PTS

        if len(top_carriers) > 1:
            # 2. Secondary Investment Carrier (Inner Smaller Circle)
            sec_cat = top_carriers.index[1]
            sec_cost = top_carriers.iloc[1]
            
            raw_size_sec = (CAPEX_BUBBLE_BASE_SIZE * 0.6) + CAPEX_BUBBLE_MAX_ADD_SIZE * (sec_cost / max_cost)
            size_sec = min(raw_size_sec, size_primary * 0.55)

            ax.scatter(
                x, y, s=size_sec,
                c=CHART_COLORS.get(sec_cat, "gray"),
                edgecolors="black", linewidths=0.7, alpha=1.0, zorder=4
            )

            sec_val_text = f"{sec_cost/1e9:.2f}B" if sec_cost >= 1e9 else f"{sec_cost/1e6:.0f}M"

            # Upper Annotation: Primary / Larger Carrier cost directly ABOVE primary outer bubble
            ax.annotate(
                primary_val_text, (x, y),
                xytext=(0, top_offset_pts), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7.5, fontweight="normal",
                zorder=5
            )

            # Lower Annotation: Secondary / Smaller Carrier cost directly BELOW primary outer bubble
            ax.annotate(
                sec_val_text, (x, y),
                xytext=(0, -bottom_offset_pts), textcoords="offset points",
                ha="center", va="top",
                fontsize=7.5, fontweight="normal",
                bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.7),
                arrowprops=dict(
                    arrowstyle="-", color="black", lw=0.6,
                    shrinkA=0, shrinkB=0,
                    connectionstyle="arc3,rad=0"
                ),
                zorder=5
            )
        else:
            # Single carrier: Centered directly ABOVE outer bubble
            ax.annotate(
                primary_val_text, (x, y),
                xytext=(0, top_offset_pts), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7.5, fontweight="normal",
                zorder=5
            )

    legend_handles = [
        mpatches.Patch(color=c, label=k) 
        for k, c in CHART_COLORS.items() 
        if k in pivot_cost.columns
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8)
    ax.set_title(
        f"PyPSA-Libya — {scenario_name}: Spatial Financial Investment Allocation\n"
        "Nested Marker Size = Total Capex at Node [EUR]; Color = Carrier Technology",
        fontsize=11
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle=":", alpha=0.6)

    safe_name = scenario_name.lower().replace(" ", "_")
    out_path = os.path.join(OUT_DIR, f"figure9b_investment_map_{safe_name}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    save_paper_pdf(fig, ax, f"investment_map_{safe_name}")
    plt.close(fig)
    print(f"Saved financial investment map -> {out_path}")
        
        
        
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

#### Adjustable Parameters for Bubble Scaling and Gap Distances

BUBBLE_BASE_SIZE = 150       # Minimum bubble area (points^2) for smallest capacities
BUBBLE_MAX_ADD_SIZE = 1500  # Dynamic max additional size scaling factor

TOP_GAP_PTS = 2.0             # Clearance above primary outer circle top edge
BOTTOM_GAP_PTS = 3.0          # Clearance below primary outer circle bottom edge

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

    buses = n.buses.loc[n.buses.index.intersection(pivot_cap.index)].copy()
    
    # ------------------------------------------------------------------
    # Spatial Adjustment: Push offshore/coastal node (lon ~20, lat ~31) south onto land
    # ------------------------------------------------------------------
    mask_sea_node = (buses["x"] >= 19.4) & (buses["x"] <= 20.3) & (buses["y"] >= 30.0) & (buses["y"] <= 31.5)
    buses.loc[mask_sea_node, "y"] = 30.15  # Shift latitude southwards onto land
    buses.loc[mask_sea_node, "x"] = 19.80  # Optional slight shift inland for visual balance
    
    # Merge spatial coordinates to handle co-located sub-buses at identical (x,y)
    spatial_cap = pivot_cap.copy()
    spatial_cap["x"] = buses["x"]
    spatial_cap["y"] = buses["y"]
    
    # Group strictly by spatial coordinates (x, y) so overlapping nodes aggregate together
    spatial_grouped = spatial_cap.groupby(["x", "y"]).sum()

    fig = plt.figure(figsize=(10, 8))
    ax = setup_map_ax(fig)
    
    # Calculate total power capacity in MW (excluding MWh storage and Oil)
    gen_cols = [c for c in spatial_grouped.columns if c not in ["Grid Expansion", "H2 Storage", "Oil"]]
    total_mw_by_loc = spatial_grouped[gen_cols].sum(axis=1) if gen_cols else spatial_grouped.sum(axis=1)
    max_mw = total_mw_by_loc.max() if total_mw_by_loc.max() > 0 else 1.0

    # Plot transmission topology lines
    for _, row in n.lines.iterrows():
        b0 = strip_h2_suffix(row.bus0)
        b1 = strip_h2_suffix(row.bus1)
        if b0 in buses.index and b1 in buses.index:
            x0, y0 = buses.loc[b0, ["x", "y"]]
            x1, y1 = buses.loc[b1, ["x", "y"]]
            ax.plot([x0, x1], [y0, y1], c="gray", lw=0.5, alpha=0.5, zorder=1)

    for (x, y), row in spatial_grouped.iterrows():
        total_mw = total_mw_by_loc.get((x, y), 0.0)
        
        # Filter out zero / minimal capacity buses (< 50 MW)
        if total_mw < 50.0:
            ax.scatter(x, y, s=20, c="lightgray", edgecolors="gray", linewidths=0.5, zorder=2)
            continue

        # Select strictly at most 2 carriers with highest capacity at this node location
        loc_series = row[gen_cols].sort_values(ascending=False)
        top_carriers = loc_series[loc_series > 0.0].head(2)

        if top_carriers.empty:
            continue

        # 1. Primary Highest Capacity Carrier (Outer Larger Circle)
        primary_cat = top_carriers.index[0]
        primary_cap = top_carriers.iloc[0]
        size_primary = BUBBLE_BASE_SIZE + BUBBLE_MAX_ADD_SIZE * (primary_cap / max_mw)
        
        p_circle = ax.scatter(
            x, y, s=size_primary,
            c=CHART_COLORS.get(primary_cat, "gray"),
            edgecolors="black", linewidths=0.7, alpha=0.9, zorder=3
        )

        primary_val_text = f"{primary_cap/1e3:.1f}GW" if primary_cap >= 1000.0 else f"{primary_cap:.0f}MW"

        # Calculate radius of primary circle in points (s = area = pi * r^2, diameter approx sqrt(s))
        radius_pts_primary = (size_primary ** 0.5) / 2.0
        top_offset_pts = radius_pts_primary + TOP_GAP_PTS
        bottom_offset_pts = radius_pts_primary + BOTTOM_GAP_PTS

        if len(top_carriers) > 1:
            # 2. Secondary Capacity Carrier (Inner Smaller Circle)
            sec_cat = top_carriers.index[1]
            sec_cap = top_carriers.iloc[1]
            
            raw_size_sec = (BUBBLE_BASE_SIZE * 0.6) + BUBBLE_MAX_ADD_SIZE * (sec_cap / max_mw)
            size_sec = min(raw_size_sec, size_primary * 0.55)

            s_circle = ax.scatter(
                x, y, s=size_sec,
                c=CHART_COLORS.get(sec_cat, "gray"),
                edgecolors="black", linewidths=0.7, alpha=1.0, zorder=4
            )

            sec_val_text = f"{sec_cap/1e3:.1f}GW" if sec_cap >= 1000.0 else f"{sec_cap:.0f}MW"

            # Upper Annotation: Primary / Larger Carrier value directly ABOVE primary outer bubble
            ax.annotate(
                primary_val_text, (x, y),
                xytext=(0, top_offset_pts), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7.5, fontweight="normal",
                zorder=5
            )

            # Lower Annotation: Secondary / Smaller Carrier value directly BELOW primary outer bubble
            ax.annotate(
                sec_val_text, (x, y),
                xytext=(0, -bottom_offset_pts), textcoords="offset points",
                ha="center", va="top",
                fontsize=7.5, fontweight="normal",
                bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.7),
                arrowprops=dict(
                    arrowstyle="-", color="black", lw=0.6,
                    shrinkA=0, shrinkB=0,
                    connectionstyle="arc3,rad=0"
                ),
                zorder=5
            )
        else:
            # Single carrier: Centered directly ABOVE outer bubble
            ax.annotate(
                primary_val_text, (x, y),
                xytext=(0, top_offset_pts), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7.5, fontweight="normal",
                zorder=5
            )

    legend_handles = [mpatches.Patch(color=c, label=k) for k, c in CHART_COLORS.items() if k in pivot_cap.columns]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8)
    ax.set_title(
        f"PyPSA-Libya — {scenario_name}: Spatial Installed Capacity Allocation\n"
        "Nested Marker Size = Carrier Capacity [MW/GW]; Color = Carrier Technology",
        fontsize=11
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle=":", alpha=0.6)

    safe_name = scenario_name.lower().replace(" ", "_")
    out_path = os.path.join(OUT_DIR, f"figure9c_capacity_map_{safe_name}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    save_paper_pdf(fig, ax, f"capacity_map_{safe_name}")
    plt.close(fig)
    print(f"Saved physical capacity map -> {out_path}")
    
    
        
        
    # =====================================================================
    # 2.3 Spatial & Graphical Transmission Line Bottlenecks for Hybrid Scenario
    # =====================================================================
    
    import numpy as np

    print("\n" + "=" * 70)
    print("Generating Transmission Line Bottlenecks (Reports & Maps) for Hybrid Scenario")
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

    def shorten_line_id(line_idx, bus0, bus1, existing_line_count):
        idx_str = str(line_idx).strip()

        # Check if line index or name indicates a new candidate/expansion line
        if "lines new" in idx_str or "new" in idx_str.lower() or idx_str.startswith("N"):
            # Extract trailing numeric ID from string if available (e.g., 'lines new 0' -> 0)
            m = re.search(r'(\d+)$', idx_str)
            new_offset = int(m.group(1)) if m else 0
            # Start 1-indexed counting continuing directly after existing lines
            return str(existing_line_count + new_offset + 1)

        # If standard existing line index is purely numeric, shift to 1-based indexing
        if idx_str.isdigit():
            return str(int(idx_str) + 1)

        return idx_str

    # ------------------------------------------------------------------
    # Data Processing & Analysis
    # ------------------------------------------------------------------
    scenario_line_data = {}
    
    with open(os.path.join(OUT_DIR, "transmission_line_bottlenecks_summary.txt"), "w", encoding="utf-8") as f_out:
        f_out.write("=" * 135 + "\n")
        f_out.write(" PyPSA-Libya — Hybrid Scenario: Transmission Line Bottlenecks Report\n")
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

            # Calculate total existing non-expansion lines in the network
            n_existing_lines = sum(
                1 for idx in n_scen.lines.index 
                if not ("lines new" in str(idx) or "new" in str(idx).lower() or str(idx).startswith("N")))

            # Generate updated 1-indexed short line IDs
            short_ids = [
                shorten_line_id(idx, row.bus0, row.bus1, n_existing_lines) 
                for idx, row in n_scen.lines.iterrows()]

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

    print(f"Saved transmission bottlenecks summary text file -> {os.path.join(OUT_DIR, 'transmission_line_bottlenecks_summary.txt')}")

    # ------------------------------------------------------------------
    # Comparative Bar Chart: Top Bottleneck Lines
    # ------------------------------------------------------------------
    if scenario_line_data:
        # Find top 10 bottleneck lines based on mean loading
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
        ax.set_title("PyPSA-Libya — Transmission Bottlenecks (Top 10 Congested Lines)\n", fontsize=12, fontweight="bold")
        ax.set_xticklabels(plot_df.index, rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(title="Scenario", fontsize=9)
        fig.tight_layout()

        out_bar_path = os.path.join(OUT_DIR, "transmission_line_bottlenecks.png")
        fig.savefig(out_bar_path, dpi=200, bbox_inches="tight")
        save_paper_pdf(fig, ax, "transmission_line_bottlenecks")
        plt.close(fig)
        print(f"Saved transmission bottlenecks bar chart -> {out_bar_path}")

    # ------------------------------------------------------------------
    # Spatial Map: Line Bottleneck Locations
    # ------------------------------------------------------------------
    for scenario_name, (n_scen, line_df) in scenario_line_data.items():
        
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

        # -------------------------------------------------------------
        # Label ALL Transmission Lines with Overlap Prevention
        # -------------------------------------------------------------
        # Keep track of midpoints to offset overlapping parallel lines
        seen_midpoints = {}

        for idx, row in n_scen.lines.iterrows():
            if row.bus0 in n_scen.buses.index and row.bus1 in n_scen.buses.index:
                x0, y0 = n_scen.buses.loc[row.bus0, ["x", "y"]]
                x1, y1 = n_scen.buses.loc[row.bus1, ["x", "y"]]
                xs, ys = route_line_land(x0, y0, x1, y1)
                
                mid_idx = len(xs) // 2
                mx, my = xs[mid_idx], ys[mid_idx]

                # Identify parallel or overlapping segments based on midpoint rounding
                key = (round(mx, 2), round(my, 2))
                count = seen_midpoints.get(key, 0)
                seen_midpoints[key] = count + 1

                # Apply a slight upward/rightward offset for multi-lines between same buses
                offset_x = 0.08 * count
                offset_y = 0.08 * count

                short_id = line_df.loc[idx, "short_line_id"]
                ax.annotate(
                    short_id, 
                    (mx + offset_x, my + offset_y), 
                    fontsize=7, 
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85, edgecolor="black", lw=0.5), 
                    zorder=6)

        macro_buses = n_scen.buses[~n_scen.buses.index.str.contains("low voltage|battery|H2", case=False)]
        ax.scatter(macro_buses.x, macro_buses.y, s=15, c="#1e293b", edgecolors="white", linewidths=0.5, zorder=4)

        sm_line = plt.cm.ScalarMappable(cmap=cmap_lines, norm=norm_metric)
        sm_line.set_array([])
        cbar = fig.colorbar(sm_line, ax=ax, orientation="vertical", shrink=0.7, pad=0.03)
        cbar.set_label("Mean Capacity Loading (%)", fontsize=10)

        ax.set_title(f"PyPSA-Libya — {scenario_name}: Transmission Bottlenecks Map", fontsize=11, fontweight="bold")
        
        safe_scen = scenario_name.lower().replace(" ", "_")
        out_map_path = os.path.join(OUT_DIR, f"transmission_line_bottlenecks_map_{safe_scen}.png")
        fig.savefig(out_map_path, dpi=200, bbox_inches="tight")
        save_paper_pdf(fig, ax, f"transmission_line_bottlenecks_map_{safe_scen}")
        plt.close(fig)
        print(f"Saved transmission map for {scenario_name} -> {out_map_path}")



# =====================================================================
# 3. LCOE & LCOH (LEVELIZED COST ANALYSIS) COMPUTATION & EXPORT
# =====================================================================
print("\n" + "=" * 70)
print("Computing System LCOE and Net-Cost Allocated LCOH across Scenarios")
print("=" * 70)

# ------------------------------------------------------------------
# Define Baseline Network Paths Explicitly (No-H2 Export Reference)
# ------------------------------------------------------------------

BASELINE_RESULTS = {}
for scenario_name, path in RESULTS.items():
    if r"\No_Hydrogen_H2_0" not in path:
        path_0export = path.replace(r"\postnetworks", r"\No_Hydrogen_H2_0\postnetworks")
        path_0export = re.sub(r'(\d+)export\.nc$', r'0export.nc', path_0export)
    else:
        path_0export = path
    BASELINE_RESULTS[scenario_name] = path_0export


DEFAULT_EXPORT_H2_MWH = 70 * 1e6      # 70 TWh export target in MWh
DEFAULT_DOMESTIC_AC_MWH = 52.04 * 1e6 # 52.04 TWh AC domestic demand in MWh

# Hydrogen Heating Values:
H2_LHV_MWH_PER_KG = 0.03333           # Lower Heating Value (33.33 kWh/kg)
H2_HHV_MWH_PER_KG = 0.03941           # Higher Heating Value (39.41 kWh/kg)

lcoe_results = {}
lcoh_results_main = {}
lcoh_results_summary = {}

for scenario_name, export_path in RESULTS.items():
    if not os.path.exists(export_path):
        print(f"[SKIP LCOE/LCOH] Export file not found for {scenario_name}: {export_path}")
        continue

    # ------------------------------------------------------------------
    # A. Combined System-Wide Levelized Cost of Energy (LCOE) Calculation
    # ------------------------------------------------------------------
    print(f"Processing Export Network LCOE/LCOH for {scenario_name}...")
    n_export = pypsa.Network(export_path)

    capex_exp = capex_by_category(n_export).sum()
    opex_exp = operating_cost(n_export)
    total_system_cost_eur = capex_exp + opex_exp
    
    # capex_exp = capex_by_category(n_export).sum()
    # opex_exp = operating_cost(n_export)
    # co2_tax_exp_eur = co2_tax_cost_eur(n_export, fallback_mt=CO2_VENTED_FALLBACK_EXPORT.get(scenario_name, 0.0))
    # total_system_cost_eur = capex_exp + opex_exp + co2_tax_exp_eur

    domestic_ac_mwh = 0.0
    ac_loads = n_export.loads[n_export.loads.carrier == "AC"]
    if not ac_loads.empty and not n_export.loads_t.p.empty:
        weights = n_export.snapshot_weightings.generators
        domestic_ac_mwh = n_export.loads_t.p[ac_loads.index].mul(weights, axis=0).sum().sum()
    if domestic_ac_mwh <= 0:
        domestic_ac_mwh = DEFAULT_DOMESTIC_AC_MWH

    exported_h2_mwh = 0.0
    h2_export_loads = n_export.loads[n_export.loads.carrier.str.contains("H2|h2|export", case=False, na=False)]
    if not h2_export_loads.empty and not n_export.loads_t.p.empty:
        weights = n_export.snapshot_weightings.generators
        exported_h2_mwh = n_export.loads_t.p[h2_export_loads.index].mul(weights, axis=0).sum().sum()
    if exported_h2_mwh <= 0:
        exported_h2_mwh = DEFAULT_EXPORT_H2_MWH

    total_system_energy_mwh = domestic_ac_mwh + exported_h2_mwh
    lcoe_system_eur_per_mwh = total_system_cost_eur / total_system_energy_mwh
    
    lcoe_results[scenario_name] = {
        "Total System Cost [Billion EUR]": total_system_cost_eur / 1e9,
        "Domestic AC Demand [TWh]": domestic_ac_mwh / 1e6,
        "Exported H2 Energy [TWh]": exported_h2_mwh / 1e6,
        "Total System Energy [TWh]": total_system_energy_mwh / 1e6,
        "System LCOE [EUR/MWh]": lcoe_system_eur_per_mwh,
    }

    # ------------------------------------------------------------------
    # B. Net-Cost Allocated LCOH Calculation (Baseline Deduction)
    # ------------------------------------------------------------------
    base_path = BASELINE_RESULTS.get(scenario_name)
    if base_path and os.path.exists(base_path):
        print(f"Loading Baseline (No-H2 Export) network for {scenario_name}...")
        n_base = pypsa.Network(base_path)
        
        capex_base = capex_by_category(n_base).sum()
        opex_base = operating_cost(n_base)
        # NOTE: CO2 tax intentionally excluded here, matching the export side
        # above (total_system_cost_eur also excludes it). Section 3 reports a
        # tax-free "cost basis" LCOH; including tax on only ONE side of this
        # subtraction was a bug that artificially deflated Net H2 Cost.
        #### co2_tax_base_eur = co2_tax_cost_eur(n_base, fallback_mt=0.0)   # no reference fallback for baseline
        #### baseline_cost_eur = capex_base + opex_base + co2_tax_base_eur
        
        baseline_cost_eur = capex_base + opex_base
        
        net_h2_cost_eur = total_system_cost_eur - baseline_cost_eur

        lcoh_eur_per_mwh = net_h2_cost_eur / exported_h2_mwh
        lcoh_eur_per_kg_lhv = lcoh_eur_per_mwh * H2_LHV_MWH_PER_KG
        lcoh_eur_per_kg_hhv = lcoh_eur_per_mwh * H2_HHV_MWH_PER_KG

        lcoh_results_main[scenario_name] = {
            "Export Cost [B EUR]": total_system_cost_eur / 1e9,
            "Baseline Cost [B EUR]": baseline_cost_eur / 1e9,
            "Net H2 Cost [B EUR]": net_h2_cost_eur / 1e9,
            "Exported H2 [TWh]": exported_h2_mwh / 1e6,
            "Net LCOH [EUR/MWh]": lcoh_eur_per_mwh,
        }

        lcoh_results_summary[scenario_name] = {
            "Net LCOH [EUR/MWh]": lcoh_eur_per_mwh,
            "Net LCOH LHV [EUR/kg]": lcoh_eur_per_kg_lhv,
            "Net LCOH HHV [EUR/kg]": lcoh_eur_per_kg_hhv,
        }
    else:
        print(f"[WARNING] Baseline file missing for {scenario_name} at: {base_path}")

# ------------------------------------------------------------------
# Save LCOE & Pure LCOH Summaries (Restored to Original Format)
# ------------------------------------------------------------------
if lcoe_results:
    lcoe_df = pd.DataFrame(lcoe_results).T
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

    lcoh_txt_path = os.path.join(OUT_DIR, "LCOH_summary_results.txt")
    with open(lcoh_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 95 + "\n")
        f.write("PyPSA-Libya: Net Levelized Cost of Hydrogen (LCOH) Export Results\n")
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
    print(f"\nSaved standalone Net LCOH summary report -> {lcoh_txt_path}")



# =====================================================================
# 4. AUGUST MONTHLY DISPATCH TIME-SERIES CHARTS (6-HOUR / 3-HOUR RESOLUTION)
# =====================================================================

print("\n" + "=" * 70)
print("Generating August Monthly Dispatch Profiles for Scenarios (Matching Native Temporal Resolution)")
print("=" * 70)

# Comprehensive carrier mapping including common variations, casing, and new categories
CAPEX_CARRIER_MAP.update({
    # Hydrogen & Fuel Cell variations
    "H2 Fuel Cell": "H2 Fuel Cell",
    "h2 fuel cell": "H2 Fuel Cell",
    "H2 fuel cell": "H2 Fuel Cell",
    "H2 turbine": "H2 Fuel Cell",
    "h2 turbine": "H2 Fuel Cell",
    "fuel cell": "H2 Fuel Cell",
    "fc": "H2 Fuel Cell",
    
    # Gas variations (Capitalized as "Gas")
    "gas": "Gas",
    "Gas": "Gas",
    "CCGT": "CCGT",
    "OCGT": "OCGT",

    # Solar Rooftop variations (Capitalized as "Solar Rooftop")
    "solar rooftop": "Solar Rooftop",
    "Solar Rooftop": "Solar Rooftop",
    "solar_rooftop": "Solar Rooftop",
    "solar": "Solar PV",
    "Solar PV": "Solar PV",
})

CHART_COLORS.update({
    "H2 Fuel Cell": "cyan",
    "Gas": "#7f8c8d",               # Professional muted gray for Gas
    "Solar Rooftop": "#e6007e",     # Distinct magenta/pink accent for Solar Rooftop
    "Solar PV": "#f5a623",          # Standard Solar PV color for consistency
})


def extract_total_dispatch(n: pypsa.Network) -> pd.DataFrame:
    """Extracts total dispatch per carrier for all generators, storage units, and fuel cell links."""
    parts = []
    
    # Diagnostic print to check available carriers in this network file
    if hasattr(n, "generators") and not n.generators.empty:
        print(f"  [Diagnostics] Generator carriers found: {n.generators.carrier.unique().tolist()}")
    if hasattr(n, "links") and not n.links.empty:
        print(f"  [Diagnostics] Link carriers found: {n.links.carrier.unique().tolist()}")

    # 1. Generator dispatch
    if hasattr(n, "generators") and not n.generators.empty and hasattr(n, "generators_t") and not n.generators_t.p.empty:
        gen_dispatch = n.generators_t.p.T.groupby(
            n.generators.carrier.map(lambda c: CAPEX_CARRIER_MAP.get(str(c).strip(), CAPEX_CARRIER_MAP.get(str(c).strip().lower(), str(c).strip())))
        ).sum().T / 1e3  # Convert MW to GW
        parts.append(gen_dispatch)
    
    # 2. Storage units discharging dispatch
    if hasattr(n, "storage_units") and not n.storage_units.empty and hasattr(n, "storage_units_t") and not n.storage_units_t.p.empty:
        su_dispatch = n.storage_units_t.p.clip(lower=0).T.groupby(
            n.storage_units.carrier.map(lambda c: CAPEX_CARRIER_MAP.get(str(c).strip(), CAPEX_CARRIER_MAP.get(str(c).strip().lower(), str(c).strip())))
        ).sum().T / 1e3
        parts.append(su_dispatch)

    # 3. Links dispatch (Fuel cells / H2 turbines injecting power via p1)
    if hasattr(n, "links") and not n.links.empty and hasattr(n, "links_t") and hasattr(n.links_t, "p1") and not n.links_t.p1.empty:
        def map_link_carrier(c):
            c_str = str(c).strip()
            c_lower = c_str.lower()
            if "fuel cell" in c_lower or "turbine" in c_lower or c_lower in ["fc", "h2 fuel cell", "h2 turbine"]:
                return "H2 Fuel Cell"
            return CAPEX_CARRIER_MAP.get(c_str, CAPEX_CARRIER_MAP.get(c_lower, c_str))

        link_mapped_carriers = n.links.carrier.map(map_link_carrier)
        fuel_cell_links = [l for l in n.links.index if link_mapped_carriers[l] == "H2 Fuel Cell" and l in n.links_t.p1.columns]
        
        if fuel_cell_links:
            # FIX: Negate p1 because PyPSA stores link outputs as negative values.
            fc_dispatch = (-n.links_t.p1[fuel_cell_links]).clip(lower=0).T.groupby(
                link_mapped_carriers[fuel_cell_links]
            ).sum().T / 1e3
            parts.append(fc_dispatch)

    # Combine all dispatch sources
    if parts:
        total_dispatch = pd.concat(parts, axis=1).T.groupby(level=0).sum().T
    else:
        total_dispatch = pd.DataFrame(index=n.snapshots)

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

    # Format X-ticks to display every 8th step
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
    save_paper_pdf(fig, ax, f"monthly_dispatch_august_bar_{safe_name}")
    plt.close(fig)
    print(f"  Saved August bar chart -> {out_path}")
    

    
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

# User-adjustable threshold in GWh defining major vs small bars
THRESHOLD_GWH = 3000.0

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
    "H2 Fuel Cell": "cyan",
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
    "Grid Expansion": "#f9b7cc",      
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
combined_ren_txt_path = os.path.join(CARRIERS_DIR, "Renewable_cf_and_curtailment_summary.txt")

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
    
    EXCLUDE_FROM_GENERATION_CHART = {"co2 vent", "process emissions", "process emissions CC"}
    phys_gen_dict = {}

    if not n_scen.generators.empty and not n_scen.generators_t.p.empty:
        gen_p = n_scen.generators_t.p.groupby(n_scen.generators.carrier.map(map_outcome_carrier), axis=1).sum()
        for carrier in gen_p.columns:
            energy_gwh = (gen_p[carrier] * dt_hours).sum() / 1e3
            if energy_gwh > 1e-3:
                phys_gen_dict[carrier] = energy_gwh

    # if not n_scen.links.empty and not n_scen.links_t.p0.empty:
    #     link_p = n_scen.links_t.p0.groupby(n_scen.links.carrier.map(map_outcome_carrier), axis=1).sum()
    #     for carrier in link_p.columns:
    #         energy_gwh = (link_p[carrier] * dt_hours).sum() / 1e3
    #         if energy_gwh > 1e-3:
    #             phys_gen_dict[carrier] = energy_gwh
    
    if not n_scen.links.empty and not n_scen.links_t.p0.empty:
        # Use OUTPUT (p0 * efficiency), not raw fuel/feedstock INPUT (p0) —
        # p0 is what the link consumes, not what it actually produces.
        eff = n_scen.links["efficiency"].reindex(n_scen.links_t.p0.columns).fillna(1.0)
        link_output = n_scen.links_t.p0.mul(eff, axis=1)
        link_p = link_output.groupby(n_scen.links.carrier.map(map_outcome_carrier), axis=1).sum()
        for carrier in link_p.columns:
            if carrier in EXCLUDE_FROM_GENERATION_CHART:   # <-- NEW: skip CO2 mass-flow carriers
                continue
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
    
    phys_gen_gwh = phys_gen_gwh.round().astype(int)

    # Classification of major vs minor carriers based on THRESHOLD_GWH
    major_carriers = phys_gen_gwh[phys_gen_gwh >= THRESHOLD_GWH].index.tolist()

    def check_is_major(c_name):
        base_name = str(c_name).replace(" (discharge)", "").replace(" (charge)", "")
        return (c_name in major_carriers) or (base_name in major_carriers)

    # -----------------------------------------------------------------
    # B. Solved Capacity Charts (Full & Filtered in GW)
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
        # Dictionary to shorten/rename specific carrier y-axis labels (matching Part C)
        LABEL_MAP = {
            "electricity distribution grid": "Grid Delivery",
            "Grid Expansion": "Grid Delivery",
            "solar rooftop": "Solar Rooftop",
            "gas": "Gas",
            "gas for industry": "Industry Gas",
            "gas for industry CC": "Industry Gas CC",
            "process emissions": "Process Emissions",
            "process emissions CC": "Process Emissions CC",
            "home battery charger": "Battery Charge",
            "home battery discharger": "Battery Discharge",
            "co2 vent": "CO2 Vented",
        }

        # Aggregate capacities in MW, filter tiny entries, then convert to GW
        all_power_cap_mw = df_cap.groupby("carrier")["capacity"].sum().sort_values()
        all_power_cap_gw = (all_power_cap_mw[all_power_cap_mw > 1e-2] / 1000.0)

        # 1. Full Solved Capacity Chart (in GW)
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (carrier, val_gw) in enumerate(all_power_cap_gw.items()):
            is_maj = check_is_major(carrier)
            color, alpha, edge_c, lw = get_smart_carrier_style(carrier, is_major=is_maj)
            ax.barh(carrier, val_gw, color=color, alpha=alpha, edgecolor=edge_c, linewidth=lw)

        ax.set_xlabel("Optimal Installed Power Capacity (GW)")
        ax.set_title(f"PyPSA-Libya — Solved Power Capacity by Carrier ({scenario_name})", fontsize=12, fontweight="bold")
        
        # Map y-axis tick labels
        full_yticklabels = [LABEL_MAP.get(c, c) for c in all_power_cap_gw.index]
        ax.set_yticks(range(len(all_power_cap_gw)))
        ax.set_yticklabels(full_yticklabels)

        ax.margins(x=0.20)
        ax.set_axisbelow(True)
        ax.grid(axis='x', linestyle='--', alpha=0.5, color='#cbd5e1')
        
        # Add GW annotations
        for p in ax.patches:
            width = p.get_width()
            if width > 0.001:
                ax.annotate(f'{width:,.2f} GW', (width, p.get_y() + p.get_height() / 2.),
                            xytext=(5, 0), textcoords='offset points', ha='left', va='center', fontsize=9)

        fig.tight_layout()
        out_cap_png = os.path.join(CARRIERS_DIR, f"{safe_prefix}_solved_capacity_by_carrier.png")
        fig.savefig(out_cap_png, dpi=150, bbox_inches="tight")
        save_paper_pdf(fig, ax, f"{safe_prefix}_solved_capacity_by_carrier")
        plt.close(fig)

        # 2. Filtered Solved Capacity Chart (Major carriers only, in GW)
        filtered_cap_gw = all_power_cap_gw[all_power_cap_gw.index.map(check_is_major)]
        if not filtered_cap_gw.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            for carrier, val_gw in filtered_cap_gw.items():
                color, alpha, edge_c, lw = get_smart_carrier_style(carrier, is_major=True)
                ax.barh(carrier, val_gw, color=color, alpha=alpha, edgecolor=edge_c, linewidth=lw)

            ax.set_xlabel("Optimal Installed Power Capacity (GW)")
            ax.set_title(f"PyPSA-Libya — Solved Capacity by Carrier [>= {THRESHOLD_GWH:,.0f} GWh Gen] ({scenario_name})", fontsize=12, fontweight="bold")
            
            # Map y-axis tick labels
            filt_yticklabels = [LABEL_MAP.get(c, c) for c in filtered_cap_gw.index]
            ax.set_yticks(range(len(filtered_cap_gw)))
            ax.set_yticklabels(filt_yticklabels)

            ax.margins(x=0.20)
            ax.set_axisbelow(True)
            ax.grid(axis='x', linestyle='--', alpha=0.5, color='#cbd5e1')

            # Add GW annotations
            for p in ax.patches:
                width = p.get_width()
                if width > 0.001:
                    ax.annotate(f'{width:,.2f} GW', (width, p.get_y() + p.get_height() / 2.),
                                xytext=(5, 0), textcoords='offset points', ha='left', va='center', fontsize=9)

            fig.tight_layout()
            out_cap_filt_png = os.path.join(CARRIERS_DIR, f"{safe_prefix}_solved_capacity_by_carrier_filtered.png")
            fig.savefig(out_cap_filt_png, dpi=150, bbox_inches="tight")
            save_paper_pdf(fig, ax, f"{safe_prefix}_solved_capacity_by_carrier_filtered")
            plt.close(fig)
            

    # -----------------------------------------------------------------
    # C. Physical Generation Charts (Full & Filtered in TWh)
    # -----------------------------------------------------------------
    if not phys_gen_gwh.empty:
        # Convert GWh to TWh
        phys_gen_twh = (phys_gen_gwh / 1000.0).sort_values()

        # Dictionary to shorten/rename specific carrier x-axis labels
        LABEL_MAP = {
            "electricity distribution grid": "Grid Delivery",
            "Grid Expansion": "Grid Delivery",
            "solar rooftop": "Solar Rooftop",
            "gas": "Gas",
            "gas for industry": "Industry Gas",
            "gas for industry CC": "Industry Gas CC",
            "process emissions": "Process Emissions",
            "process emissions CC": "Process Emissions CC",
            "home battery charger": "Battery Charge",
            "home battery discharger": "Battery Discharge",
            "co2 vent": "CO2 Vented",
        }

        # 1. Full Physical Generation Chart (in TWh)
        fig, ax = plt.subplots(figsize=(10, 6))
        for carrier, val_twh in phys_gen_twh.items():
            is_maj = check_is_major(carrier)
            color, alpha, edge_c, lw = get_smart_carrier_style(carrier, is_major=is_maj)
            ax.bar(carrier, val_twh, color=color, alpha=alpha, edgecolor=edge_c, linewidth=lw)

        ax.set_ylabel("Physical Generation / Conversion Activity (TWh)")
        ax.set_title(f"PyPSA-Libya — Physical Generation by Carrier ({scenario_name})", fontsize=12, fontweight="bold")
        
        # Apply updated mapped x-tick labels
        full_labels = [LABEL_MAP.get(c, c) for c in phys_gen_twh.index]
        ax.set_xticks(range(len(phys_gen_twh)))
        ax.set_xticklabels(full_labels, rotation=45, ha="right")
        
        ax.margins(y=0.20)
        ax.set_axisbelow(True)
        ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cbd5e1')

        # Annotate values above bars in TWh
        for p in ax.patches:
            height = p.get_height()
            if height > 0.001:
                ax.annotate(f'{height:,.2f}', (p.get_x() + p.get_width() / 2., height),
                            xytext=(0, 5), textcoords='offset points', ha='center', va='bottom', fontsize=9)

        fig.tight_layout()
        out_gen_png = os.path.join(CARRIERS_DIR, f"{safe_prefix}_physical_generation_by_carrier.png")
        fig.savefig(out_gen_png, dpi=150, bbox_inches="tight")
        save_paper_pdf(fig, ax, f"{safe_prefix}_physical_generation_by_carrier")
        plt.close(fig)

        # 2. Filtered Physical Generation Chart (Major carriers only, in TWh)
        filtered_gen_twh = phys_gen_twh[phys_gen_twh.index.map(check_is_major)]
        if not filtered_gen_twh.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            for carrier, val_twh in filtered_gen_twh.items():
                color, alpha, edge_c, lw = get_smart_carrier_style(carrier, is_major=True)
                ax.bar(carrier, val_twh, color=color, alpha=alpha, edgecolor=edge_c, linewidth=lw)

            ax.set_ylabel("Physical Generation / Conversion Activity (TWh)")
            ax.set_title(f"PyPSA-Libya — Physical Generation by Carrier [>= {THRESHOLD_GWH:,.0f} GWh] ({scenario_name})", fontsize=12, fontweight="bold")
            
            # Apply updated mapped x-tick labels
            filt_labels = [LABEL_MAP.get(c, c) for c in filtered_gen_twh.index]
            ax.set_xticks(range(len(filtered_gen_twh)))
            ax.set_xticklabels(filt_labels, rotation=45, ha="right")
            
            ax.margins(y=0.20)
            ax.set_axisbelow(True)
            ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cbd5e1')

            # Annotate values above bars in TWh
            for p in ax.patches:
                height = p.get_height()
                if height > 0.001:
                    ax.annotate(f'{height:,.2f}', (p.get_x() + p.get_width() / 2., height),
                                xytext=(0, 5), textcoords='offset points', ha='center', va='bottom', fontsize=9)

            fig.tight_layout()
            out_gen_filt_png = os.path.join(CARRIERS_DIR, f"{safe_prefix}_physical_generation_by_carrier_filtered.png")
            fig.savefig(out_gen_filt_png, dpi=150, bbox_inches="tight")
            save_paper_pdf(fig, ax, f"{safe_prefix}_physical_generation_by_carrier_filtered")
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
        save_paper_pdf(fig, ax, f"{safe_prefix}_renewable_cf_and_curtailment")
        plt.close(fig)

print(f"\nAll updated outputs successfully generated in: {CARRIERS_DIR}")



# =====================================================================
# 6. WATER CONSUMPTION & DESALINATION ANALYSIS FOR HYDROGEN PRODUCTION
# =====================================================================

print("\n" + "=" * 70)
print("Computing Water Demand and Desalination Overhead across Scenarios")
print("=" * 70)

# Target subfolder path alignment inside Energy Carriers Analysis
WATER_ANALYSIS_DIR = os.path.join(OUT_DIR, "Energy Carriers Analysis")
os.makedirs(WATER_ANALYSIS_DIR, exist_ok=True)

# Desalination cost parameters
DESAL_COST_EUR_PER_M3 = 1.50         # Estimated RO desalination cost (€/m3 of fresh water)
DESAL_ENERGY_KWH_PER_M3 = 3.5        # Energy overhead for RO desalination (kWh/m3)
ELECTROLYZER_SEC_KWH_PER_KG = 52.5   # Average electrolyzer electrical consumption (kWh/kg H2)

# Technology-specific water intensity factors (Liters of water per kg H2 produced)
WATER_INTENSITY_L_PER_KG = {
    "H2 Electrolysis": 10.0,
    "SMR CC": 7.5,
    "SMR": 5.0
}

water_analysis_results = {}

for scenario_name, export_path in RESULTS.items():
    if not os.path.exists(export_path):
        continue

    n_scen = pypsa.Network(export_path)
    weights = n_scen.snapshot_weightings.generators
    dt_hours = (n_scen.snapshots[1] - n_scen.snapshots[0]).total_seconds() / 3600.0 if len(n_scen.snapshots) > 1 else 1.0

    # 1. Calculate Technology-Specific Hydrogen Mass (kg) & Total Water Demand (m³)
    total_h2_mwh = 0.0
    total_h2_mass_kg = 0.0
    total_water_demand_m3 = 0.0

    # Extract H2 production from links (Electrolysis, SMR, SMR CC)
    if not n_scen.links.empty and not n_scen.links_t.p1.empty:
        for tech, intensity_l_per_kg in WATER_INTENSITY_L_PER_KG.items():
            tech_links = n_scen.links[n_scen.links.carrier == tech].index
            if not tech_links.empty:
                # Active links output power (p1 is output power for standard 2-bus links, or abs(p1))
                sel_links = [l for l in tech_links if l in n_scen.links_t.p1.columns]
                if sel_links:
                    tech_h2_mwh = n_scen.links_t.p1[sel_links].abs().mul(weights, axis=0).sum().sum()
                    tech_h2_mass_kg = tech_h2_mwh / H2_LHV_MWH_PER_KG
                    tech_water_m3 = tech_h2_mass_kg * (intensity_l_per_kg / 1000.0)

                    total_h2_mwh += tech_h2_mwh
                    total_h2_mass_kg += tech_h2_mass_kg
                    total_water_demand_m3 += tech_water_m3

    # Fallback to exported load if no links match
    if total_h2_mass_kg <= 0:
        h2_export_loads = n_scen.loads[n_scen.loads.carrier.str.contains("H2|h2|export", case=False, na=False)]
        if not h2_export_loads.empty and not n_scen.loads_t.p.empty:
            total_h2_mwh = n_scen.loads_t.p[h2_export_loads.index].mul(weights, axis=0).sum().sum()
        else:
            total_h2_mwh = DEFAULT_EXPORT_H2_MWH
        total_h2_mass_kg = total_h2_mwh / H2_LHV_MWH_PER_KG
        total_water_demand_m3 = total_h2_mass_kg * (WATER_INTENSITY_L_PER_KG["H2 Electrolysis"] / 1000.0)

    # 2. Financial and Energy Metrics
    total_system_cost_eur = scenario_costs[scenario_name].sum() if scenario_name in scenario_costs else 0.0
    
    h2_mass_tonnes = total_h2_mass_kg / 1000.0
    h2_mass_million_tonnes = total_h2_mass_kg / 1e9
    water_demand_million_m3 = total_water_demand_m3 / 1e6

    desal_cost_eur = total_water_demand_m3 * DESAL_COST_EUR_PER_M3
    desal_cost_million_eur = desal_cost_eur / 1e6
    desal_cost_share_pct = (desal_cost_eur / total_system_cost_eur * 100.0) if total_system_cost_eur > 0 else 0.0
    desal_marginal_cost_eur_per_kg = desal_cost_eur / total_h2_mass_kg if total_h2_mass_kg > 0 else 0.0

    desal_energy_kwh = total_water_demand_m3 * DESAL_ENERGY_KWH_PER_M3
    desal_energy_gwh = desal_energy_kwh / 1e6
    desal_energy_per_kg_h2 = desal_energy_kwh / total_h2_mass_kg if total_h2_mass_kg > 0 else 0.0
    desal_energy_share_pct = (desal_energy_per_kg_h2 / ELECTROLYZER_SEC_KWH_PER_KG * 100.0)

    water_analysis_results[scenario_name] = {
        "Total H2 Output [MWh]": total_h2_mwh,
        "Total H2 Mass [Tonnes]": h2_mass_tonnes,
        "H2 Mass (Million Tonnes)": h2_mass_million_tonnes,
        "Water Required [m³]": total_water_demand_m3,
        "Water Demand (Million m3)": water_demand_million_m3,
        "Est. Water Cost [M EUR/yr]": desal_cost_million_eur,
        "Desal Cost Share of System (%)": desal_cost_share_pct,
        "Desal Marginal Cost (EUR/kg H2)": desal_marginal_cost_eur_per_kg,
        "Desal Energy Overhead (GWh)": desal_energy_gwh,
        "Desal Specific Energy (kWh/kg H2)": desal_energy_per_kg_h2,
        "Desal Share of Elec Demand (%)": desal_energy_share_pct,
    }


# ------------------------------------------------------------------
# Save Water Demand Table Export (.txt)
# ------------------------------------------------------------------
if water_analysis_results:
    df_water = pd.DataFrame(water_analysis_results)

    water_table_path = os.path.join(WATER_ANALYSIS_DIR, "water_demand_and_desalination_costs_summary.txt")
    with open(water_table_path, "w", encoding="utf-8") as f:
        f.write("=" * 95 + "\n")
        f.write("PyPSA-Libya: Dynamic Multi-Technology Water Consumption & Desalination Summary\n")
        f.write("=" * 95 + "\n\n")

        formatted_df = df_water.copy()
        formatted_df.loc["Total H2 Output [MWh]"] = formatted_df.loc["Total H2 Output [MWh]"].map("{:,.2f}".format)
        formatted_df.loc["Total H2 Mass [Tonnes]"] = formatted_df.loc["Total H2 Mass [Tonnes]"].map("{:,.2f}".format)
        formatted_df.loc["H2 Mass (Million Tonnes)"] = formatted_df.loc["H2 Mass (Million Tonnes)"].map("{:,.3f}".format)
        formatted_df.loc["Water Required [m³]"] = formatted_df.loc["Water Required [m³]"].map("{:,.2f}".format)
        formatted_df.loc["Water Demand (Million m3)"] = formatted_df.loc["Water Demand (Million m3)"].map("{:,.2f}".format)
        formatted_df.loc["Est. Water Cost [M EUR/yr]"] = formatted_df.loc["Est. Water Cost [M EUR/yr]"].map("{:,.6f}".format)
        formatted_df.loc["Desal Cost Share of System (%)"] = formatted_df.loc["Desal Cost Share of System (%)"].map("{:.3f}%".format)
        formatted_df.loc["Desal Marginal Cost (EUR/kg H2)"] = formatted_df.loc["Desal Marginal Cost (EUR/kg H2)"].map("€{:.4f}".format)
        formatted_df.loc["Desal Energy Overhead (GWh)"] = formatted_df.loc["Desal Energy Overhead (GWh)"].map("{:,.2f}".format)
        formatted_df.loc["Desal Specific Energy (kWh/kg H2)"] = formatted_df.loc["Desal Specific Energy (kWh/kg H2)"].map("{:.4f}".format)
        formatted_df.loc["Desal Share of Elec Demand (%)"] = formatted_df.loc["Desal Share of Elec Demand (%)"].map("{:.3f}%".format)

        f.write(formatted_df.to_string())
        f.write("\n\n" + "=" * 95 + "\n")
        f.write("Notes:\n")
        f.write(" - H2 output is aggregated across Electrolysis, SMR CC, and SMR technologies.\n")
        f.write(" - Water intensities applied: Electrolysis = 10 L/kg, SMR CC = 7.5 L/kg, SMR = 5 L/kg.\n")
        f.write(f" - Desalination cost estimate is based on {DESAL_COST_EUR_PER_M3:.2f} EUR/m³ of purified water.\n")
        f.write("=" * 95 + "\n")

    print(f"Saved desalination text table -> {water_table_path}")

    # ------------------------------------------------------------------
    # Generate Visual Water Demand & Desalination Cost Subfigures (.png)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    scenarios = df_water.columns
    water_volumes = df_water.loc["Water Demand (Million m3)"]
    desal_costs = df_water.loc["Est. Water Cost [M EUR/yr]"]

    # Colors defined per scenario: Scenario A (Green), Scenario B (Blue), Scenario C (Grey)
    scenario_color_map = {
    "Hybrid": "#9b59b6"}
    
    bar_colors = [scenario_color_map.get(sc, "#95a5a6") for sc in scenarios]

    x = np.arange(len(scenarios))
    bar_width = 0.55

    # Subfigure 1: Water Demand Volume
    rects1 = ax1.bar(x, water_volumes, width=bar_width, color=bar_colors, edgecolor='black', linewidth=0.6, alpha=0.9)
    ax1.set_ylabel('Water Demand [Million m³]', fontsize=11, fontweight='normal')
    ax1.set_title('Annual Water Demand Volume', fontsize=11, fontweight='normal')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=10)
    ax1.bar_label(rects1, fmt='%.2f Mm³', padding=3, fontsize=9, fontweight='normal')
    ax1.set_axisbelow(True)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    # Subfigure 2: Estimated Desalination Cost
    rects2 = ax2.bar(x, desal_costs, width=bar_width, color=bar_colors, edgecolor='black', linewidth=0.6, alpha=0.9)
    ax2.set_ylabel('Annualized Desalination Cost [Million €]', fontsize=11, fontweight='normal')
    ax2.set_title('Annual Desalination Cost', fontsize=11, fontweight='normal')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, fontsize=10)
    ax2.bar_label(rects2, fmt='€%.2f M', padding=3, fontsize=9, fontweight='normal')
    ax2.set_axisbelow(True)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    fig.suptitle('PyPSA-Libya: Water Demand and Desalination Overhead by Scenario', fontsize=13, fontweight='normal', y=1.02)
    fig.tight_layout()

    water_png_path = os.path.join(WATER_ANALYSIS_DIR, "water_demand_and_desalination_analysis.png")
    fig.savefig(water_png_path, dpi=200, bbox_inches="tight")
    save_paper_pdf(fig, ax, "water_demand_and_desalination_analysis")
    plt.close(fig)

    print(f"Saved desalination comparison figure -> {water_png_path}")




# =====================================================================
# SECTION 7: FINANCIAL ASSESSMENT, DYNAMIC NPV, IRR & PAYBACK COMPUTATION
# =====================================================================
print("\n" + "=" * 70)
print("Evaluating Financial Cash Flows, Dynamic NPV, IRR & Payback Period across Scenarios")
print("=" * 70)

import numpy_financial as npf

# Standard end-of-period discounting vector (t = 1 to 21) -> Sums to ~10.7099
discount_factors = 1.0 / ((1.0 + DEFAULT_WACC) ** np.arange(1, PROJECT_LIFETIME_YEARS + 1))
pv_annuity_factor = discount_factors.sum()

# Helper function to compute total overnight CAPEX from network component costs
def overnight_capex_total(n: pypsa.Network, discount_rate: float = DEFAULT_WACC, lifetime: int = PROJECT_LIFETIME_YEARS) -> float:
    ann_factor = (discount_rate * (1 + discount_rate)**lifetime) / (((1 + discount_rate)**lifetime) - 1)
    annualized_capex_total = capex_by_category(n).sum()
    return annualized_capex_total / ann_factor

# Pre-populate Baseline (No-H2 Export) network paths explicitly
BASELINE_RESULTS = {}
for scenario_name, path in RESULTS.items():
    if r"\No_Hydrogen_H2_0" not in path:
        path_0export = path.replace(r"\postnetworks", r"\No_Hydrogen_H2_0\postnetworks")
        path_0export = re.sub(r'(\d+)export\.nc$', r'0export.nc', path_0export)
    else:
        path_0export = path
    BASELINE_RESULTS[scenario_name] = path_0export

table1_records = []
table2_records = []
table3_records = []

for scenario_name, export_path in RESULTS.items():
    if not os.path.exists(export_path):
        print(f"[SKIP Financial Analysis] {scenario_name}: File not found at {export_path}")
        continue

    # Load Export Network & System Costs
    n_exp = pypsa.Network(export_path)
    ann_capex_exp = capex_by_category(n_exp).sum()
    opex_exp = operating_cost(n_exp)
    
    # CO2 Tax on Net Atmospheric (Vented) Emissions only — captured/sequestered CO2 is not taxed
    # Computed LIVE from n_exp (the network actually loaded here), not looked
    # up from co2_summary_results by scenario_name -- that dict is reset for
    # each case inside the CO2_RUN_CASES loop in Section 7 and, by the time
    # this section runs, only ever holds the LAST case processed there
    # (No_H2_Export), which silently fed the wrong scenario's emissions in.
    vented_mt = compute_vented_co2_mt(n_exp)
    co2_tax_annual_eur = vented_mt * 1e6 * CO2_TAX_EUR_PER_TON   
    export_run_cost_eur = ann_capex_exp + opex_exp   # tax-free cost, captured BEFORE tax is added below
    opex_exp += co2_tax_annual_eur
    total_ann_cost_exp = ann_capex_exp + opex_exp

    # Exported H2 Energy & Mass Calculation
    exported_h2_mwh = 0.0
    h2_export_loads = n_exp.loads[n_exp.loads.carrier.str.contains("H2|h2|export", case=False, na=False)]
    if not h2_export_loads.empty and not n_exp.loads_t.p.empty:
        weights = n_exp.snapshot_weightings.generators
        exported_h2_mwh = n_exp.loads_t.p[h2_export_loads.index].mul(weights, axis=0).sum().sum()
    if exported_h2_mwh <= 0:
        exported_h2_mwh = DEFAULT_EXPORT_H2_MWH

    exported_h2_kg = exported_h2_mwh / H2_LHV_MWH_PER_KG
    exported_h2_mtons = exported_h2_kg / 1e9

    # Load Baseline (0-Export Reference) Network
    base_path = BASELINE_RESULTS.get(scenario_name)
    if base_path and os.path.exists(base_path) and base_path != export_path:
        n_base = pypsa.Network(base_path)
        ann_capex_base = capex_by_category(n_base).sum()
        opex_base = operating_cost(n_base)
        overnight_capex_base = overnight_capex_total(n_base)
    else:
        ann_capex_base = 0.0
        opex_base = 0.0
        overnight_capex_base = 0.0

    total_ann_cost_base = ann_capex_base + opex_base
    net_ann_cost = total_ann_cost_exp - total_ann_cost_base
    net_lcoh_eur_mwh = net_ann_cost / exported_h2_mwh
    lcoh_lhv_cost = net_lcoh_eur_mwh * H2_LHV_MWH_PER_KG
    
    # Compute LCOH margin base WITHOUT CO2 tax
    net_ann_cost_pretax = (total_ann_cost_exp - co2_tax_annual_eur) - total_ann_cost_base
    lcoh_lhv_cost_pretax = (net_ann_cost_pretax / exported_h2_mwh) * H2_LHV_MWH_PER_KG
    selling_price = lcoh_lhv_cost_pretax + REVENUE_MARGIN   # price does NOT track the CO2 tax

    # # Dynamic Selling Price (+1 EUR/kg margin over Net LCOH LHV)
    # selling_price = lcoh_lhv_cost + REVENUE_MARGIN

    # Table 1: Annual Operational Metrics
    annual_cost_b = total_ann_cost_exp / 1e9
    annual_revenue_b = (exported_h2_kg * selling_price) / 1e9
    annual_net_cash_flow_b = annual_revenue_b - annual_cost_b

    # Table 2: Multi-Year Discounted Valuation Metrics
    pv_system_cost = annual_cost_b * pv_annuity_factor
    pv_net_h2_cost = (net_ann_cost / 1e9) * pv_annuity_factor
    pv_gross_revenue = annual_revenue_b * pv_annuity_factor
    npv_total_system = pv_gross_revenue - pv_system_cost

    # Table 3: Incremental Cash Inflows, Project IRR & Payback Period
    overnight_capex_exp = overnight_capex_total(n_exp)
    net_overnight_capex = max(0.0, overnight_capex_exp - overnight_capex_base)
    net_annual_opex = max(0.0, opex_exp - opex_base)
    annual_net_cash_inflow = (annual_revenue_b * 1e9) - net_annual_opex

    # Project IRR Cash Flow Vector: [-Net Overnight CAPEX, Net Annual Cash Inflow * 21 Years]
    cash_flows = [-net_overnight_capex] + [annual_net_cash_inflow] * PROJECT_LIFETIME_YEARS

    try:
        irr_val = npf.irr(cash_flows) * 100.0
    except Exception:
        irr_val = np.nan

    if annual_net_cash_inflow > 0 and net_overnight_capex > 0:
        payback_years = net_overnight_capex / annual_net_cash_inflow
    else:
        payback_years = np.nan

    net_cash_flow_b = (annual_net_cash_inflow - (net_overnight_capex / PROJECT_LIFETIME_YEARS)) / 1e9

    # Append Records
    table1_records.append({
        "Scenario": scenario_name,
        "LCOH LHV [EUR/kg]": lcoh_lhv_cost_pretax,
        "Price (+1) [EUR/kg]": selling_price,
        "Export Cost [B EUR/yr]": export_run_cost_eur / 1e9,
        "CO2 Tax [B EUR/yr]": co2_tax_annual_eur / 1e9,   
        "Ann. Cost [B EUR/yr]": annual_cost_b,
        "Ann. Revenue [B EUR/yr]": annual_revenue_b,
        "Ann. Net Cash Flow [B EUR/yr]": annual_net_cash_flow_b,
        # "H2 Exported [M Tons/yr]": exported_h2_mtons,
    })

    table2_records.append({
        "Scenario": scenario_name,
        "PV System Cost [B EUR]": pv_system_cost,
        "PV Net H2 Cost [B EUR]": pv_net_h2_cost,
        "PV Gross Revenue (+1 EUR) [B EUR]": pv_gross_revenue,
        "NPV Total System [B EUR]": npv_total_system,
    })

    table3_records.append({
        "Scenario": scenario_name,
        "Price [EUR/kg]": selling_price,
        "Annual Revenue [B EUR]": annual_revenue_b,
        "Net Cash Flow [B EUR]": net_cash_flow_b,
        "Project IRR (%)": irr_val,
        "Payback Period [Years]": payback_years
    })

# ------------------------------------------------------------------
# WRITE COMBINED SUMMARY REPORT: NPV_IRR_Payback_summary.txt
# ------------------------------------------------------------------
if table1_records:
    df_t1 = pd.DataFrame(table1_records).set_index("Scenario")
    df_t2 = pd.DataFrame(table2_records).set_index("Scenario")
    df_t3 = pd.DataFrame(table3_records).set_index("Scenario")

    summary_txt_path = os.path.join(OUT_DIR, "NPV_IRR_Payback_summary.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 105 + "\n")
        f.write("PyPSA-Libya: NPV Analysis with +1 Euro/kg Revenue Margin over Scenario LCOH Cost\n")
        f.write("=" * 105 + "\n\n")

        f.write("--- Configured Dynamic Financial & Margin Parameters ---\n")
        f.write(f"  Discount Rate (r):                    {DEFAULT_WACC * 100:.2f}%\n")
        f.write(f"  Operational Horizon:                 2030 - 2050 ({PROJECT_LIFETIME_YEARS} years inclusive)\n")
        f.write(f"  Revenue Margin Added to LCOH:        +{REVENUE_MARGIN:.2f} EUR/kg H2\n\n")

        f.write("--- Table 1: Scenario Specific LCOH, Dynamic Price & Annual Cash Flows ---\n")
        f.write(df_t1.to_string())
        f.write("\n\n" + "-" * 105 + "\n\n")

        f.write("--- Table 2: Multi-Year Discounted Valuation Summary (2030-2050) ---\n")
        f.write(df_t2.to_string())
        f.write("\n\n" + "-" * 105 + "\n\n")

        f.write("--- Table 3: Financial Cash Flow, Project IRR & Payback Assessment Summary ---\n")
        f.write(df_t3.to_string())
        f.write("\n\n" + "=" * 105 + "\n")

        f.write("Notes:\n")
        f.write(" - PV stands for Present Value (discounted cash flows over the 2030–2050 horizon at r = 7.10%).\n")
        f.write(" - Price is computed dynamically per scenario as (Net LCOH LHV + 1.00 EUR/kg).\n")
        f.write(" - Cash flow vector for Project IRR (%): Year 0 = -Net Overnight CAPEX, Years 1..N = Annual Net Cash Inflow.\n")
        f.write(" - Payback Period [Years] = Net Overnight CAPEX / Annual Net Operating Cash Inflow.\n")
        f.write(f"  CO2 Tax Rate Applied:                +{CO2_TAX_EUR_PER_TON:.2f} EUR/tonne (on Net Atmospheric/Vented CO2 only)\n\n")
        f.write("=" * 105 + "\n")

    print(f"\nSaved combined NPV, IRR & Payback summary report -> {summary_txt_path}")




# =====================================================================
# 8. ELECTRICITY-ONLY PHYSICAL GENERATION BY CARRIER (Hybrid Scenario)
# =====================================================================

print("\n" + "=" * 70)
print("Generating Section 8: Electricity-Only Physical Generation Analysis (Hybrid Scenario)")
print("=" * 70)

# Carrier definitions strictly belonging to the Electricity Sector
ELEC_ONLY_GEN_CARRIERS = ["solar", "solar rooftop", "onwind", "offwind", "oil"]
ELEC_ONLY_LINK_CARRIERS = ["CCGT", "OCGT"]

for scenario_name, path in RESULTS.items():
    if not os.path.exists(path):
        print(f"[SKIP Section 8] {scenario_name}: File not found at '{path}'")
        continue

    print(f"\nProcessing Electricity-Only Generation Profile for: {scenario_name}...")
    n_scen = pypsa.Network(path)
    safe_prefix = scenario_name.replace(" ", "_")
    
    # Extract resolution timestep in hours
    dt_hours = (n_scen.snapshots[1] - n_scen.snapshots[0]).total_seconds() / 3600.0 if len(n_scen.snapshots) > 1 else 1.0

    elec_gen_dict = {}

    # 1. Generators (Solar, Wind, Oil)
    if not n_scen.generators.empty and not n_scen.generators_t.p.empty:
        elec_gens = n_scen.generators[n_scen.generators.carrier.isin(ELEC_ONLY_GEN_CARRIERS)]
        if not elec_gens.empty:
            sel_gens = [g for g in elec_gens.index if g in n_scen.generators_t.p.columns]
            if sel_gens:
                gen_p = n_scen.generators_t.p[sel_gens].groupby(
                    n_scen.generators.loc[sel_gens, "carrier"].map(map_outcome_carrier), axis=1
                ).sum()
                for carrier in gen_p.columns:
                    energy_twh = (gen_p[carrier] * dt_hours).sum() / 1e6
                    if energy_twh > 1e-6:
                        elec_gen_dict[carrier] = elec_gen_dict.get(carrier, 0.0) + energy_twh

    # 2. Thermal Links (CCGT, OCGT) — Output Generation (p0 * efficiency)
    if not n_scen.links.empty and not n_scen.links_t.p0.empty:
        elec_links = n_scen.links[n_scen.links.carrier.isin(ELEC_ONLY_LINK_CARRIERS)]
        if not elec_links.empty:
            sel_links = [l for l in elec_links.index if l in n_scen.links_t.p0.columns]
            if sel_links:
                eff = n_scen.links.loc[sel_links, "efficiency"].fillna(1.0)
                link_output = n_scen.links_t.p0[sel_links].mul(eff, axis=1)
                link_p = link_output.groupby(
                    n_scen.links.loc[sel_links, "carrier"].map(map_outcome_carrier), axis=1
                ).sum()
                for carrier in link_p.columns:
                    energy_twh = (link_p[carrier] * dt_hours).sum() / 1e6
                    if energy_twh > 1e-6:
                        elec_gen_dict[carrier] = elec_gen_dict.get(carrier, 0.0) + energy_twh

    elec_gen_twh = pd.Series(elec_gen_dict).sort_values(ascending=False)

    if elec_gen_twh.empty:
        print(f"  [SKIP Section 8] No electricity-only generation activity detected for {scenario_name}.")
        continue

    # Plot Electricity-Only Physical Generation Vertical Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))

    for carrier, val in elec_gen_twh.items():
        color = CHART_COLORS.get(carrier, PRIMARY_CARRIER_COLORS.get(carrier, "#95a5a6"))
        ax.bar(carrier, val, color=color, alpha=0.9, edgecolor="black", linewidth=0.6)

    ax.set_ylabel("Electricity Generation (TWh)", fontsize=11)
    ax.set_title(
        f"PyPSA-Libya — {scenario_name}: Electricity-Only Physical Generation by Carrier",
        fontsize=12,
        fontweight="bold"
    )
    ax.margins(y=0.18)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#cbd5e1")
    plt.xticks(rotation=15, ha="right", fontsize=10)

    # Numerical Annotations Above Vertical Bars in TWh
    for p in ax.patches:
        height = p.get_height()
        if height >= 0:
            # Format display dynamically for small values (e.g. 0.005 TWh vs 24.94 TWh)
            label_text = f"{height:,.3f} TWh" if height < 0.1 else f"{height:,.2f} TWh"
            ax.annotate(
                label_text,
                (p.get_x() + p.get_width() / 2.0, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="normal"
            )

    fig.tight_layout()

    out_elec_png = os.path.join(CARRIERS_DIR, f"{safe_prefix}_electricity_only_generation_by_carrier.png")
    fig.savefig(out_elec_png, dpi=200, bbox_inches="tight")
    save_paper_pdf(fig, ax, f"{safe_prefix}_electricity_only_generation_by_carrier")
    plt.close(fig)
    print(f"Saved Vertical Electricity-Only Generation Figure -> {out_elec_png}")

    # Export Text Report Summary for Section 8
    txt_elec_path = os.path.join(CARRIERS_DIR, f"{safe_prefix}_elec_only_generation_summary.txt")
    with open(txt_elec_path, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write(f"PyPSA-Libya: Electricity-Only Physical Generation ({scenario_name})\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"{'Carrier / Technology':30s} | {'Generation (TWh)':18s} | {'Share (%)':10s}\n")
        f.write("-" * 65 + "\n")
        total_elec_twh = elec_gen_twh.sum()
        for carrier, val in elec_gen_twh.items():
            share_pct = (val / total_elec_twh * 100.0) if total_elec_twh > 0 else 0.0
            f.write(f"{carrier:30s} | {val:18.4f} | {share_pct:9.2f}%\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'TOTAL Electricity Generation':30s} | {total_elec_twh:18.4f} | 100.00%\n")
        f.write("=" * 65 + "\n")

    print(f"Saved Electricity-Only Summary Text Report -> {txt_elec_path}")
    
    


# =====================================================================
# 9. OPEX BREAKDOWN ANALYSIS (STACKED BAR CHART - BOTH SCENARIOS)
# =====================================================================

print("\n" + "=" * 70)
print("Computing OPEX Breakdown for Hybrid Model")
print("=" * 70)

# Extended OPEX Carrier Map matching diagnosis breakdown standards
OPEX_CARRIER_MAP = {
    # Thermal & Primary Fuel Inputs
    "gas": "Gas",
    "natural gas": "Gas",
    "gas supply": "Gas",
    "gas for industry": "Gas",
    "gas for industry CC": "Gas",
    "oil": "Oil",
    "heavy fuel oil": "Oil",
    "diesel": "Oil",
    "CCGT": "CCGT",
    "OCGT": "OCGT",
    # Renewable Generation
    "solar": "Solar PV",
    "Solar PV": "Solar PV",
    "onwind": "Onshore Wind",
    "Offshore Wind": "Offshore Wind",
    "offwind": "Offshore Wind",
    # Hydrogen & Industrial Components
    "H2 Electrolysis": "Electrolysis",
    "electrolysis": "Electrolysis",
    "SMR": "SMR",
    "SMR CC": "SMR with CC",
    "H2": "H2 Storage",
    "H2 store": "H2 Storage",
    "H2 Store": "H2 Storage",
    "H2 Store Tank": "H2 Storage",
    "H2 Store Underground": "H2 Storage",
    "H2 UHS": "H2 Storage",
    "H2 Fuel Cell": "H2 Fuel Cell",
    "H2 turbine": "H2 Fuel Cell",
    "co2 sequestered": "CO2 sequestration",
    "co2 sequestration": "CO2 sequestration",
    "H2 pipeline": "H2 pipeline",
    # Infrastructure & Grid Extensions
    "electricity distribution grid": "Grid Expansion",
    "process emissions": "Process Emissions",
    "load shedding": "Load Shedding",
    "Haber-Bosch": "Haber-Bosch"
}

# Unified Color Palette
OPEX_CHART_COLORS = {
    "Gas": "#7f8c8d",                  # Grey/Slate fuel color
    "Gas Supply": "#7f8c8d",           # Grey/Slate fuel color
    "Oil": "#8b4513",                  # Brown
    "Solar PV": "#f5a623",             # Yellow/Orange
    "Onshore Wind": "#2ecc71",         # Light Green
    "Offshore Wind": "#1abc9c",        # Dark Teal
    "Electrolysis": "#7b2d8e",         # Purple
    "SMR": "#4f4f2f",                  # Olive/Dark Grey
    "SMR with CC": "#1a5276",          # Dark Blue
    "H2 Storage": "#3498db",           # Bright Blue
    "H2 Fuel Cell": "cyan",            # Cyan
    "CCGT": "#95a5a6",                 # Light Grey
    "OCGT": "#e74c3c",                 # Soft Red
    "CO2 sequestration": "#0b1f3a",    # Navy Blue
    "H2 pipeline": "#c0392b",          # Red
    "CO2 Tax": "#000000",              # Black
    "Grid Expansion": "#f9b7cc",       # Pink
    "Process Emissions": "#d35400",    # Rust
    "Load Shedding": "#e74c3c",        # Red
    "Haber-Bosch": "#16a085"           # Green Teal
}


def opex_by_category(n: pypsa.Network) -> pd.Series:
    """Calculates OPEX grouped by carrier category incorporating marginal operational
    costs, primary fuel supply costs (Gas/Oil), and component dispatches using mapped carriers.
    """
    records = []
    weights = n.snapshot_weightings.generators

    def clean_map(carrier_series):
        return carrier_series.astype(str).str.strip().map(lambda c: OPEX_CARRIER_MAP.get(c, c))

    # 1. Generators OPEX (Primary Fuel Supply Generators like Gas/Oil)
    if not n.generators.empty and not n.generators_t.p.empty:
        g = n.generators
        sel_g = [c for c in g.index if c in n.generators_t.p.columns]
        if sel_g:
            mc = g.loc[sel_g, "marginal_cost"]
            cost = n.generators_t.p[sel_g].mul(mc, axis=1).mul(weights, axis=0).sum(axis=0)
            records.append(pd.DataFrame({
                "category": clean_map(g.loc[sel_g, "carrier"]), 
                "cost": cost.values
            }))

    # 2. Links OPEX (Marginal Cost x Primary Input Power p0)
    if not n.links.empty and not n.links_t.p0.empty:
        l = n.links
        sel_l = [c for c in l.index if c in n.links_t.p0.columns]
        if sel_l:
            mc = l.loc[sel_l, "marginal_cost"]
            cost = n.links_t.p0[sel_l].mul(mc, axis=1).mul(weights, axis=0).sum(axis=0)
            records.append(pd.DataFrame({
                "category": clean_map(l.loc[sel_l, "carrier"]), 
                "cost": cost.values
            }))

    # 3. Stores OPEX
    if not n.stores.empty and hasattr(n, "stores_t") and hasattr(n.stores_t, "p") and not n.stores_t.p.empty:
        s = n.stores
        sel_s = [c for c in s.index if c in n.stores_t.p.columns]
        if sel_s and "marginal_cost" in s.columns:
            mc = s.loc[sel_s, "marginal_cost"]
            cost = n.stores_t.p[sel_s].mul(mc, axis=1).mul(weights, axis=0).sum(axis=0)
            records.append(pd.DataFrame({
                "category": clean_map(s.loc[sel_s, "carrier"]), 
                "cost": cost.values
            }))

    # 4. Storage Units OPEX
    if not n.storage_units.empty and not n.storage_units_t.p.empty:
        su = n.storage_units
        sel_su = [c for c in su.index if c in n.storage_units_t.p.columns]
        if sel_su and "marginal_cost" in su.columns:
            mc = su.loc[sel_su, "marginal_cost"]
            cost = n.storage_units_t.p[sel_su].clip(lower=0).mul(mc, axis=1).mul(weights, axis=0).sum(axis=0)
            records.append(pd.DataFrame({
                "category": clean_map(su.loc[sel_su, "carrier"]), 
                "cost": cost.values
            }))

    if not records:
        return pd.Series(dtype=float)

    df_all = pd.concat(records, ignore_index=True)
    return df_all.groupby("category")["cost"].sum()


OPEX_RUN_CASES = {
    "H2_Export": {
        "results": RESULTS,
        "prefix": "",
        "title_label": "Export Hydrogen"
    },
    "No_H2_Export": {
        "results": {
            "Hybrid": r"Output\libya_h2export_hybrid_emispr_100\3hrs\No_Hydrogen_H2_0\postnetworks\elec_s_12_ec_lcopt_Co2L-Ep-3h__2030_0.071_AB_0export.nc",
        },
        "prefix": "No_H2_export_",
        "title_label": "No Export Hydrogen"
    },
}

for case_key, case_info in OPEX_RUN_CASES.items():
    results_dict = case_info["results"]
    prefix = case_info["prefix"]
    title_label = case_info["title_label"]

    opex_scenario_costs = {}

    for scenario_name, path in results_dict.items():
        if not os.path.exists(path):
            print(f"[SKIP OPEX] {scenario_name} ({case_key}): File not found at '{path}'")
            continue

        print(f"Extracting OPEX for {scenario_name} ({case_key})...")
        n_scen = pypsa.Network(path)

        opex_breakdown = opex_by_category(n_scen)
        
        # Compute vented CO2 directly from the network loaded in THIS
        # iteration (correct for both H2_Export and No_H2_Export cases)
        vented_mt = compute_vented_co2_mt(n_scen)
        co2_tax = vented_mt * 1e6 * CO2_TAX_EUR_PER_TON
        
        if co2_tax > 0:
            opex_breakdown["CO2 Tax"] = co2_tax

        opex_scenario_costs[scenario_name] = opex_breakdown

    if opex_scenario_costs:
        # 1. Full OPEX DataFrame (Unfiltered)
        df_opex_full = pd.DataFrame(opex_scenario_costs).T / 1e9  # Convert EUR to Billion EUR
        df_opex_full = df_opex_full.fillna(0.0)

        preferred_order = [
            "Gas", "Gas Supply", "Oil", "SMR", "H2 Storage", "Grid Expansion",
            "Process Emissions", "Load Shedding", "Haber-Bosch",
            "Solar PV", "Onshore Wind", "Offshore Wind", "CCGT", "OCGT",
            "Electrolysis", "H2 Fuel Cell", "SMR with CC", "CO2 sequestration",
            "H2 pipeline", "CO2 Tax"
        ]
        
        # Filter ordering list strictly to categories present in full dataframe
        category_order_opex = [c for c in preferred_order if c in df_opex_full.columns]
        for col in df_opex_full.columns:
            if col not in category_order_opex:
                category_order_opex.append(col)

        df_opex_full = df_opex_full[category_order_opex]

        # 2. Filtered OPEX DataFrame for Visualizations & Significant Components (>= 0.001 Billion EUR)
        significant_cols = df_opex_full.columns[df_opex_full.max(axis=0) >= 0.001]
        df_opex_filtered = df_opex_full[significant_cols]

        # 3. Safe Stacked Bar Chart Generation
        fig_opex, ax_opex = plt.subplots(figsize=(5.5, 6))
        bottom_opex = pd.Series(0.0, index=df_opex_filtered.index)

        # Iterate safely over df_opex_filtered columns
        for cat in df_opex_filtered.columns:
            values = df_opex_filtered[cat]
            ax_opex.bar(
                df_opex_filtered.index, values, bottom=bottom_opex,
                color=OPEX_CHART_COLORS.get(cat, "#7f8c8d"), label=cat,
                edgecolor="black", linewidth=0.5, width=0.45
            )
            bottom_opex += values

        ax_opex.set_ylabel("Annual Operational Cost [Billion EUR]", fontsize=11)
        ax_opex.set_title(f"PyPSA-Libya: Operational Expenditure (OPEX) Breakdown\nHybrid Model ({title_label})", fontsize=11, fontweight="bold")
        ax_opex.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, frameon=True)
        ax_opex.grid(axis="y", linestyle="--", alpha=0.5)
        ax_opex.set_xlim(-0.6, len(df_opex_filtered.index) - 0.4)

        fig_opex.tight_layout()

        # Save Figures
        png_filename_opex = f"figure9d_{prefix}opex_cost_hybrid.png" if prefix else "figure9d_opex_cost_hybrid.png"
        out_path_opex = os.path.join(OUT_DIR, png_filename_opex)
        
        fig_opex.savefig(out_path_opex, dpi=300, bbox_inches="tight")
        pdf_key_opex = f"{prefix}opex_cost_hybrid" if prefix else "opex_cost_hybrid"
        save_paper_pdf(fig_opex, ax_opex, pdf_key_opex)
        plt.close(fig_opex)
        print(f"Saved OPEX breakdown chart -> {out_path_opex}")

        # 4. Output Detailed Text Summary File
        txt_filename_opex = f"Breakdown_{prefix}opex_cost_summary.txt" if prefix else "Breakdown_opex_cost_summary.txt"
        txt_out_path_opex = os.path.join(OUT_DIR, txt_filename_opex)
        df_opex_totals = df_opex_full.sum(axis=1)

        # Extract Top OPEX Components per scenario
        top5_dict = {}
        for scenario_k in df_opex_full.index:
            top5 = df_opex_full.loc[scenario_k].nlargest(10)
            top5_dict[scenario_k] = top5
        df_top5 = pd.DataFrame(top5_dict)
        
        with open(txt_out_path_opex, "w", encoding="utf-8") as f:
            f.write("=" * 100 + "\n")
            f.write(f"PyPSA-Libya: Operational Expenditure (OPEX) Breakdown Summary ({title_label}) (Billion EUR/year)\n")
            f.write("=" * 100 + "\n\n")
            f.write("--- Itemized OPEX Breakdown (>= 0.001 Billion EUR/year) ---\n")
            f.write(df_opex_filtered.round(4).to_string())
            f.write("\n\n" + "-" * 100 + "\n")
            f.write("--- Top 10 Shared OPEX Components (Billion EUR/year) ---\n")
            f.write(df_top5.round(4).to_string())
            f.write("\n\n" + "-" * 100 + "\n")
            f.write("--- Total OPEX Summary ---\n")
            for scenario_k, total_val in df_opex_totals.items():
                f.write(f"  {scenario_k:12s} Total OPEX: {total_val:8.3f} Billion EUR/year\n")
            f.write("=" * 100 + "\n")

        print(f"Saved OPEX breakdown text report -> {txt_out_path_opex}\n")
    
    
    
    
# =====================================================================
# FINAL PART: ORGANIZE TXT SUMMARIES FOR LATEX
# =====================================================================

print("\n" + "=" * 70)
print("Organizing TXT Summaries for Paper")
print("=" * 70)

import shutil

# List of folders to scan for text summaries
folders_to_scan = [OUT_DIR, CARRIERS_DIR]
table_count = 0

for folder in folders_to_scan:
    if not os.path.exists(folder):
        continue
        
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        
        # Handle Tables (Copy .txt files containing 'summary')
        if file.lower().endswith(".txt") and "summary" in file.lower():
            try:
                shutil.copy(file_path, os.path.join(PAPER_DIR, file))
                print(f" Copied Table: {file}")
                table_count += 1
            except Exception as e:
                print(f" [ERROR] Could not copy {file}: {e}")

print(f"\nSummary of Export:")
print(f" - Paper-ready PDFs (Directly saved): Created during run")
print(f" - Summary TXT files copied: {table_count}")
print(f" Output Location: ./{PAPER_DIR}/")
print("=" * 70)