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
import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =====================================================================
# PATH CONFIGURATION
# =====================================================================

#### H2 export, week

# RESULTS = {
#     "Scenario A": r"Output\libya_h2export_test\scenario_a\postnetworks\elec_s_36_ec_lcopt_Co2L-3h_144h_2030_0.071_AB_0.25export.nc",
#     "Scenario B": r"Output\libya_h2export_test\scenario_b\postnetworks\elec_s_36_ec_lcopt_Co2L-3h_144h_2030_0.071_AB_0.25export.nc",
#     "Scenario C": r"Output\libya_h2export_test\scenario_c\postnetworks\elec_s_36_ec_lcopt_Co2L-3h_144h_2030_0.071_AB_0.25export.nc",
#           }


### H2 export, year, 6-hrs 

# RESULTS = {
#     "Scenario A": r"Output\libya_h2export_test\scenario_a\postnetworks\elec_s_36_ec_lcopt_Co2L-6h_144h_2030_0.071_AB_70export.nc",
#     "Scenario B": r"Output\libya_h2export_test\scenario_b\postnetworks\elec_s_36_ec_lcopt_Co2L-6h_144h_2030_0.071_AB_70export.nc",
#     "Scenario C": r"Output\libya_h2export_test\scenario_c\postnetworks\elec_s_36_ec_lcopt_Co2L-6h_144h_2030_0.071_AB_70export.nc",
#           }


### H2 export, year, 6-hrs 

RESULTS = {
    "Scenario A": r"Output\libya_h2export_test\scenario_a\3hrs\postnetworks\elec_s_36_ec_lcopt_Co2L-3h_144h_2030_0.071_AB_70export.nc",
    "Scenario B": r"Output\libya_h2export_test\scenario_b\3hrs\postnetworks\elec_s_36_ec_lcopt_Co2L-3h_144h_2030_0.071_AB_70export.nc",
    "Scenario C": r"Output\libya_h2export_test\scenario_c\3hrs\postnetworks\elec_s_36_ec_lcopt_Co2L-3h_144h_2030_0.071_AB_70export.nc",
          }


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

    # ------------------------------------------------------------------
    # 2. Spatial Investment Maps for All Scenarios (A, B, and C)
    # ------------------------------------------------------------------
    def per_bus_capex(net):
        frames = []
        if not net.generators.empty:
            g = net.generators
            cost = g.capital_cost * g.p_nom_opt
            cat = g.carrier.map(lambda c: CAPEX_CARRIER_MAP.get(c))
            frames.append(pd.DataFrame({"bus": g.bus, "category": cat, "cost": cost}))
        if not net.links.empty:
            l = net.links
            cost = l.capital_cost * l.p_nom_opt
            cat = l.carrier.map(lambda c: CAPEX_CARRIER_MAP.get(c))
            frames.append(pd.DataFrame({"bus": l.bus0, "category": cat, "cost": cost}))
        
        if not frames:
            return pd.DataFrame()
            
        res = pd.concat(frames, ignore_index=True)
        return res.dropna(subset=["category"])

    def strip_h2_suffix(b):
        return b[:-3] if b.endswith(" H2") else b

    print("\nGenerating Spatial Investment Maps...")

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

        for bus in pivot.index:
            if bus not in buses.index:
                continue
            x, y = buses.loc[bus, ["x", "y"]]
            total = total_by_bus[bus]
            if total <= 0:
                continue
            size = 200 + 3000 * (total / max_cost)
            dominant = pivot.loc[bus].idxmax()
            ax.scatter(x, y, s=size, c=CHART_COLORS.get(dominant, "gray"),
                       edgecolors="black", linewidths=0.5, alpha=0.85, zorder=3)
            ax.annotate(f"{total/1e6:,.0f}M", (x, y), fontsize=7, fontweight="bold",
                        textcoords="offset points", xytext=(5, 5))

        for _, row in n.lines.iterrows():
            if row.bus0 in buses.index and row.bus1 in buses.index:
                x0, y0 = buses.loc[row.bus0, ["x", "y"]]
                x1, y1 = buses.loc[row.bus1, ["x", "y"]]
                ax.plot([x0, x1], [y0, y1], c="gray", lw=0.5, alpha=0.5, zorder=1)

        legend_handles = [mpatches.Patch(color=c, label=k) for k, c in CHART_COLORS.items() if k in pivot.columns]
        ax.legend(handles=legend_handles, loc="lower left", fontsize=8)
        ax.set_title(f"PyPSA-Libya — {scenario_name}: Spatial Investment Allocation\n"
                     "Marker size = Total Capex at node; Color = Dominant Technology", fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle=":", alpha=0.6)

        safe_name = scenario_name.lower().replace(" ", "_")
        out_path = os.path.join(OUT_DIR, f"figure9b_style_investment_map_{safe_name}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved spatial map -> {out_path}")

print("\n" + "=" * 70)
print("Execution Complete.")
print("=" * 70)