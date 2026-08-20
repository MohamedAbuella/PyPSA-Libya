"""
compute_h2_scenario_costs.py

Replicates Figure 9(a) and 9(b) from Abdel-Khalek et al. 2025
(PyPSA-Earth sector-coupled, Applied Energy 383, 125316):
  (a) stacked bar chart of ANNUALIZED ADDITIONAL SYSTEM COST by technology,
      across your 3 scenarios (a/b/c), where "additional" = cost of the
      WITH-export run minus the NO-export run for that same scenario.
  (b) geospatial map of investment magnitude by bus/region for scenario a
      (with export), pie-chart-style markers colored by technology.

IMPORTANT: you must have already run all 6 solves before this script is
useful — 3 scenarios x {with export, without export}. See the 6 config
files (config_scenario_{a,b,c}_{export,noexport}.yaml) for the exact
setup for each.

Run from the pypsa-libya folder, pypsa-earth conda environment:
    python compute_h2_scenario_costs.py
"""

import os
import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =====================================================================
# EDIT THESE PATHS once you know your actual output filenames (check
# results/<run_name>/postnetworks/ after each solve completes — the
# discountrate/clusters/etc. segments in the filename come from your
# config and should match what's shown here if you used the provided
# scenario configs unchanged).
# =====================================================================
RESULTS = {
    "scenario a": {
        "export":   r"results\libya_h2export_scenario_a\postnetworks\elec_s_36_ec_lcopt_Co2L-3h_144h_2030_0.071_AB_0.25export.nc",
        "noexport": r"results\libya_h2export_scenario_a_noexport\postnetworks\elec_s_36_ec_lcopt_Co2L-3h_144h_2030_0.071_AB_0export.nc",
    },
    "scenario b": {
        "export":   r"results\libya_h2export_scenario_b\postnetworks\elec_s_36_ec_lcopt_Co2L-3h_144h_2030_0.071_AB_0.25export.nc",
        "noexport": r"results\libya_h2export_scenario_b_noexport\postnetworks\elec_s_36_ec_lcopt_Co2L-3h_144h_2030_0.071_AB_0export.nc",
    },
    "scenario c": {
        "export":   r"results\libya_h2export_scenario_c\postnetworks\elec_s_36_ec_lcopt_Co2L-3h_144h_2030_0.071_AB_0.25export.nc",
        "noexport": r"results\libya_h2export_scenario_c_noexport\postnetworks\elec_s_36_ec_lcopt_Co2L-3h_144h_2030_0.071_AB_0export.nc",
    },
}

OUT_DIR = "viz_outputs/h2_scenario_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

# Map raw PyPSA carrier names -> the display categories used in Figure 9a.
# Anything not listed here falls into "Operating cost" if it's an opex
# term, or is dropped from the capex breakdown (e.g. unrelated
# electricity-only generators like existing CCGT/oil that aren't part of
# the H2 export story).
CAPEX_CARRIER_MAP = {
    "H2 Electrolysis": "Electrolysis",
    "solar": "Solar PV",
    "SMR": "SMR",
    "SMR CC": "SMR with CC",
    "co2 sequestered": "CO2 sequestration",
    "co2 sequestration": "CO2 sequestration",
    "H2 pipeline": "H2 pipeline",
}

CHART_COLORS = {
    "Electrolysis": "#7b2d8e",
    "Solar PV": "#f5a623",
    "CO2 sequestration": "#0b1f3a",
    "SMR": "#4F4F2F",
    "SMR with CC": "#1a5276",
    "H2 pipeline": "#c0392b",
    "Operating cost": "#d2b48c",
}


def capex_by_category(n: pypsa.Network) -> pd.Series:
    """Annualized capital cost (capital_cost x optimal size), grouped into
    Figure-9a-style display categories."""
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
        # incremental line expansion cost only (s_nom_opt - s_nom), not
        # the full nominal capacity, since only NEW capacity is "invested"
        parts.append((pd.Series("H2 pipeline", index=ln.index),
                       ln.capital_cost * (ln.s_nom_opt - ln.s_nom).clip(lower=0)))

    combined = pd.concat(
        [pd.Series(cost.values, index=carrier.values) for carrier, cost in parts]
    )
    combined = combined.groupby(level=0).sum()

    # Map to display categories; anything unmapped is dropped from capex
    # (it's not part of the H2-export cost story - e.g. existing CCGT/oil)
    mapped = combined.rename(index=lambda c: CAPEX_CARRIER_MAP.get(c, None))
    mapped = mapped[mapped.index.notna()]
    mapped = mapped.groupby(level=0).sum()
    return mapped


def operating_cost(n: pypsa.Network) -> float:
    """Total operating cost = marginal_cost x dispatch x snapshot weighting,
    summed across all dispatchable components. Returned as a single lumped
    number, matching how Figure 9a shows one 'Operating cost' segment."""
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
print("Computing annualized ADDITIONAL system cost by scenario")
print("(export run minus no-export run, per scenario)")
print("=" * 70)

scenario_costs = {}  # scenario name -> pd.Series of category -> cost (delta)

for scenario_name, paths in RESULTS.items():
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        print(f"\n[SKIP] {scenario_name}: missing {missing} — "
              f"run that solve first, or fix the path at the top of this script.")
        continue

    n_export = pypsa.Network(paths["export"])
    n_noexport = pypsa.Network(paths["noexport"])

    capex_export = capex_by_category(n_export)
    capex_noexport = capex_by_category(n_noexport)
    capex_delta = capex_export.subtract(capex_noexport, fill_value=0.0)

    opex_delta = operating_cost(n_export) - operating_cost(n_noexport)

    combined = capex_delta.copy()
    combined["Operating cost"] = opex_delta
    combined = combined.clip(lower=0)  # additional cost shouldn't go negative;
                                         # if it does, that's worth investigating
                                         # separately, not silently plotting

    scenario_costs[scenario_name] = combined
    print(f"\n{scenario_name}:")
    print((combined / 1e9).round(3).to_string())  # billions of currency units
    print(f"  TOTAL additional cost: {combined.sum() / 1e9:,.3f} B")

if not scenario_costs:
    print("\nNo scenarios had both export/no-export results available yet — "
          "nothing to plot. Run the solves first.")
else:
    # ------------------------------------------------------------------
    # Figure 9(a)-style stacked bar chart
    # ------------------------------------------------------------------
    df = pd.DataFrame(scenario_costs).T / 1e9  # billions, rows=scenario
    df = df.fillna(0.0)

    # consistent category order matching the reference figure's stacking
    category_order = [c for c in
                       ["Electrolysis", "Solar PV", "CO2 sequestration",
                        "SMR", "SMR with CC", "H2 pipeline", "Operating cost"]
                       if c in df.columns]
    df = df[category_order]

    fig, ax = plt.subplots(figsize=(8, 6))
    bottom = pd.Series(0.0, index=df.index)
    for cat in category_order:
        ax.bar(df.index, df[cat], bottom=bottom,
               color=CHART_COLORS.get(cat, "gray"), label=cat,
               edgecolor="black", linewidth=0.5)
        bottom += df[cat]

    ax.set_ylabel("Annualized additional system cost [B]")
    ax.set_title("PyPSA-Libya — H2 export: additional system cost by scenario\n"
                  "(with-export minus no-export, per scenario)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "figure9a_style_cost_by_scenario.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {out_path}")

    # ------------------------------------------------------------------
    # Figure 9(b)-style spatial investment map for scenario a (if available)
    # ------------------------------------------------------------------
    if "scenario a" in RESULTS and os.path.exists(RESULTS["scenario a"]["export"]):
        n = pypsa.Network(RESULTS["scenario a"]["export"])

        # per-bus annualized capex, by category (same categories as above)
        def per_bus_capex(n):
            frames = []
            if not n.generators.empty:
                g = n.generators
                cost = g.capital_cost * g.p_nom_opt
                cat = g.carrier.map(lambda c: CAPEX_CARRIER_MAP.get(c))
                frames.append(pd.DataFrame({"bus": g.bus, "category": cat, "cost": cost}))
            if not n.links.empty:
                l = n.links
                cost = l.capital_cost * l.p_nom_opt
                cat = l.carrier.map(lambda c: CAPEX_CARRIER_MAP.get(c))
                frames.append(pd.DataFrame({"bus": l.bus0, "category": cat, "cost": cost}))
            df = pd.concat(frames, ignore_index=True)
            df = df.dropna(subset=["category"])
            return df

        bus_cost = per_bus_capex(n)
        pivot = bus_cost.pivot_table(index="bus", columns="category",
                                      values="cost", aggfunc="sum", fill_value=0.0)

        # strip carrier suffixes to map sub-buses back to primary location
        # (same fix as in the earlier corrected viz script)
        def strip_h2_suffix(b):
            return b[:-3] if b.endswith(" H2") else b
        pivot.index = [strip_h2_suffix(b) for b in pivot.index]
        pivot = pivot.groupby(level=0).sum()

        buses = n.buses.loc[n.buses.index.intersection(pivot.index)]
        fig, ax = plt.subplots(figsize=(9, 9))
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
            ax.annotate(f"{total/1e6:,.0f}M", (x, y), fontsize=6,
                        textcoords="offset points", xytext=(5, 5))

        for _, row in n.lines.iterrows():
            if row.bus0 in buses.index and row.bus1 in buses.index:
                x0, y0 = buses.loc[row.bus0, ["x", "y"]]
                x1, y1 = buses.loc[row.bus1, ["x", "y"]]
                ax.plot([x0, x1], [y0, y1], c="gray", lw=0.5, alpha=0.5, zorder=1)

        legend_handles = [mpatches.Patch(color=c, label=k) for k, c in CHART_COLORS.items()]
        ax.legend(handles=legend_handles, loc="lower left", fontsize=8)
        ax.set_title("PyPSA-Libya — Scenario A: spatial investment allocation\n"
                     "Marker size = total capex at bus; color = dominant technology")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        out_path = os.path.join(OUT_DIR, "figure9b_style_investment_map_scenario_a.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved -> {out_path}")
    else:
        print("\n[SKIP map] scenario a export results not found.")

print("\n" + "=" * 70)
print("Done.")
print("=" * 70)
