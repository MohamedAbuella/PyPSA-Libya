"""
visualize_libya.py
Run from the pypsa-libya folder, in the pypsa-earth conda environment:

    python visualize_libya.py

Produces two sets of PNGs in a new `viz_outputs/` folder:
  1) network_topology_<stage>.png  — buses + lines at each pipeline stage
     (base -> elec -> elec_s -> elec_s_8 -> elec_s_8_ec), so you can see
     exactly where the line count drops to zero.
  2) solved_capacity_by_carrier.png — bar chart of p_nom_opt by carrier
     solved_capacity_map.png        — geographic map, bus markers sized by
                                       total installed capacity, colored by
                                       dominant carrier at that bus
"""

import os
import pypsa
import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = "viz_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# PART 1 — Topology at each pipeline stage
# ---------------------------------------------------------------------

STAGES = [
    ("base",        "networks/libya-test/base.nc"),
    ("elec",        "networks/libya-test/elec.nc"),
    ("elec_s",      "networks/libya-test/elec_s.nc"),
    ("elec_s_8",    "networks/libya-test/elec_s_8.nc"),
    ("elec_s_8_ec", "networks/libya-test/elec_s_8_ec.nc"),
]

print("=" * 70)
print("PART 1: Network topology at each pipeline stage")
print("=" * 70)

for stage_name, path in STAGES:
    if not os.path.exists(path):
        print(f"  [skip] {stage_name}: file not found at {path}")
        continue

    n = pypsa.Network(path)
    n_lines = len(n.lines)
    n_buses = len(n.buses)
    print(f"  {stage_name:14s} buses={n_buses:4d}  lines={n_lines:4d}")

    fig, ax = plt.subplots(figsize=(8, 8))
    try:
        n.plot(
            ax=ax,
            bus_sizes=0.02,
            bus_color="tab:blue",
            line_widths=1.5 if n_lines > 0 else 0,
            line_colors="tab:orange",
            title=f"PyPSA-Libya — {stage_name}  (buses={n_buses}, lines={n_lines})",
        )
    except Exception as e:
        # Some stages (e.g. base.nc straight from OSM) may lack projection
        # info that n.plot() expects; fall back to a plain scatter so you
        # still get *something* to look at.
        print(f"    n.plot() failed ({e}); falling back to scatter")
        ax.clear()
        ax.scatter(n.buses.x, n.buses.y, s=20, c="tab:blue")
        for _, row in n.lines.iterrows():
            x0, y0 = n.buses.loc[row.bus0, ["x", "y"]]
            x1, y1 = n.buses.loc[row.bus1, ["x", "y"]]
            ax.plot([x0, x1], [y0, y1], c="tab:orange", lw=1)
        ax.set_title(f"PyPSA-Libya — {stage_name}  (buses={n_buses}, lines={n_lines})")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    out_path = os.path.join(OUT_DIR, f"network_topology_{stage_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved -> {out_path}")

# ---------------------------------------------------------------------
# PART 2 — Solved network: capacity results
# ---------------------------------------------------------------------

print("=" * 70)
print("PART 2: Solved network capacity results")
print("=" * 70)

SOLVED_PATH = "results/libya-test/networks/elec_s_8_ec_lcopt_Co2L-3h.nc"
n = pypsa.Network(SOLVED_PATH)

# --- 2a) Bar chart: total optimal capacity by carrier ---
cap_by_carrier = n.generators.groupby("carrier").p_nom_opt.sum().sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
colors = {
    "solar": "gold",
    "onwind": "steelblue",
    "CCGT": "dimgray",
    "oil": "saddlebrown",
    "load shedding": "crimson",
}
bar_colors = [colors.get(c, "gray") for c in cap_by_carrier.index]
cap_by_carrier.plot(kind="barh", ax=ax, color=bar_colors)
ax.set_xlabel("Optimal capacity (MW)")
ax.set_title("PyPSA-Libya — Optimal generation capacity by carrier\n(1-day test, 8 clusters — NOT a real annual result)")
for i, v in enumerate(cap_by_carrier.values):
    ax.text(v, i, f"  {v:,.0f} MW", va="center")
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "solved_capacity_by_carrier.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out_path}")

# --- 2b) Geographic map: bus markers sized by installed capacity ---
gen_by_bus = n.generators.groupby(["bus", "carrier"]).p_nom_opt.sum().unstack(fill_value=0)
total_cap_by_bus = gen_by_bus.sum(axis=1)
dominant_carrier_by_bus = gen_by_bus.idxmax(axis=1)

# only electricity buses (skip the paired "... H2" buses) for a cleaner map
elec_buses = [b for b in n.buses.index if not b.endswith("H2")]

fig, ax = plt.subplots(figsize=(8, 8))
for bus in elec_buses:
    x, y = n.buses.loc[bus, ["x", "y"]]
    cap = total_cap_by_bus.get(bus, 0)
    dom = dominant_carrier_by_bus.get(bus, "none")
    color = colors.get(dom, "gray")
    # size scaling: sqrt so huge/small caps don't dominate the plot visually
    size = max(30, (cap ** 0.5) * 15)
    ax.scatter(x, y, s=size, c=color, edgecolors="black", linewidths=0.5, zorder=3)
    ax.annotate(
        f"{bus}\n{cap:,.0f} MW",
        (x, y),
        textcoords="offset points",
        xytext=(6, 6),
        fontsize=7,
    )

# overlay lines if any exist (currently expected to be none — see Part 1)
for _, row in n.lines.iterrows():
    x0, y0 = n.buses.loc[row.bus0, ["x", "y"]]
    x1, y1 = n.buses.loc[row.bus1, ["x", "y"]]
    ax.plot([x0, x1], [y0, y1], c="black", lw=1, zorder=1, alpha=0.6)

ax.set_title(
    f"PyPSA-Libya — solved capacity by bus (lines shown: {len(n.lines)})\n"
    "Marker size = total capacity at bus; color = dominant carrier"
)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# simple legend
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=k)
    for k, c in colors.items()
]
ax.legend(handles=legend_elems, loc="lower left", fontsize=8)

out_path = os.path.join(OUT_DIR, "solved_capacity_map.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out_path}")

print("=" * 70)
print(f"Done. All images saved in ./{OUT_DIR}/")
print("=" * 70)
