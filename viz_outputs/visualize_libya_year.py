"""
visualize_libya_year.py
Run from the pypsa-libya folder, in the pypsa-earth conda environment:

    python visualize_libya_year.py

Includes geographic basemaps (Cartopy country borders & coastlines).
Produces PNGs in: viz_outputs/libya-year-<CLUSTERS>clusters/
"""

import os
import re
import pypsa
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Optional Cartopy import for high-quality country borders
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: cartopy not found. Falling back to simple coordinate plotting.")

# =====================================================================
# CONFIGURATION
# =====================================================================

CLUSTERS = 40   # Set cluster count (40, 100, etc.)

RUN_NAME_BY_CLUSTERS = {
    40: "libya-year-40AggBuses",
    100: "libya-year-full",
}

if CLUSTERS not in RUN_NAME_BY_CLUSTERS:
    raise ValueError(
        f"No known run folder for CLUSTERS={CLUSTERS}. "
        f"Add it to RUN_NAME_BY_CLUSTERS with the exact folder name under "
        f"results/ and networks/ before running this script."
    )

RUN_NAME = RUN_NAME_BY_CLUSTERS[CLUSTERS]

OUT_DIR = os.path.join("viz_outputs", f"libya-year-{CLUSTERS}clusters")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"CLUSTERS = {CLUSTERS}  ->  run folder = '{RUN_NAME}'  ->  output = '{OUT_DIR}'")

# Helper function to configure map basemap with borders
def setup_map_ax(fig):
    if HAS_CARTOPY:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent([9.0, 25.5, 19.0, 33.5], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#f4f4f0", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="#e0f2fe", zorder=0)
        ax.add_feature(cfeature.BORDERS, linestyle="--", edgecolor="#666666", linewidth=0.8, zorder=1)
        ax.add_feature(cfeature.COASTLINE, edgecolor="#333333", linewidth=1.0, zorder=1)
        
        # Gridlines with labels
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle=":")
        gl.top_labels = False
        gl.right_labels = False
        return ax
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(9.0, 25.5)
        ax.set_ylim(19.0, 33.5)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle=":", alpha=0.5)
        return ax

# ---------------------------------------------------------------------
# PART 1 — Topology at each pipeline stage
# ---------------------------------------------------------------------

STAGES = [
    ("base",                 f"networks/{RUN_NAME}/base.nc"),
    ("elec",                 f"networks/{RUN_NAME}/elec.nc"),
    ("elec_s",                f"networks/{RUN_NAME}/elec_s.nc"),
    (f"elec_s_{CLUSTERS}",    f"networks/{RUN_NAME}/elec_s_{CLUSTERS}.nc"),
    (f"elec_s_{CLUSTERS}_ec", f"networks/{RUN_NAME}/elec_s_{CLUSTERS}_ec.nc"),
]

print("=" * 70)
print(f"PART 1: Network topology at each pipeline stage (run: {RUN_NAME})")
print("=" * 70)

for stage_name, path in STAGES:
    if not os.path.exists(path):
        print(f"  [skip] {stage_name}: file not found at {path}")
        continue

    n = pypsa.Network(path)
    n_lines = len(n.lines)
    # n_buses = len(n.buses)
    
    # --- CHANGE THIS LINE ---
    # Filter out virtual low-voltage/distribution buses if they exist
    macro_buses = n.buses[~n.buses.index.str.contains("low voltage|battery|H2", case=False)]
    n_buses = len(macro_buses) 
    
    print(f"  {stage_name:18s} buses={n_buses:4d}  lines={n_lines:4d}")

    fig = plt.figure(figsize=(10, 8))
    ax = setup_map_ax(fig)

    try:
        n.plot(
            ax=ax,
            bus_sizes=0.01,
            line_widths=1.2 if n_lines > 0 else 0,
            line_colors="tab:orange",
            title=f"PyPSA-Libya — {stage_name} (buses={n_buses}, lines={n_lines})",
        )
    except Exception:
        ax.scatter(n.buses.x, n.buses.y, s=15, c="tab:blue", zorder=3)
        for _, row in n.lines.iterrows():
            x0, y0 = n.buses.loc[row.bus0, ["x", "y"]]
            x1, y1 = n.buses.loc[row.bus1, ["x", "y"]]
            ax.plot([x0, x1], [y0, y1], c="tab:orange", lw=1, zorder=2)
        ax.set_title(f"PyPSA-Libya — {stage_name} (buses={n_buses}, lines={n_lines})")

    out_path = os.path.join(OUT_DIR, f"network_topology_{stage_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved -> {out_path}")

# ---------------------------------------------------------------------
# PART 2 — Solved network capacity & map
# ---------------------------------------------------------------------

print("=" * 70)
print(f"PART 2: Solved network capacity results (Annual Model, {CLUSTERS} clusters)")
print("=" * 70)

SOLVED_PATH = f"results/{RUN_NAME}/networks/elec_s_{CLUSTERS}_ec_lcopt_Co2L-3h.nc"
if not os.path.exists(SOLVED_PATH):
    raise FileNotFoundError(f"Solved network not found at {SOLVED_PATH}")

n = pypsa.Network(SOLVED_PATH)

colors = {
    "solar": "gold",
    "onwind": "steelblue",
    "CCGT": "dimgray",
    "OCGT": "darkorange",
    "oil": "saddlebrown",
    "battery": "purple",
    "H2": "teal",
    "load shedding": "crimson",
}

# 2a) Bar chart
gen_cap = n.generators[n.generators.carrier != "load shedding"].groupby("carrier").p_nom_opt.sum()
su_cap = pd.Series(dtype=float)
if not n.storage_units.empty:
    su_cap = n.storage_units[n.storage_units.carrier != "load shedding"].groupby("carrier").p_nom_opt.sum()
link_cap = pd.Series(dtype=float)
if not n.links.empty:
    link_cap = n.links[n.links.carrier != "load shedding"].groupby("carrier").p_nom_opt.sum()

all_power_cap = pd.concat([gen_cap, su_cap, link_cap]).groupby(level=0).sum().sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = [colors.get(c, "gray") for c in all_power_cap.index]
all_power_cap.plot(kind="barh", ax=ax, color=bar_colors)
ax.set_xlabel("Optimal Installed Capacity (MW)")
ax.set_title(f"PyPSA-Libya — Optimal Capacity by Carrier (Power: MW)\n(Full Year Optimization, {CLUSTERS} Clusters)")
for i, v in enumerate(all_power_cap.values):
    ax.text(v, i, f"  {v:,.1f} MW", va="center")
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "solved_capacity_by_carrier.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out_path}")

# 2b) Geographic map with borders
def get_parent_bus(b):
    b_str = str(b)
    if "location" in n.buses.columns and b in n.buses.index:
        loc = n.buses.loc[b, "location"]
        if pd.notna(loc) and loc != "":
            return str(loc)
    return re.sub(r"\s+(battery|H2|H2 Electrolysis|H2 Fuel Cell)$", "", b_str)

cap_records = []
for _, row in n.generators[n.generators.carrier != "load shedding"].iterrows():
    cap_records.append({"bus": get_parent_bus(row.bus), "carrier": row.carrier, "capacity": row.p_nom_opt})
if not n.storage_units.empty:
    for _, row in n.storage_units[n.storage_units.carrier != "load shedding"].iterrows():
        cap_records.append({"bus": get_parent_bus(row.bus), "carrier": row.carrier, "capacity": row.p_nom_opt})
if not n.links.empty:
    for _, row in n.links[n.links.carrier != "load shedding"].iterrows():
        cap_records.append({"bus": get_parent_bus(row.bus0), "carrier": row.carrier, "capacity": row.p_nom_opt})

cap_df = pd.DataFrame(cap_records)
gen_by_bus = cap_df.groupby(["bus", "carrier"])["capacity"].sum().unstack(fill_value=0)
total_cap_by_bus = gen_by_bus.sum(axis=1)
dominant_carrier_by_bus = gen_by_bus.idxmax(axis=1)

elec_buses = [b for b in n.buses.index if not (b.endswith("H2") or b.endswith("battery"))]

fig = plt.figure(figsize=(10, 8))
ax = setup_map_ax(fig)

for _, row in n.lines.iterrows():
    if row.bus0 in n.buses.index and row.bus1 in n.buses.index:
        x0, y0 = n.buses.loc[row.bus0, ["x", "y"]]
        x1, y1 = n.buses.loc[row.bus1, ["x", "y"]]
        ax.plot([x0, x1], [y0, y1], c="black", lw=0.8, zorder=2, alpha=0.6)

for bus in elec_buses:
    x, y = n.buses.loc[bus, ["x", "y"]]
    cap = total_cap_by_bus.get(bus, 0)
    dom = dominant_carrier_by_bus.get(bus, "none")
    color = colors.get(dom, "gray")
    size = max(20, (cap ** 0.5) * 12)
    
    ax.scatter(x, y, s=size, c=color, edgecolors="black", linewidths=0.4, zorder=3)
    
    if cap > (total_cap_by_bus.max() * 0.05):
        ax.annotate(
            f"{bus}\n{cap:,.0f} MW",
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=6,
            zorder=4
        )

ax.set_title(
    f"PyPSA-Libya — Solved Capacity by Bus (Full Year Run, {CLUSTERS} Clusters)\n"
    f"Marker size = capacity at bus; color = dominant carrier"
)

present_carriers = gen_by_bus.columns.unique()
legend_elems = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=colors.get(c, "gray"), markersize=10, label=c)
    for c in present_carriers if c in colors
]
ax.legend(handles=legend_elems, loc="lower left", fontsize=8)

out_path = os.path.join(OUT_DIR, "solved_capacity_map.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out_path}")

# ---------------------------------------------------------------------
# PART 3 — Annual Energy Generation & Dispatch
# ---------------------------------------------------------------------

print("=" * 70)
print(f"PART 3: Annual Energy Generation & Dispatch Profile")
print("=" * 70)

weightings = n.snapshot_weightings.generators
gen_p = n.generators_t.p.mul(weightings, axis=0)
annual_gen_twh = gen_p.T.groupby(n.generators.carrier).sum().sum(axis=1) / 1e6

if not n.storage_units.empty and not n.storage_units_t.p.empty:
    su_p = n.storage_units_t.p.clip(lower=0).mul(weightings, axis=0)
    su_gen_twh = su_p.T.groupby(n.storage_units.carrier).sum().sum(axis=1) / 1e6
    annual_energy_twh = pd.concat([annual_gen_twh, su_gen_twh]).groupby(level=0).sum()
else:
    annual_energy_twh = annual_gen_twh

# 3a) Annual Generation Bar Chart
fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = [colors.get(c, "gray") for c in annual_energy_twh.index]
annual_energy_twh.plot(kind="bar", ax=ax, color=bar_colors)
ax.set_ylabel("Annual Generation / Discharge (TWh)")
ax.set_title(f"PyPSA-Libya — Total Annual Energy Mix by Carrier\n({CLUSTERS} Clusters)")
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(annual_energy_twh.values):
    ax.text(i, v, f" {v:.2f} TWh", ha="center", va="bottom")
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "annual_generation_by_carrier.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out_path}")

# 3b) Annual Dispatch Area Plot
fig, ax = plt.subplots(figsize=(14, 6))
gen_dispatch = n.generators_t.p.T.groupby(n.generators.carrier).sum().T / 1e3 # GW
if not n.storage_units.empty and not n.storage_units_t.p.empty:
    su_dispatch = n.storage_units_t.p.clip(lower=0).T.groupby(n.storage_units.carrier).sum().T / 1e3 # GW
    total_dispatch = pd.concat([gen_dispatch, su_dispatch], axis=1).T.groupby(level=0).sum().T
else:
    total_dispatch = gen_dispatch

active_carriers = [c for c in colors.keys() if c in total_dispatch.columns and total_dispatch[c].sum() > 1e-3]
total_dispatch[active_carriers].plot.area(ax=ax, color=[colors[c] for c in active_carriers], alpha=0.8)
ax.set_ylabel("Power (GW)")
ax.set_xlabel("Time")
ax.set_title(f"PyPSA-Libya — Full Year Hourly Generation & Battery Discharge ({CLUSTERS} Clusters)")
ax.legend(loc="upper right")
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "annual_dispatch_timeseries.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out_path}")

print("=" * 70)
print(f"Done. All annual visualizations saved in ./{OUT_DIR}/")
print("=" * 70)