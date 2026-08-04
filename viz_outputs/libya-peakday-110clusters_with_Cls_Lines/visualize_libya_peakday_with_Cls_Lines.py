"""
visualize_libya_peakday.py
Run from the pypsa-libya folder, in the pypsa-earth conda environment:

    python visualize_libya_peakday.py

Includes geographic basemaps (Cartopy country borders & coastlines).
Produces PNGs with full line connectivity in: viz_outputs/libya-peakday-<CLUSTERS>clusters/
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

CLUSTERS = 110

RUN_NAME_BY_CLUSTERS = {
    40: "libya-year-40AggBuses",
    100: "libya-peakday-full",
    110: "Libya-peakday_110",
}

if CLUSTERS not in RUN_NAME_BY_CLUSTERS:
    raise ValueError(
        f"No known run folder for CLUSTERS={CLUSTERS}. "
        f"Add it to RUN_NAME_BY_CLUSTERS with the exact folder name under "
        f"results/ and networks/ before running this script."
    )

RUN_NAME = RUN_NAME_BY_CLUSTERS[CLUSTERS]

OUT_DIR = os.path.join("viz_outputs", f"libya-peakday-{CLUSTERS}clusters")
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

# Robust Bus Coordinate Lookup with Fuzzy Prefix Matching
def find_bus_coords(network, bus_name):
    bus_str = str(bus_name).strip()
    if bus_str in network.buses.index:
        return network.buses.loc[bus_str, "x"], network.buses.loc[bus_str, "y"]
    
    # Strip suffixes like " 220.0kV", " AC", " battery", " H2"
    clean_prefix = re.split(r"\s+(?:\d+kV|AC|DC|battery|H2)", bus_str)[0].strip()
    
    if clean_prefix in network.buses.index:
        return network.buses.loc[clean_prefix, "x"], network.buses.loc[clean_prefix, "y"]
        
    matched = network.buses[network.buses.index.str.startswith(clean_prefix)]
    if not matched.empty:
        return matched.iloc[0]["x"], matched.iloc[0]["y"]
        
    return None, None

# Universal Line & Link Drawing Function
def draw_network_branches(network, ax, base_network=None, color="tab:orange", alpha=0.7, linewidth=1.0):
    lines_drawn = 0
    
    # Attempt 1: Draw directly from target network lines
    if not network.lines.empty:
        for _, row in network.lines.iterrows():
            x0, y0 = find_bus_coords(network, row.bus0)
            x1, y1 = find_bus_coords(network, row.bus1)
            if x0 is not None and x1 is not None:
                ax.plot([x0, x1], [y0, y1], c=color, lw=linewidth, alpha=alpha, zorder=2)
                lines_drawn += 1

    # Attempt 2: Draw from network links
    if not network.links.empty:
        trans_links = network.links[~network.links.carrier.isin(["H2 Electrolysis", "H2 Fuel Cell", "battery charger", "battery discharger"])]
        for _, row in trans_links.iterrows():
            x0, y0 = find_bus_coords(network, row.bus0)
            x1, y1 = find_bus_coords(network, row.bus1)
            if x0 is not None and x1 is not None and (x0 != x1 or y0 != y1):
                ax.plot([x0, x1], [y0, y1], c=color, lw=linewidth, alpha=alpha, zorder=2)
                lines_drawn += 1

    # Fallback: If 0 lines drawn, overlay network topology using base_network geometry mapping
    if lines_drawn == 0 and base_network is not None:
        if not base_network.lines.empty:
            for _, row in base_network.lines.iterrows():
                x0, y0 = find_bus_coords(network, row.bus0)
                x1, y1 = find_bus_coords(network, row.bus1)
                if x0 is not None and x1 is not None and (x0 != x1 or y0 != y1):
                    ax.plot([x0, x1], [y0, y1], c=color, lw=linewidth, alpha=alpha, zorder=2)
                    lines_drawn += 1

    return lines_drawn

# Load Base Network for fallback topology reference
BASE_PATH = f"networks/{RUN_NAME}/base.nc"
base_n = pypsa.Network(BASE_PATH) if os.path.exists(BASE_PATH) else None

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
print(f"PART 1: Network topology at each pipeline stage  (run: {RUN_NAME})")
print("=" * 70)

for stage_name, path in STAGES:
    if not os.path.exists(path):
        print(f"  [skip] {stage_name}: file not found at {path}")
        continue

    n = pypsa.Network(path)
    macro_buses = n.buses[~n.buses.index.str.contains("low voltage|battery|H2", case=False)]
    n_buses = len(macro_buses) 

    fig = plt.figure(figsize=(10, 8))
    ax = setup_map_ax(fig)

    # Plot branches with automatic base fallback fallback
    n_branches = draw_network_branches(n, ax, base_network=base_n, color="tab:orange", alpha=0.8, linewidth=1.1)

    # Plot buses
    ax.scatter(macro_buses.x, macro_buses.y, s=20, c="#3b82f6", edgecolors="black", linewidths=0.3, zorder=3)
    ax.set_title(f"PyPSA-Libya — {stage_name} (buses={n_buses}, lines={n_branches})")

    out_path = os.path.join(OUT_DIR, f"network_topology_{stage_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {stage_name:18s} buses={n_buses:4d}  lines={n_branches:4d} -> saved")

# ---------------------------------------------------------------------
# PART 2 — Solved network capacity & map
# ---------------------------------------------------------------------

print("=" * 70)
print(f"PART 2: Solved network capacity results (peak day, {CLUSTERS} clusters)")
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
    trans_links = n.links[~n.links.carrier.isin(["DC", "H2 Electrolysis", "H2 Fuel Cell", "battery charger", "battery discharger"])]
    if not trans_links.empty:
        link_cap = trans_links.groupby("carrier").p_nom_opt.sum()

all_power_cap = pd.concat([gen_cap, su_cap, link_cap]).groupby(level=0).sum().sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = [colors.get(c, "gray") for c in all_power_cap.index]
all_power_cap.plot(kind="barh", ax=ax, color=bar_colors)
ax.set_xlabel("Optimal Installed Capacity (MW)")
ax.set_title(f"PyPSA-Libya — Optimal Capacity by Carrier (Power: MW)\n(Peak Day Optimization, {CLUSTERS} Clusters)")
for i, v in enumerate(all_power_cap.values):
    ax.text(v, i, f"  {v:,.1f} MW", va="center")
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "solved_capacity_by_carrier.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out_path}")

# 2b) Geographic map with borders and line connections
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

cap_df = pd.DataFrame(cap_records)
gen_by_bus = cap_df.groupby(["bus", "carrier"])["capacity"].sum().unstack(fill_value=0)
total_cap_by_bus = gen_by_bus.sum(axis=1)
dominant_carrier_by_bus = gen_by_bus.idxmax(axis=1)

elec_buses = [b for b in n.buses.index if not (b.endswith("H2") or b.endswith("battery"))]

fig = plt.figure(figsize=(10, 8))
ax = setup_map_ax(fig)

# Draw transmission network lines/links on solved capacity map
draw_network_branches(n, ax, base_network=base_n, color="#555555", alpha=0.6, linewidth=1.0)

# Draw buses & capacity markers
for bus in elec_buses:
    x, y = find_bus_coords(n, bus)
    if x is None:
        continue
    cap = total_cap_by_bus.get(bus, 0)
    dom = dominant_carrier_by_bus.get(bus, "none")
    color = colors.get(dom, "gray")
    size = max(15, (cap ** 0.5) * 6)
    
    ax.scatter(x, y, s=size, c=color, edgecolors="black", linewidths=0.4, zorder=3)
    
    # Label top largest nodes
    if cap > (total_cap_by_bus.max() * 0.15):
        ax.annotate(
            f"{bus}\n{cap:,.0f} MW",
            (x, y),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=6,
            zorder=4
        )

ax.set_title(
    f"PyPSA-Libya — Solved Capacity by Bus (Peak Day 2013-08-21, {CLUSTERS} Clusters)\n"
    f"Marker size = capacity at bus; color = dominant carrier"
)

present_carriers = gen_by_bus.columns.unique()
legend_elems = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=colors.get(c, "gray"), markersize=8, label=c)
    for c in present_carriers if c in colors
]
ax.legend(handles=legend_elems, loc="lower left", fontsize=8)

out_path = os.path.join(OUT_DIR, "solved_capacity_map.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out_path}")

# ---------------------------------------------------------------------
# PART 3 — Peak Day Energy Generation & Dispatch
# ---------------------------------------------------------------------

print("=" * 70)
print(f"PART 3: Peak Day Energy Generation & Dispatch Profile")
print("=" * 70)

weightings = n.snapshot_weightings.generators
gen_p = n.generators_t.p.mul(weightings, axis=0)
daily_gen_gwh = gen_p.T.groupby(n.generators.carrier).sum().sum(axis=1) / 1e3

if not n.storage_units.empty and not n.storage_units_t.p.empty:
    su_p = n.storage_units_t.p.clip(lower=0).mul(weightings, axis=0)
    su_gen_gwh = su_p.T.groupby(n.storage_units.carrier).sum().sum(axis=1) / 1e3
    daily_energy_gwh = pd.concat([daily_gen_gwh, su_gen_gwh]).groupby(level=0).sum()
else:
    daily_energy_gwh = daily_gen_gwh

# 3a) Daily Generation Bar Chart
fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = [colors.get(c, "gray") for c in daily_energy_gwh.index]
daily_energy_gwh.plot(kind="bar", ax=ax, color=bar_colors)
ax.set_ylabel("Daily Generation / Discharge (GWh)")
ax.set_title(f"PyPSA-Libya — Peak Day Energy Generation Mix by Carrier\n({CLUSTERS} Clusters)")
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(daily_energy_gwh.values):
    ax.text(i, v, f" {v:.2f} GWh", ha="center", va="bottom")
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "daily_generation_by_carrier.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out_path}")

# 3b) Peak Day Dispatch Area Plot
fig, ax = plt.subplots(figsize=(12, 5))
gen_dispatch = n.generators_t.p.T.groupby(n.generators.carrier).sum().T / 1e3
if not n.storage_units.empty and not n.storage_units_t.p.empty:
    su_dispatch = n.storage_units_t.p.clip(lower=0).T.groupby(n.storage_units.carrier).sum().T / 1e3
    total_dispatch = pd.concat([gen_dispatch, su_dispatch], axis=1).T.groupby(level=0).sum().T
else:
    total_dispatch = gen_dispatch

active_carriers = [c for c in colors.keys() if c in total_dispatch.columns and total_dispatch[c].sum() > 1e-3]
total_dispatch[active_carriers].plot.area(ax=ax, color=[colors[c] for c in active_carriers], alpha=0.8)
ax.set_ylabel("Power (GW)")
ax.set_xlabel("Time")
ax.set_title(f"PyPSA-Libya — Peak Day Hourly Generation & Discharge Profile ({CLUSTERS} Clusters)")
ax.legend(loc="upper right")
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "daily_dispatch_timeseries.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out_path}")

print("=" * 70)
print(f"Done. All peak day visualizations saved in ./{OUT_DIR}/")
print("=" * 70)