"""
visualize_libya_year.py
Run from the pypsa-libya folder in the pypsa-earth conda environment:

    python visualize_libya_year.py

Generates PNG charts, text/CSV reports, dual/shadow prices, LCOE, renewable curtailment, 
and transmission bottleneck maps in: viz_outputs/libya-year-<CLUSTERS>clusters-<HOURS>h/
"""

import os
import re
import math
import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D


# =====================================================================
# RUN CONFIGURATION
# =====================================================================

CLUSTERS = 36  

HOURS = 6
# HOURS = 3

USE_CCL = False

RUN_NAME_BY_CLUSTERS = {
    100: "libya-year-full",
    40: "libya-year-40AggBuses",
    50: "libya-year-50AggBuses",
    36: "libya-year-36",
}

RUN_NAME = RUN_NAME_BY_CLUSTERS.get(CLUSTERS, f"libya-year-{CLUSTERS}clusters")
ccl_suffix = "_CCL" if USE_CCL else ""
OUT_DIR = os.path.join("viz_outputs", f"libya-year-{CLUSTERS}clusters-{HOURS}h{ccl_suffix}")
os.makedirs(OUT_DIR, exist_ok=True)



# Cartopy import for geographic basemaps
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: cartopy not found. Falling back to simple coordinate plotting.")

# =====================================================================
# COASTAL LINE ROUTING CONFIGURATION (LIBYA LAND BOUNDARY)
# =====================================================================

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
    """Returns the latitude of the Libyan coastline at a given longitude."""
    return np.interp(lon, COAST_LONS, COAST_LATS)

def route_line_land(x0, y0, x1, y1, land_offset=0.13, num_pts=50, allow_waypoint=True):
    """
    Routes transmission lines safely on land following coastal contours if 
    a straight Euclidean segment crosses water. Inland lines remain straight.
    """
    if x0 == x1 or (abs(x0 - x1) < 1e-5 and abs(y0 - y1) < 1e-5):
        return [x0, x1], [y0, y1]

    # Special waypoint routing for the Gulf of Sirte corner (Brega/Ajdabiya to Benghazi corridor)
    # where lines connect southwest (lon < 20.15, lat < 30.8) to north (lon 19.8-20.4, lat > 31.3)
    min_x, max_x = min(x0, x1), max(x0, x1)
    min_y, max_y = min(y0, y1), max(y0, y1)

    is_region2_crossing = (
        (min_x < 20.15 and max_x > 19.8) and
        (min_y < 30.8 and max_y > 31.3)
    )

    if is_region2_crossing and allow_waypoint:
        x_mid, y_mid = 20.32, 30.70  # Inland waypoint east of Ajdabiya
        xs1, ys1 = route_line_land(x0, y0, x_mid, y_mid, land_offset=land_offset, num_pts=num_pts//2, allow_waypoint=False)
        xs2, ys2 = route_line_land(x_mid, y_mid, x1, y1, land_offset=land_offset, num_pts=num_pts//2, allow_waypoint=False)
        return np.concatenate([xs1, xs2]), np.concatenate([ys1, ys2])

    xs = np.linspace(x0, x1, num_pts)
    ys_linear = np.linspace(y0, y1, num_pts)
    coast_lats = get_coast_lat(xs) - land_offset
    
    # Clamp y-coordinates to stay on land (south of coastal water)
    ys_land = np.minimum(ys_linear, coast_lats)
    ys_land[0] = y0
    ys_land[-1] = y1
    
    # Smooth line path if it was adjusted landward
    if num_pts > 5 and np.any(ys_linear > coast_lats):
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        ys_smooth = np.convolve(ys_land, kernel, mode='same')
        ys_smooth[0] = y0
        ys_smooth[1] = ys_land[1]
        ys_smooth[-2] = ys_land[-2]
        ys_smooth[-1] = y1
        ys_land = np.minimum(ys_smooth, coast_lats)
        ys_land[0] = y0
        ys_land[-1] = y1

    return xs, ys_land



print("=" * 70)
print(f"RUN CONFIGURATION: Clusters={CLUSTERS}, Resolution={HOURS}h, USE_CCL={USE_CCL}")
print(f"Target Directory: '{RUN_NAME}'  -->  Outputs: '{OUT_DIR}'")
print("=" * 70)

# Carrier & Storage operation color mapping
colors = {
    "solar": "gold",
    "onwind": "steelblue",
    "offwind": "deepskyblue",
    "CCGT": "dimgray",
    "OCGT": "darkorange",
    "oil": "saddlebrown",
    "battery": "purple",
    "battery (discharge)": "purple",
    "battery (charge)": "mediumpurple",
    "H2": "teal",
    "H2 (discharge)": "teal",
    "H2 (charge)": "cadetblue",
    "load shedding": "crimson",
}

# Reference Libyan geographic landmarks for landmark lookup
LIBYAN_LANDMARKS = [
    ("Tripoli", 13.1913, 32.8872),
    ("Benghazi", 20.0667, 32.1167),
    ("Misrata", 15.0906, 32.3754),
    ("Zawiya", 12.7278, 32.7569),
    ("Bayda", 21.7550, 32.7628),
    ("Tobruk", 23.9600, 32.0836),
    ("Sabha", 14.4283, 27.0377),
    ("Sirte", 16.5889, 31.2089),
    ("Ajdabiya", 20.2231, 30.7554),
    ("Derna", 22.6367, 32.7670),
    ("Zliten", 14.5687, 32.4674),
    ("Khoms", 14.2619, 32.6486),
    ("Zwara", 12.0819, 32.9312),
    ("Gharyan", 13.0206, 32.1722),
    ("Tarhuna", 13.6333, 32.4333),
    ("Bani Walid", 13.9833, 31.7583),
    ("Hun (Jufra)", 15.9472, 29.1268),
    ("Kufra", 23.2917, 24.1833),
    ("Ubari", 12.7806, 26.5917),
    ("Murzuq", 13.9183, 25.9156),
    ("Ghadames", 9.4969, 30.1336),
    ("Nalut", 10.9833, 31.8667),
    ("Marj", 20.8297, 32.4864),
    ("Brega", 19.5739, 30.4062),
    ("Ras Lanuf", 18.5603, 30.5089),
    ("Shatti", 14.1500, 27.5000),
    ("Bardia", 25.0861, 31.7561)
]

def map_carrier_name(carrier_str):
    c = str(carrier_str).strip()
    if "battery" in c.lower():
        return "battery"
    elif "h2" in c.lower() or "hydrogen" in c.lower():
        return "H2"
    elif "wind" in c.lower():
        return "onwind"
    return c

def shorten_line_id(line_idx, bus0, bus1):
    idx_str = str(line_idx).strip()
    if "lines new" in idx_str or "new" in idx_str.lower() or idx_str.startswith("N"):
        def find_isolated_node(text):
            m = re.search(r'LY\s*([1-9]\d*)\s*(\d*)', str(text))
            if m:
                sub_id, node_id = m.group(1), m.group(2)
                return f"NLY{sub_id} {node_id}".strip() if node_id else f"NLY{sub_id}"
            return None

        iso_id = find_isolated_node(bus1) or find_isolated_node(bus0) or find_isolated_node(idx_str)
        if iso_id:
            return iso_id

        clean_idx = idx_str.replace("lines new ", "NL_").replace(" <-> ", "_").replace(" AC", "")
        clean_idx = re.sub(r'\s+', '', clean_idx)
        return clean_idx[:10]
    return idx_str

def setup_map_ax(fig):
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

def find_bus_coords(network, bus_name):
    bus_str = str(bus_name).strip()
    if bus_str in network.buses.index:
        return network.buses.loc[bus_str, "x"], network.buses.loc[bus_str, "y"]
    clean_prefix = re.split(r"\s+(?:\d+kV|AC|DC|battery|H2)", bus_str)[0].strip()
    if clean_prefix in network.buses.index:
        return network.buses.loc[clean_prefix, "x"], network.buses.loc[clean_prefix, "y"]
    matched = network.buses[network.buses.index.str.startswith(clean_prefix)]
    if not matched.empty:
        return matched.iloc[0]["x"], matched.iloc[0]["y"]
    return None, None

def clean_region_string(region_str):
    if not region_str:
        return "Unknown"
    cleaned = re.sub(r"\b(Area|Region|Near)\b", "", str(region_str), flags=re.IGNORECASE)
    cleaned = cleaned.replace("/ LANDMARK", "").strip()
    return cleaned if cleaned else str(region_str).strip()

def reverse_geocode_landmark(lon, lat):
    if lon is None or lat is None or pd.isna(lon) or pd.isna(lat):
        return "Unknown"
    
    best_landmark = "Unknown"
    min_dist = float("inf")
    for name, ref_lon, ref_lat in LIBYAN_LANDMARKS:
        dist = math.hypot(lon - ref_lon, lat - ref_lat)
        if dist < min_dist:
            min_dist = dist
            best_landmark = name
    
    return best_landmark

def get_bus_info(network, bus_name):
    bus_str = str(bus_name).strip()
    x, y = find_bus_coords(network, bus_str)
    clean_prefix = re.split(r"\s+(?:\d+kV|AC|DC|battery|H2)", bus_str)[0].strip()
    
    region_name = None
    if clean_prefix in network.buses.index:
        bus_row = network.buses.loc[clean_prefix]
        if "location" in network.buses.columns and pd.notna(bus_row.get("location")) and str(bus_row["location"]).strip() and not str(bus_row["location"]).startswith("LY"):
            region_name = str(bus_row["location"]).strip()
            
    if not region_name and x is not None and y is not None:
        region_name = reverse_geocode_landmark(x, y)
    elif not region_name:
        region_name = clean_prefix

    return x, y, clean_region_string(region_name)

def get_safe_offset(x, y, base_offset=(12, 12)):
    dx, dy = base_offset
    if y is not None and y > 32.5:
        dy = -abs(dy) - 6
    if x is not None and x > 24.0:
        dx = -abs(dx) - 12
    if x is not None and x < 10.5:
        dx = abs(dx) + 6
    return (dx, dy)

def draw_base_lines(base_network, ax, color="tab:orange", alpha=0.75, linewidth=1.0):
    lines_drawn = 0
    if base_network is None or base_network.lines.empty:
        return lines_drawn
    for _, row in base_network.lines.iterrows():
        x0, y0 = find_bus_coords(base_network, row.bus0)
        x1, y1 = find_bus_coords(base_network, row.bus1)
        if x0 is not None and x1 is not None and (x0 != x1 or y0 != y1):
            xs, ys = route_line_land(x0, y0, x1, y1)
            ax.plot(xs, ys, c=color, lw=linewidth, alpha=alpha, zorder=2)
            lines_drawn += 1
    return lines_drawn

BASE_PATH = f"networks/{RUN_NAME}/base.nc"
ELEC_PATH = f"networks/{RUN_NAME}/elec.nc"
base_n = pypsa.Network(BASE_PATH) if os.path.exists(BASE_PATH) else (pypsa.Network(ELEC_PATH) if os.path.exists(ELEC_PATH) else None)

# ---------------------------------------------------------------------
# PART 1 — Topology at Pipeline Stages
# ---------------------------------------------------------------------

STAGES = [
    ("base",                 f"networks/{RUN_NAME}/base.nc"),
    ("elec",                 f"networks/{RUN_NAME}/elec.nc"),
    ("elec_s",                f"networks/{RUN_NAME}/elec_s.nc"),
    (f"elec_s_{CLUSTERS}",    f"networks/{RUN_NAME}/elec_s_{CLUSTERS}.nc"),
    (f"elec_s_{CLUSTERS}_ec", f"networks/{RUN_NAME}/elec_s_{CLUSTERS}_ec.nc"),
]

print("\nPART 1: Network topology at pipeline stages")
print("-" * 70)

for stage_name, path in STAGES:
    if not os.path.exists(path):
        print(f"  [skip] {stage_name}: file not found at {path}")
        continue
    n_stage = pypsa.Network(path)
    macro_buses = n_stage.buses[~n_stage.buses.index.str.contains("low voltage|battery|H2", case=False)]
    
    fig = plt.figure(figsize=(10, 8))
    ax = setup_map_ax(fig)
    n_branches = draw_base_lines(base_n if base_n is not None else n_stage, ax, color="tab:orange", alpha=0.8, linewidth=1.1)
    ax.scatter(macro_buses.x, macro_buses.y, s=20, c="#3b82f6", edgecolors="black", linewidths=0.3, zorder=3)
    ax.set_title(f"PyPSA-Libya — {stage_name} (buses={len(macro_buses)}, lines={n_branches})")
    out_path = os.path.join(OUT_DIR, f"network_topology_{stage_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {stage_name:18s} buses={len(macro_buses):4d}  lines={n_branches:4d} -> saved")

# ---------------------------------------------------------------------
# PART 2 — Solved Network Capacity & Map
# ---------------------------------------------------------------------

print("\nPART 2: Solved network capacity results")
print("-" * 70)

opts_str = f"CCL-Co2L-{HOURS}h" if USE_CCL else f"Co2L-{HOURS}h"
possible_solved_paths = [
    f"Output/{RUN_NAME}/networks/elec_s_{CLUSTERS}_ec_lcopt_{opts_str}.nc",
    f"Output/{RUN_NAME}/networks/elec_s_{CLUSTERS}_ec_lcopt_Co2L-{HOURS}h.nc",
]

SOLVED_PATH = next((p for p in possible_solved_paths if os.path.exists(p)), None)
if SOLVED_PATH is None:
    raise FileNotFoundError(f"Could not locate solved network file in Output/{RUN_NAME}/networks/")

print(f"Loading solved network from: {SOLVED_PATH}")
n = pypsa.Network(SOLVED_PATH)

dt_hours = (n.snapshots[1] - n.snapshots[0]).total_seconds() / 3600.0 if len(n.snapshots) > 1 and isinstance(n.snapshots, pd.DatetimeIndex) else float(HOURS)
total_physical_hours = len(n.snapshots) * dt_hours
duration_label = f"Full Year ({total_physical_hours:.0f} Hours)"

def get_parent_bus(b):
    if b in n.buses.index and "location" in n.buses.columns:
        loc = n.buses.loc[b, "location"]
        if pd.notna(loc) and str(loc).strip() != "":
            return str(loc).strip()
    return re.split(r"\s+(?:\d+kV|AC|DC|battery|H2)", str(b))[0].strip()

def get_clean_power_capacity_df(network):
    records = []
    if not network.generators.empty:
        gen_df = network.generators[network.generators.carrier != "load shedding"].copy()
        cap = gen_df["p_nom_opt"].fillna(gen_df.get("p_nom", 0)) if "p_nom_opt" in gen_df.columns else gen_df.get("p_nom", 0)
        for idx, row in gen_df.iterrows():
            if cap.loc[idx] > 1e-3:
                records.append({"bus": get_parent_bus(row.bus), "carrier": map_carrier_name(row.carrier), "capacity": cap.loc[idx]})
    
    if not network.storage_units.empty:
        su_df = network.storage_units.copy()
        cap = su_df["p_nom_opt"].fillna(su_df.get("p_nom", 0)) if "p_nom_opt" in su_df.columns else su_df.get("p_nom", 0)
        for idx, row in su_df.iterrows():
            if cap.loc[idx] > 1e-3:
                records.append({"bus": get_parent_bus(row.bus), "carrier": map_carrier_name(row.carrier), "capacity": cap.loc[idx]})

    return pd.DataFrame(records)

cap_df = get_clean_power_capacity_df(n)

if not cap_df.empty:
    all_power_cap = cap_df.groupby("carrier")["capacity"].sum().sort_values()
    all_power_cap = all_power_cap[all_power_cap > 1e-2]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(all_power_cap.index, all_power_cap.values, color=[colors.get(c, "gray") for c in all_power_cap.index])
    ax.set_xlabel("Optimal Installed Power Capacity (MW)")
    ax.set_title("PyPSA-Libya — Optimal Power Capacity by Carrier (MW)")
    
    ax.bar_label(bars, fmt='{:,.1f} MW', padding=5, fontsize=9, fontweight='normal')
    ax.margins(x=0.20)
    
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cbd5e1')
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "solved_capacity_by_carrier.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

# Solved Capacity Map
fig = plt.figure(figsize=(11, 8))
ax = setup_map_ax(fig)
draw_base_lines(base_n if base_n is not None else n, ax, color="#888888", alpha=0.5, linewidth=0.9)

macro_buses = n.buses[~n.buses.index.str.contains("low voltage|battery|H2", case=False)]

if not cap_df.empty:
    gen_by_bus = cap_df.groupby(["bus", "carrier"])["capacity"].sum().unstack(fill_value=0)
    total_cap_by_bus = gen_by_bus.sum(axis=1)
    dominant_carrier_by_bus = gen_by_bus.idxmax(axis=1)
    for bus_id, cap in total_cap_by_bus.items():
        if cap <= 0.1: continue
        x, y = find_bus_coords(n, bus_id)
        if x is None: continue
        color = colors.get(dominant_carrier_by_bus.get(bus_id, "none"), "gray")
        ax.scatter(x, y, s=max(20, (cap ** 0.5) * 7), c=color, edgecolors="black", linewidths=0.5, zorder=4, alpha=0.85)

stagger_offsets = [
    (8, 8), (-12, 10), (10, -12), (-12, -10),
    (12, 0), (-14, 2), (0, 12), (0, -14),
    (10, -6), (-10, -12)
]

for idx, (bus_id, bus_row) in enumerate(macro_buses.iterrows()):
    x, y = bus_row.x, bus_row.y
    if x is not None and y is not None:
        match = re.search(r"LY\d+", str(bus_id))
        clean_bus_label = match.group(0) if match else str(bus_id).split()[0]
        
        raw_offset = stagger_offsets[idx % len(stagger_offsets)]
        safe_offset = get_safe_offset(x, y, raw_offset)
        
        ax.annotate(
            clean_bus_label,
            (x, y),
            xytext=safe_offset,
            textcoords="offset points",
            fontsize=7.5,
            fontweight="normal",
            color="black",
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.6, alpha=0.75),
            zorder=6
        )

ax.set_title(f"PyPSA-Libya — Solved Power Capacity by Bus ({CLUSTERS} Clusters, {HOURS}h)")
fig.savefig(os.path.join(OUT_DIR, "solved_capacity_map.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------
# PART 3 — Energy Generation Mix & Renewable Performance (CF & Curtailment)
# ---------------------------------------------------------------------

print("\nPART 3: Energy Generation, Dispatch Profile & Renewable CF/Curtailment")
print("-" * 70)

phys_gen_dict = {}

# 1. Generator Energy Production
if not n.generators.empty:
    gen_p = n.generators_t.p.groupby(n.generators.carrier.map(map_carrier_name), axis=1).sum()
    for carrier in gen_p.columns:
        energy_gwh = (gen_p[carrier] * dt_hours).sum() / 1e3
        if energy_gwh > 1e-3:
            phys_gen_dict[carrier] = energy_gwh

# 2. Storage Discharging (+) and Charging (-) Energy Separated
if not n.storage_units.empty and not n.storage_units_t.p.empty:
    su_p = n.storage_units_t.p
    su_carrier_mapped = n.storage_units.carrier.map(map_carrier_name)
    for carrier in su_carrier_mapped.unique():
        cols = n.storage_units[su_carrier_mapped == carrier].index
        p_carrier = su_p[cols]

        dis_gwh = (p_carrier.clip(lower=0).sum(axis=1) * dt_hours).sum() / 1e3
        if dis_gwh > 1e-3:
            phys_gen_dict[f"{carrier} (discharge)"] = dis_gwh

        chg_gwh = ((-p_carrier.clip(upper=0)).sum(axis=1) * dt_hours).sum() / 1e3
        if chg_gwh > 1e-3:
            phys_gen_dict[f"{carrier} (charge)"] = chg_gwh

phys_gen_gwh = pd.Series(phys_gen_dict)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(phys_gen_gwh.index, phys_gen_gwh.values, color=[colors.get(c, "gray") for c in phys_gen_gwh.index])
ax.set_ylabel("Physical Generation / Activity (GWh)")
ax.set_title("PyPSA-Libya — Physical Energy Generation & Storage Activity")
plt.xticks(rotation=30, ha='right')

ax.bar_label(bars, fmt='{:,.1f}', padding=4, fontsize=9, fontweight='normal')
ax.margins(y=0.18)

ax.set_axisbelow(True)
ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cbd5e1')

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "physical_generation_by_carrier.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# Total hourly dispatch dataframe creation
gen_dispatch = n.generators_t.p.T.groupby(n.generators.carrier.map(map_carrier_name)).sum().T / 1e3
if not n.storage_units.empty and not n.storage_units_t.p.empty:
    su_dispatch = n.storage_units_t.p.clip(lower=0).T.groupby(n.storage_units.carrier.map(map_carrier_name)).sum().T / 1e3
    total_dispatch = pd.concat([gen_dispatch, su_dispatch], axis=1).T.groupby(level=0).sum().T
else:
    total_dispatch = gen_dispatch

active_carriers = [c for c in colors.keys() if c in total_dispatch.columns and total_dispatch[c].sum() > 1e-3]

# Annual Time Series Area Plot
if active_carriers:
    fig, ax = plt.subplots(figsize=(14, 6))
    
    color_list = [colors.get(c, "gray") for c in active_carriers]
    series_list = [total_dispatch[c].values for c in active_carriers]
    
    ax.stackplot(
        total_dispatch.index,
        *series_list,
        labels=active_carriers,
        colors=color_list,
        alpha=0.85
    )
    
    ax.set_ylabel("Power (GW)")
    ax.set_xlabel("Time")
    ax.set_title(f"PyPSA-Libya — Full Year Hourly Generation & Dispatch Profile ({CLUSTERS} Clusters, {HOURS}h" + (" + CCL)" if USE_CCL else ")"))
    ax.legend(loc="upper right")

    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cbd5e1')
        
    fig.tight_layout()
    out_path_ts = os.path.join(OUT_DIR, "annual_dispatch_timeseries.png")
    fig.savefig(out_path_ts, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved annual dispatch profile -> {out_path_ts}")

# Monthly Slice Stacked Bar Plot (1 Aug - 1 Sep)
year = n.snapshots[0].year if isinstance(n.snapshots, pd.DatetimeIndex) else 2013
start_date = f"{year}-08-01"
end_date = f"{year}-09-01"

dispatch_aug = total_dispatch.loc[start_date:end_date]

if not dispatch_aug.empty and active_carriers:
    fig, ax = plt.subplots(figsize=(14, 6))
    
    color_list = [colors.get(c, "gray") for c in active_carriers]
    dispatch_aug[active_carriers].plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=color_list,
        width=0.85,
        alpha=0.9
    )
    
    ax.set_ylabel("Power (GW)")
    ax.set_xlabel("Time (August)")
    ax.set_title(f"PyPSA-Libya — Hourly Generation & Dispatch Profile ({start_date} to {end_date})")
    ax.legend(loc="upper right")

    n_ticks = max(1, len(dispatch_aug) // 8)
    ticks = range(0, len(dispatch_aug), n_ticks)
    labels = [dispatch_aug.index[i].strftime("%d %b %H:%M") for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=30, ha="right")

    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cbd5e1')

    fig.tight_layout()
    out_path_aug = os.path.join(OUT_DIR, "monthly_dispatch_august_bar.png")
    fig.savefig(out_path_aug, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved August bar chart -> {out_path_aug}")

# ---------------------------------------------------------------------
# RENEWABLE CAPACITY FACTOR & CURTAILMENT ANALYSIS (Solar & Wind)
# ---------------------------------------------------------------------

renewable_carriers = [c for c in ["solar", "onwind", "offwind"] if c in n.generators.carrier.unique()]
cf_curtailment_results = []

for carrier in renewable_carriers:
    gens = n.generators[n.generators.carrier == carrier]
    if gens.empty:
        continue
    
    p_nom = gens["p_nom_opt"].fillna(gens.get("p_nom", 0)) if "p_nom_opt" in gens.columns else gens.get("p_nom", 0)
    total_cap_mw = p_nom.sum()
    
    actual_p = n.generators_t.p[gens.index]
    actual_gen_gwh = (actual_p.sum().sum() * dt_hours) / 1e3
    
    if carrier in n.generators_t.p_max_pu.columns or any(idx in n.generators_t.p_max_pu.columns for idx in gens.index):
        cols_in_pu = [col for col in gens.index if col in n.generators_t.p_max_pu.columns]
        if cols_in_pu:
            potential_p = n.generators_t.p_max_pu[cols_in_pu] * p_nom.loc[cols_in_pu]
            potential_gen_gwh = (potential_p.sum().sum() * dt_hours) / 1e3
        else:
            potential_gen_gwh = actual_gen_gwh
    else:
        potential_gen_gwh = actual_gen_gwh
        
    curtailed_gwh = max(0.0, potential_gen_gwh - actual_gen_gwh)
    curtailment_rate_pct = (curtailed_gwh / potential_gen_gwh * 100.0) if potential_gen_gwh > 0 else 0.0
    
    max_possible_gen_gwh = (total_cap_mw * total_physical_hours) / 1e3
    capacity_factor_pct = (actual_gen_gwh / max_possible_gen_gwh * 100.0) if max_possible_gen_gwh > 0 else 0.0
    
    cf_curtailment_results.append({
        "carrier": carrier,
        "capacity_mw": total_cap_mw,
        "potential_gwh": potential_gen_gwh,
        "actual_gwh": actual_gen_gwh,
        "curtailed_gwh": curtailed_gwh,
        "capacity_factor_pct": capacity_factor_pct,
        "curtailment_rate_pct": curtailment_rate_pct
    })

df_renewables = pd.DataFrame(cf_curtailment_results)

# 1. Renewable Text Report
ren_txt_path = os.path.join(OUT_DIR, "renewable_cf_and_curtailment.txt")
with open(ren_txt_path, "w", encoding="utf-8") as f:
    f.write("=" * 95 + "\n")
    f.write(f" PyPSA-Libya — Renewable Capacity Factor (CF) & Curtailment Report ({duration_label})\n")
    f.write("=" * 95 + "\n\n")
    f.write(f"{'CARRIER':10s} | {'CAPACITY (MW)':14s} | {'POTENTIAL (GWh)':16s} | {'ACTUAL (GWh)':14s} | {'CURTAILED (GWh)':16s} | {'CF (%)':8s} | {'CURTAIL %':10s}\n")
    f.write("-" * 95 + "\n")
    for _, row in df_renewables.iterrows():
        f.write(f"{row['carrier']:10s} | {row['capacity_mw']:14,.1f} | {row['potential_gwh']:16,.1f} | {row['actual_gwh']:14,.1f} | {row['curtailed_gwh']:16,.1f} | {row['capacity_factor_pct']:7.2f}% | {row['curtailment_rate_pct']:9.2f}%\n")
    f.write("-" * 95 + "\n")

print(f"  saved Renewable CF & Curtailment report -> {ren_txt_path}")

# 2. Renewable Performance Graph
if not df_renewables.empty:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    bars1 = ax1.bar(df_renewables["carrier"], df_renewables["capacity_factor_pct"], color=[colors.get(c, "gold") for c in df_renewables["carrier"]], width=0.5)
    ax1.set_ylabel("Capacity Factor (%)")
    ax1.set_title("Capacity Factor (CF) by Carrier")
    ax1.bar_label(bars1, fmt='%.2f%%', padding=3, fontsize=9)
    ax1.margins(y=0.2)
    ax1.set_axisbelow(True)
    ax1.grid(axis='y', linestyle='--', alpha=0.5, color='#cbd5e1')

    bars2 = ax2.bar(df_renewables["carrier"], df_renewables["curtailment_rate_pct"], color="#ef4444", width=0.5)
    ax2.set_ylabel("Curtailment Rate (%)")
    ax2.set_title("Curtailment Rate (%) by Carrier")
    ax2.bar_label(bars2, fmt='%.2f%%', padding=3, fontsize=9)
    ax2.margins(y=0.2)
    ax2.set_axisbelow(True)
    ax2.grid(axis='y', linestyle='--', alpha=0.5, color='#cbd5e1')

    fig.suptitle("PyPSA-Libya — Renewable Energy Performance & Curtailment", fontsize=12, fontweight='bold')
    fig.tight_layout()
    ren_png_path = os.path.join(OUT_DIR, "renewable_cf_and_curtailment.png")
    fig.savefig(ren_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved Renewable CF & Curtailment chart -> {ren_png_path}")

# ---------------------------------------------------------------------
# PART 4 — Dual Variables & Transmission Bottleneck Analysis
# ---------------------------------------------------------------------

print("\nPART 4: Shadow Prices & Transmission Bottleneck Analysis")
print("-" * 70)

if hasattr(n, "buses_t") and not n.buses_t.marginal_price.empty:
    mean_lmp = n.buses_t.marginal_price.mean(axis=0).sort_values(ascending=False)
    
    lmp_txt_path = os.path.join(OUT_DIR, "locational_marginal_prices.txt")
    with open(lmp_txt_path, "w", encoding="utf-8") as f:
        f.write("-" * 80 + "\n")
        f.write(f"{'BUS / LOCATION':22s} | {'MEAN MARGINAL PRICE (€/MWh)':28s} | {'LONGITUDE':10s} | {'LATITUDE':10s} | {'REGION'}\n")
        f.write("-" * 80 + "\n")
        for bus, price in mean_lmp.items():
            x, y, landmark = get_bus_info(n, bus)
            lon_str = f"{x:.4f}" if x is not None else "N/A"
            lat_str = f"{y:.4f}" if y is not None else "N/A"
            f.write(f"{str(bus):22s} | €{price:12,.2f}/MWh            | {lon_str:10s} | {lat_str:10s} | {landmark}\n")
            
    print(f"  saved enhanced LMP report      -> {lmp_txt_path}")
    
    top_15_lmp = mean_lmp.head(15)
    fig, ax = plt.subplots(figsize=(9, 5))
    top_15_lmp.plot(kind="bar", ax=ax, color="#ef4444")
    ax.set_ylabel("Average Marginal Price (€/MWh)")
    ax.set_title("PyPSA-Libya — Top 15 Highest Cost Nodes (LMP)")
    plt.xticks(rotation=45, ha='right')
    
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cbd5e1')
    
    fig.tight_layout()
    out_lmp_png = os.path.join(OUT_DIR, "lmp_by_bus.png")
    fig.savefig(out_lmp_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved LMP chart                -> {out_lmp_png}")

    top_10_lmp = mean_lmp.head(10)
    fig = plt.figure(figsize=(11, 8))
    ax = setup_map_ax(fig)
    draw_base_lines(base_n if base_n is not None else n, ax, color="#cbd5e1", alpha=0.5, linewidth=0.8)

    min_lmp_val = top_10_lmp.min()
    max_lmp_val = top_10_lmp.max()
    norm_lmp = mcolors.Normalize(vmin=min_lmp_val, vmax=max_lmp_val)
    cmap_lmp = cm.get_cmap("YlOrRd")

    node_offsets = [
        (12, 12), (-18, 14), (14, -16), (-18, -14), 
        (18, 0), (-20, 2), (0, 16), (0, -18), 
        (16, -10), (-16, -16)
    ]

    for idx, (bus_id, price) in enumerate(top_10_lmp.items()):
        x, y = find_bus_coords(n, bus_id)
        if x is not None and y is not None:
            color = cmap_lmp(norm_lmp(price))
            ax.scatter(x, y, s=35, color=color, edgecolors="black", linewidths=0.8, zorder=5)
            
            match = re.search(r"LY\d+", str(bus_id))
            clean_bus_label = match.group(0) if match else str(bus_id).split()[0]
            
            raw_offset = node_offsets[idx % len(node_offsets)]
            safe_offset = get_safe_offset(x, y, raw_offset)
            
            ax.annotate(
                clean_bus_label,
                (x, y),
                xytext=safe_offset,
                textcoords="offset points",
                fontsize=8.0,
                fontweight="normal",
                color="black",
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.6, alpha=0.8),
                zorder=6
            )

    sm_lmp = cm.ScalarMappable(cmap=cmap_lmp, norm=norm_lmp)
    sm_lmp.set_array([])
    cbar_lmp = fig.colorbar(sm_lmp, ax=ax, orientation="vertical", shrink=0.7, pad=0.03)
    cbar_lmp.set_label("Average LMP (€/MWh)", fontsize=10, fontweight="normal")

    ax.set_title("PyPSA-Libya — Spatial Distribution of Top 10 Highest LMP Nodes")
    out_lmp_map_png = os.path.join(OUT_DIR, "lmp_locations_map.png")
    fig.savefig(out_lmp_map_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved LMP locations map        -> {out_lmp_map_png}")

else:
    print("  [skip] No buses_t.marginal_price data found in solved network.")

# Carbon Shadow Price
co2_txt_path = os.path.join(OUT_DIR, "carbon_shadow_price.txt")
with open(co2_txt_path, "w", encoding="utf-8") as f:
    f.write("=" * 65 + "\n")
    f.write(f" PyPSA-Libya — CO2 Constraint & Shadow Price Report ({duration_label})\n")
    f.write("=" * 65 + "\n\n")
    
    if hasattr(n, "global_constraints") and not n.global_constraints.empty:
        if "CO2Limit" in n.global_constraints.index:
            co2_mu = n.global_constraints.loc["CO2Limit", "mu"]
            limit_val = n.global_constraints.loc["CO2Limit", "constant"]
            f.write(f"Target CO2 Limit Cap         : {limit_val:,.2f} Tonnes\n")
            f.write(f"CO2 Dual Value (mu)          : {co2_mu:,.4f}\n\n")
            f.write("-" * 65 + "\n")
            f.write(f"EQUIVALENT CARBON TAX VALUE  : €{abs(co2_mu):,.2f} / Tonne CO2\n")
            f.write("-" * 65 + "\n\n")
            if abs(co2_mu) < 1e-3:
                f.write("Interpretation: The CO2 cap was non-binding (mu ~ 0). The system\n")
                f.write("                naturally achieved the emissions target without extra cost.\n")
            else:
                f.write("Interpretation: The CO2 cap was active and binding. This value represents\n")
                f.write("                the exact marginal cost of eliminating the final ton of CO2.\n")
        else:
            f.write("CO2Limit row not found in global_constraints table.\n")
    else:
        f.write("No global_constraints stored in the solved network.\n")
print(f"  saved Carbon Shadow Price      -> {co2_txt_path}")

# Transmission Line Bottlenecks & Flow Analysis
has_mu = hasattr(n.lines_t, "mu_upper") and not n.lines_t.mu_upper.empty and (n.lines_t.mu_upper.abs().sum().sum() > 0)

flows = n.lines_t.p0.abs()
caps = n.lines["s_nom_opt"] if "s_nom_opt" in n.lines.columns else n.lines["s_nom"]
caps = caps.replace(0, 1e-6)
loading_ratios = flows.div(caps, axis=1) * 100.0

short_line_ids = [shorten_line_id(idx, row.bus0, row.bus1) for idx, row in n.lines.iterrows()]

line_summary = pd.DataFrame({
    "short_line_id": short_line_ids,
    "bus0": n.lines["bus0"],
    "bus1": n.lines["bus1"],
    "capacity_MW": caps,
    "max_flow_MW": flows.max(axis=0),
    "max_loading_pct": loading_ratios.max(axis=0),
    "mean_loading_pct": loading_ratios.mean(axis=0),
    "hours_near_capacity_65pct": (loading_ratios >= 65.0).sum(axis=0)
}, index=n.lines.index).sort_values(by="mean_loading_pct", ascending=False)

csv_out_path = os.path.join(OUT_DIR, "transmission_line_loading_analysis.csv")
line_summary.to_csv(csv_out_path)
print(f"  saved CSV line analysis        -> {csv_out_path}")

if has_mu:
    metric_series = n.lines_t.mu_upper.abs().mean(axis=0).sort_values(ascending=False)
    metric_name = "Mean Shadow Price (€/MW)"
else:
    metric_series = line_summary["mean_loading_pct"]
    metric_name = "Mean Capacity Loading (%)"

line_txt_path = os.path.join(OUT_DIR, "transmission_line_bottlenecks.txt")
with open(line_txt_path, "w", encoding="utf-8") as f:
    f.write("=" * 125 + "\n")
    f.write(f" PyPSA-Libya — Transmission Line Bottlenecks Report ({metric_name})\n")
    f.write("=" * 125 + "\n\n")
    
    header = f"{'LINE ID':12s} | {'From BUS':10s} | {'To BUS':10s} | {'CAP (MW)':12s} | {'MAX FLOW (MW)':14s} | {'MAX LOAD %':11s} | {'MEAN LOAD %':11s} | {'From BUS REGION':18s} | {'To BUS REGION':18s}"
    f.write(header + "\n")
    f.write("-" * 125 + "\n")
    
    for idx, row in line_summary.iterrows():
        _, _, region0 = get_bus_info(n, row.bus0)
        _, _, region1 = get_bus_info(n, row.bus1)
        short_id = row["short_line_id"]
        
        line_row = f"{short_id:12s} | {str(row.bus0):10s} | {str(row.bus1):10s} | {row.capacity_MW:12.1f} | {row.max_flow_MW:14.1f} | {row.max_loading_pct:10.2f}% | {row.mean_loading_pct:10.2f}% | {region0:18s} | {region1:18s}"
        f.write(line_row + "\n")

print(f"  saved Line Bottlenecks report  -> {line_txt_path}")

top_lines = metric_series.head(15)
short_top_labels = [line_summary.loc[idx, "short_line_id"] if idx in line_summary.index else str(idx) for idx in top_lines.index]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(short_top_labels, top_lines.values, color="#dc2626")
ax.set_ylabel(metric_name)
ax.set_title(f"PyPSA-Libya — Top 15 Transmission Bottlenecks ({metric_name})")
plt.xticks(rotation=45, ha='right')

ax.bar_label(bars, fmt='{:,.1f}', padding=3, fontsize=8)
ax.margins(y=0.15)

ax.set_axisbelow(True)
ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cbd5e1')
    
fig.tight_layout()
out_line_png = os.path.join(OUT_DIR, "transmission_line_bottlenecks.png")
fig.savefig(out_line_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved Line Bottlenecks chart   -> {out_line_png}")

fig = plt.figure(figsize=(11, 8))
ax = setup_map_ax(fig)

min_val = metric_series.min()
max_val = metric_series.max() if metric_series.max() > 0 else 1.0

norm_metric = mcolors.Normalize(vmin=min_val, vmax=max_val)
cmap_lines = cm.get_cmap("Reds")

for idx, row in n.lines.iterrows():
    x0, y0 = find_bus_coords(n, row.bus0)
    x1, y1 = find_bus_coords(n, row.bus1)
    if x0 is not None and x1 is not None and (x0 != x1 or y0 != y1):
        val = metric_series.get(idx, 0.0)
        line_color = cmap_lines(norm_metric(val)) if val > (min_val + 0.05 * (max_val - min_val)) else "#a1a1aa"
        linewidth = 1.0 + 3.0 * norm_metric(val)
        xs, ys = route_line_land(x0, y0, x1, y1)
        ax.plot(xs, ys, color=line_color, lw=linewidth, zorder=3, alpha=0.85)

top_10_lines = metric_series.head(10)
line_offsets = [
    (12, 12), (-16, 14), (14, -16), (-18, -14), 
    (16, -6), (-16, 6), (6, 14), (-6, -14),
    (14, 14), (-14, -14)
]

for i, idx in enumerate(top_10_lines.index):
    if idx in n.lines.index:
        row = n.lines.loc[idx]
        x0, y0 = find_bus_coords(n, row.bus0)
        x1, y1 = find_bus_coords(n, row.bus1)
        if x0 is not None and x1 is not None:
            xs, ys = route_line_land(x0, y0, x1, y1)
            mid_idx = len(xs) // 2
            mid_x = xs[mid_idx]
            mid_y = ys[mid_idx]
            
            raw_offset = line_offsets[i % len(line_offsets)]
            safe_offset = get_safe_offset(mid_x, mid_y, raw_offset)
            
            short_id = line_summary.loc[idx, "short_line_id"] if idx in line_summary.index else f"L{idx}"
            
            ax.annotate(
                short_id,
                (mid_x, mid_y),
                xytext=safe_offset,
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="normal",
                color="black",
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.6, alpha=0.8),
                zorder=6
            )

macro_buses = n.buses[~n.buses.index.str.contains("low voltage|battery|H2", case=False)]
ax.scatter(macro_buses.x, macro_buses.y, s=15, c="#1e293b", edgecolors="white", linewidths=0.5, zorder=4)

sm_line = cm.ScalarMappable(cmap=cmap_lines, norm=norm_metric)
sm_line.set_array([])
cbar_line = fig.colorbar(sm_line, ax=ax, orientation="vertical", shrink=0.7, pad=0.03)
cbar_line.set_label(metric_name, fontsize=10, fontweight="normal")

ax.set_title(f"PyPSA-Libya — Transmission Bottlenecks Map\nLine labels highlight Top 10 Congested Corridors ({metric_name})")

out_map_png = os.path.join(OUT_DIR, "transmission_line_bottlenecks_map.png")
fig.savefig(out_map_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved Line Bottlenecks Map     -> {out_map_png}")

# ---------------------------------------------------------------------
# PART 5 — System-Wide Objective Cost Report & LCOE
# ---------------------------------------------------------------------

print("\nPART 5: System-Wide Objective Cost & LCOE Output")
print("-" * 70)

objective_cost = getattr(n, "objective", None)
if objective_cost is None or pd.isna(objective_cost):
    objective_cost = getattr(n, "objective_value", np.nan)

if hasattr(n, "loads_t") and not n.loads_t.p.empty:
    total_demand_mwh = (n.loads_t.p.sum().sum() * dt_hours)
else:
    total_demand_mwh = 0.0

total_demand_gwh = total_demand_mwh / 1e3

if total_demand_mwh > 0 and pd.notna(objective_cost):
    lcoe_eur_mwh = objective_cost / total_demand_mwh
    lcoe_eur_kwh = lcoe_eur_mwh / 1000.0
    lcoe_cents_kwh = lcoe_eur_kwh * 100.0
else:
    lcoe_eur_mwh = np.nan
    lcoe_eur_kwh = np.nan
    lcoe_cents_kwh = np.nan

def format_cost_details(cost_val):
    if cost_val is None or pd.isna(cost_val):
        return "N/A", "N/A", "N/A"
    
    std_str = f"€{cost_val:,.2f}"
    sci_str = f"{cost_val:.3E}".replace("E+0", "E").replace("E+", "E")
    
    abs_val = abs(cost_val)
    if abs_val >= 1e12:
        scale_str = f"€{cost_val / 1e12:.3f} Trillion"
    elif abs_val >= 1e9:
        scale_str = f"€{cost_val / 1e9:.3f} Billion"
    elif abs_val >= 1e6:
        scale_str = f"€{cost_val / 1e6:.3f} Million"
    else:
        scale_str = f"€{cost_val:,.2f}"
        
    return std_str, sci_str, scale_str

std_cost, sci_cost, scale_cost = format_cost_details(objective_cost)

print(f"  Solver System Objective Cost: {std_cost} ({sci_cost} €) -> {scale_cost}")
if pd.notna(lcoe_eur_mwh):
    print(f"  Levelized Cost of Electricity (LCOE): €{lcoe_eur_mwh:.2f} / MWh  (€{lcoe_eur_kwh:.4f} / kWh)")

cost_file_path = os.path.join(OUT_DIR, "system_objective_cost.txt")
with open(cost_file_path, "w", encoding="utf-8") as f:
    f.write("=" * 65 + "\n")
    f.write(" PyPSA-Libya — System-Wide Optimized Objective Cost Report\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"  Run Name                 : {RUN_NAME}\n")
    f.write(f"  Clusters                 : {CLUSTERS}\n")
    f.write(f"  Run Scope                : {duration_label}\n")
    f.write(f"  CCL Limits Applied       : {USE_CCL}\n")
    f.write(f"  Solved Network File      : {SOLVED_PATH}\n\n")
    f.write("-" * 65 + "\n")
    f.write(f"  TOTAL SYSTEM OBJECTIVE COST : {std_cost}\n")
    f.write(f"  SCIENTIFIC NOTATION         : ({sci_cost} €)\n")
    f.write(f"  SCALE REPRESENTATION        : {scale_cost}\n")
    f.write("-" * 65 + "\n")
    f.write(f"  TOTAL ANNUAL DEMAND SERVED  : {total_demand_gwh:,.2f} GWh ({total_demand_mwh:,.2f} MWh)\n")
    if pd.notna(lcoe_eur_mwh):
        f.write(f"  LEVELIZED COST OF ENERGY    : €{lcoe_eur_mwh:,.2f} / MWh\n")
        f.write(f"  LCOE PER KILOWATT-HOUR      : €{lcoe_eur_kwh:.4f} / kWh  ({lcoe_cents_kwh:.2f} Cent/kWh)\n")
    else:
        f.write("  LEVELIZED COST OF ENERGY    : N/A\n")
    f.write("-" * 65 + "\n\n")
    f.write("Note: This scalar value represents total annualized system cost\n")
    f.write("      (CAPEX + OPEX + load shedding penalties) minimized by solver.\n")
    f.write("      LCOE is calculated as Total System Cost divided by Total Annual Demand.\n")

print(f"  Saved objective cost report    -> {cost_file_path}")
print("\n" + "=" * 70)
print(f"Done. All outputs generated in ./{OUT_DIR}/")
print("=" * 70)