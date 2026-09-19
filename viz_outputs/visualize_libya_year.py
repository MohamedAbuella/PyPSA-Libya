"""
visualize_libya_year.py
Run from the pypsa-libya folder in the pypsa-earth conda environment:

    python visualize_libya_year.py

Generates PNG charts, text/CSV reports, dual/shadow prices, LCOE, renewable curtailment, 
and transmission bottleneck maps in: viz_outputs/libya-year-<CLUSTERS>clusters-<HOURS>h/
"""

import os
import sys
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

#### SWITCH 1: CO2 EMISSION CASE ("LCO2" or "NoLCO2") ---
CASE = "LCO2"
# CASE = "NoLCO2"

#####SWITCH 2: TEMPORAL RESOLUTION (6 or 3) ---
# HOURS = 6
HOURS = 3

USE_CCL = False

#### Mapping base run names
RUN_NAME_BY_CLUSTERS = {
    100: "libya-year-full",
    40: "libya-year-40AggBuses",
    50: "libya-year-50AggBuses",
    36: "libya-year-36",
}

RUN_NAME = RUN_NAME_BY_CLUSTERS.get(CLUSTERS, f"libya-year-{CLUSTERS}clusters")

# --- DYNAMIC INPUT SUB-FOLDER ROUTING ---
# Ensures "NoLCO2" maps to "No_LCO2" directory name for all resolutions
input_hours_folder = f"{HOURS}h"
input_case_folder = "No_LCO2" if CASE == "NoLCO2" else CASE


# --- DYNAMIC OUTPUT DIRECTORY ROUTING ---
OUT_DIR = os.path.join("viz_outputs", f"libya-year-{CLUSTERS}clusters", CASE, f"{HOURS}h")
os.makedirs(OUT_DIR, exist_ok=True)

#### --- THIS FOR PAPER PDF SUPPORT ---
PAPER_DIR = os.path.join(OUT_DIR, "Paper_Figures")
os.makedirs(PAPER_DIR, exist_ok=True)


def save_paper_pdf(fig, ax, filename, is_map=False):
    """
    Saves the publication PDF to Paper_Figures.

    For map figures:
      - Rasterizes geographic/background layers and transmission lines
      - Keeps markers, labels, annotations, legends and text as vectors
      - Uses 300 dpi for the rasterized map layer

    This keeps PDF file sizes reasonable while preserving publication quality.
    """
    original_title = ax.get_title()
    ax.set_title("")

    # ------------------------------------------------------------
    # Standard naming
    # ------------------------------------------------------------
    clean_name = re.sub(r'^figure\d+[a-z]?_', '', filename)

    if not clean_name.startswith('fig_'):
        clean_name = 'fig_' + clean_name

    if not clean_name.endswith('.pdf'):
        clean_name += '.pdf'

    pdf_path = os.path.join(PAPER_DIR, clean_name)

    # ------------------------------------------------------------
    # MAP PDF OPTIMIZATION
    # ------------------------------------------------------------
    if is_map:
        # Rasterize everything below zorder 4.
        #
        # zorder 0-1 : Cartopy LAND/OCEAN/BORDERS/COASTLINE
        # zorder 2-3 : transmission/network lines
        # zorder 4+  : bubbles, markers, labels, annotations
        #
        # Thus the expensive geographic geometry becomes one raster
        # image while important scientific annotations remain vectors.
        ax.set_rasterization_zorder(4.0)

    # ------------------------------------------------------------
    # Save PDF
    # ------------------------------------------------------------
    fig.savefig(
        pdf_path,
        format="pdf",
        dpi=300,
        bbox_inches="tight"
    )

    # Restore original title for normal PNG/display output
    ax.set_title(original_title)

    print(f" Saved Paper PDF: {clean_name}")
    
    

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



# =====================================================================
# EARLY CHECK: Validate solved file existence before running Part 1
# =====================================================================
sol_file_name = f"elec_s_{CLUSTERS}_ec_lcopt_Co2L-{HOURS}h.nc"
SOLVED_PATH = os.path.join("Output", RUN_NAME, input_case_folder, input_hours_folder, "networks", sol_file_name)

if not os.path.exists(SOLVED_PATH):
    print("\n" + "!" * 75)
    print(f"[NOTICE] Solved network file does not exist at expected location:")
    print(f"         '{SOLVED_PATH}'")
    print(f"Skipping visualization script for CASE='{CASE}' and HOURS={HOURS}h.")
    print("!" * 75 + "\n")
    sys.exit(0)

BASE_PATH = f"networks/{RUN_NAME}/base.nc"
ELEC_PATH = f"networks/{RUN_NAME}/elec.nc"
base_n = pypsa.Network(BASE_PATH) if os.path.exists(BASE_PATH) else (pypsa.Network(ELEC_PATH) if os.path.exists(ELEC_PATH) else None)




# ---------------------------------------------------------------------
# PART 2 — Solved Network Capacity & Map
# ---------------------------------------------------------------------

print("\nPART 2: Solved network capacity results")
print("-" * 70)

print(f"Loading solved network from: {SOLVED_PATH}")

# Mappings and Color Palette
CAPEX_CARRIER_MAP_ELEC = {
    "solar": "Solar PV",
    "onwind": "Onshore Wind",
    "offwind": "Offshore Wind",
    "CCGT": "CCGT",
    "OCGT": "OCGT",
    "oil": "Oil",
    "heavy fuel oil": "Oil",
    "battery": "Battery",
    "H2 Fuel Cell": "H2 Fuel Cell",
    "H2": "H2 Storage",
}

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
    # Aggregate power capacities, filter tiny values, and convert MW -> GW
    all_power_cap_mw = cap_df.groupby("carrier")["capacity"].sum().sort_values()
    all_power_cap_gw = all_power_cap_mw[all_power_cap_mw > 1e-2] / 1000.0
    
    # Map raw carrier names to clean capitalized labels
    formatted_labels = [CAPEX_CARRIER_MAP_ELEC.get(c, c.capitalize()) for c in all_power_cap_gw.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(formatted_labels, all_power_cap_gw.values, color=[colors.get(c, "gray") for c in all_power_cap_gw.index])
    ax.set_xlabel("Optimal Installed Power Capacity (GW)")
    ax.set_title("PyPSA-Libya — Optimal Power Capacity by Carrier (GW)")
    
    ax.bar_label(bars, fmt='{:,.2f} GW', padding=5, fontsize=9, fontweight='normal')
    ax.margins(x=0.20)
    
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cbd5e1')
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "solved_capacity_by_carrier.png"), dpi=150, bbox_inches="tight")
    save_paper_pdf(fig, ax, "solved_capacity_by_carrier")
    plt.close(fig)

# ---------------------------------------------------------------------
# Barplot Update: Physical Generation & Storage Activity in TWh
# ---------------------------------------------------------------------

phys_gen_dict = {}

def get_clean_label(raw_carrier):
    suffix = ""
    clean_c = raw_carrier
    if " (discharge)" in raw_carrier:
        clean_c = raw_carrier.replace(" (discharge)", "")
        suffix = " (discharge)"
    elif " (charge)" in raw_carrier:
        clean_c = raw_carrier.replace(" (charge)", "")
        suffix = " (charge)"
    mapped_name = CAPEX_CARRIER_MAP_ELEC.get(clean_c, clean_c.title())
    return f"{mapped_name}{suffix}"

if not n.generators.empty:
    valid_gens = n.generators[n.generators.carrier != "load shedding"].copy()
    if not valid_gens.empty:
        mapped_carriers = valid_gens.carrier.map(map_carrier_name)
        gen_p = n.generators_t.p[valid_gens.index].groupby(mapped_carriers, axis=1).sum()
        for carrier in gen_p.columns:
            # Convert GWh to TWh (/ 1e6 instead of / 1e3)
            energy_twh = (gen_p[carrier] * dt_hours).sum() / 1e6
            if energy_twh > 1e-4:
                phys_gen_dict[carrier] = energy_twh

if not n.storage_units.empty and not n.storage_units_t.p.empty:
    su_p = n.storage_units_t.p
    su_carrier_mapped = n.storage_units.carrier.map(map_carrier_name)
    for carrier in su_carrier_mapped.unique():
        cols = n.storage_units[su_carrier_mapped == carrier].index
        p_carrier = su_p[cols]

        dis_twh = (p_carrier.clip(lower=0).sum(axis=1) * dt_hours).sum() / 1e6
        if dis_twh > 1e-4:
            phys_gen_dict[f"{carrier} (discharge)"] = dis_twh

        chg_twh = ((-p_carrier.clip(upper=0)).sum(axis=1) * dt_hours).sum() / 1e6
        if chg_twh > 1e-4:
            phys_gen_dict[f"{carrier} (charge)"] = chg_twh

phys_gen_twh = pd.Series(phys_gen_dict)

if "load shedding" in phys_gen_twh.index:
    phys_gen_twh = phys_gen_twh.drop(index="load shedding")

mapped_x_labels = [get_clean_label(c) for c in phys_gen_twh.index]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(mapped_x_labels, phys_gen_twh.values, color=[colors.get(c, "gray") for c in phys_gen_twh.index])
ax.set_ylabel("Physical Generation / Activity (TWh)")
ax.set_title("PyPSA-Libya — Physical Energy Generation & Storage Activity")
plt.xticks(rotation=30, ha='right')

ax.bar_label(bars, fmt='{:,.2f}', padding=4, fontsize=9, fontweight='normal')
ax.margins(y=0.18)

ax.set_axisbelow(True)
ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cbd5e1')

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "physical_generation_by_carrier.png"), dpi=150, bbox_inches="tight")
save_paper_pdf(fig, ax, "physical_generation_by_carrier")
plt.close(fig)

# ---------------------------------------------------------------------
# Solved Capacity Map (Labels Filtered to Capacity >= 1 GW)
# ---------------------------------------------------------------------

fig = plt.figure(figsize=(11, 8))
ax = setup_map_ax(fig)
draw_base_lines(base_n if base_n is not None else n, ax, color="#888888", alpha=0.5, linewidth=0.9)

BUBBLE_BASE_SIZE = 40.0
BUBBLE_MAX_ADD_SIZE = 600.0
TOP_GAP_PTS = 3.0

if not cap_df.empty:
    gen_by_bus = cap_df.groupby(["bus", "carrier"])["capacity"].sum().unstack(fill_value=0)
    total_cap_by_bus = gen_by_bus.sum(axis=1)
    dominant_carrier_by_bus = gen_by_bus.idxmax(axis=1)
    max_bus_cap = total_cap_by_bus.max() if total_cap_by_bus.max() > 0 else 1.0

    for bus_id, total_cap in total_cap_by_bus.items():
        if total_cap <= 0.1:
            continue

        x, y = find_bus_coords(n, bus_id)
        if x is None or y is None:
            continue

        dominant_carrier = dominant_carrier_by_bus.get(bus_id, "none")
        color = colors.get(dominant_carrier, "gray")

        # Draw bubble for all locations with capacity > 0.1 MW
        size_bubble = BUBBLE_BASE_SIZE + BUBBLE_MAX_ADD_SIZE * (total_cap / max_bus_cap)

        ax.scatter(
            x, y,
            s=size_bubble,
            c=color,
            edgecolors="black",
            linewidths=0.6,
            zorder=4,
            alpha=0.85
        )

        # Annotate capacity label ONLY if total capacity is >= 1 GW (1000 MW)
        if total_cap >= 1000.0:
            cap_label_text = f"{total_cap/1e3:.1f}GW"
            radius_pts = (size_bubble ** 0.5) / 2.0
            top_offset_pts = radius_pts + TOP_GAP_PTS

            ax.annotate(
                cap_label_text,
                (x, y),
                xytext=(0, top_offset_pts),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="normal",
                color="black",
                zorder=6
            )

ax.set_title(f"PyPSA-Libya — Solved Power Capacity Allocation by Bus ({CLUSTERS} Clusters, {HOURS}h)")
fig.savefig(os.path.join(OUT_DIR, "solved_capacity_map.png"), dpi=150, bbox_inches="tight")
save_paper_pdf(fig, ax, "solved_capacity_map", is_map=True)
plt.close(fig)


# ---------------------------------------------------------------------
# PART 3 — Energy Generation Mix & Renewable Performance (CF & Curtailment)
# ---------------------------------------------------------------------

print("\nPART 3: Energy Generation, Dispatch Profile & Renewable CF/Curtailment")
print("-" * 70)


# Mappings and Color Palette
CAPEX_CARRIER_MAP_ELEC = {
    "solar": "Solar PV",
    "onwind": "Onshore Wind",
    "offwind": "Offshore Wind",
    "CCGT": "CCGT",
    "OCGT": "OCGT",
    "oil": "Oil",
    "heavy fuel oil": "Oil",
    "battery": "Battery",
    "H2 Fuel Cell": "Fuel Cell",
    "H2": "H2 Storage",
}


phys_gen_dict = {}

# Helper function to map carrier names using CAPEX_CARRIER_MAP_ELEC
def get_clean_label(raw_carrier):
    # Handle suffixes for storage state
    suffix = ""
    clean_c = raw_carrier
    if " (discharge)" in raw_carrier:
        clean_c = raw_carrier.replace(" (discharge)", "")
        suffix = " (discharge)"
    elif " (charge)" in raw_carrier:
        clean_c = raw_carrier.replace(" (charge)", "")
        suffix = " (charge)"
    
    # Map or fallback to title case
    mapped_name = CAPEX_CARRIER_MAP_ELEC.get(clean_c, clean_c.title())
    return f"{mapped_name}{suffix}"

# 1. Generator Energy Production in TWh (MWh / 1e6)
if not n.generators.empty:
    valid_gens = n.generators[n.generators.carrier != "load shedding"].copy()
    if not valid_gens.empty:
        mapped_carriers = valid_gens.carrier.map(map_carrier_name)
        gen_p = n.generators_t.p[valid_gens.index].groupby(mapped_carriers, axis=1).sum()
        for carrier in gen_p.columns:
            energy_twh = (gen_p[carrier] * dt_hours).sum() / 1e6
            if energy_twh > 1e-4:  # Filter out negligible/zero generation (< 0.0001 TWh)
                phys_gen_dict[carrier] = energy_twh

# 2. Storage Discharging (+) and Charging (-) Energy Separated in TWh
if not n.storage_units.empty and not n.storage_units_t.p.empty:
    su_p = n.storage_units_t.p
    su_carrier_mapped = n.storage_units.carrier.map(map_carrier_name)
    for carrier in su_carrier_mapped.unique():
        cols = n.storage_units[su_carrier_mapped == carrier].index
        p_carrier = su_p[cols]

        dis_twh = (p_carrier.clip(lower=0).sum(axis=1) * dt_hours).sum() / 1e6
        if dis_twh > 1e-4:
            phys_gen_dict[f"{carrier} (discharge)"] = dis_twh

        chg_twh = ((-p_carrier.clip(upper=0)).sum(axis=1) * dt_hours).sum() / 1e6
        if chg_twh > 1e-4:
            phys_gen_dict[f"{carrier} (charge)"] = chg_twh

phys_gen_twh = pd.Series(phys_gen_dict)

# Further ensure load shedding is removed if it enters via mapping
if "load shedding" in phys_gen_twh.index:
    phys_gen_twh = phys_gen_twh.drop(index="load shedding")

# Create capitalized / mapped x-axis labels
mapped_x_labels = [get_clean_label(c) for c in phys_gen_twh.index]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(mapped_x_labels, phys_gen_twh.values, color=[colors.get(c, "gray") for c in phys_gen_twh.index])
ax.set_ylabel("Physical Generation / Activity (TWh)")
ax.set_title("PyPSA-Libya — Physical Energy Generation & Storage Activity")
plt.xticks(rotation=30, ha='right')

# Format labels with 2 decimal places in TWh scale
ax.bar_label(bars, fmt='{:,.2f}', padding=4, fontsize=9, fontweight='normal')
ax.margins(y=0.18)

ax.set_axisbelow(True)
ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cbd5e1')

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "physical_generation_by_carrier.png"), dpi=150, bbox_inches="tight")
save_paper_pdf(fig, ax, "physical_generation_by_carrier")
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
    save_paper_pdf(fig, ax, "annual_dispatch_timeseries")
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
    save_paper_pdf(fig, ax, "monthly_dispatch_august_bar")
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
ren_txt_path = os.path.join(OUT_DIR, "renewable_cf_and_curtailment_summary.txt")
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
    save_paper_pdf(fig, ax, "renewable_cf_and_curtailment")
    plt.close(fig)
    print(f"  saved Renewable CF & Curtailment chart -> {ren_png_path}")

# ---------------------------------------------------------------------
# PART 4 — Dual Variables & Transmission Bottleneck Analysis
# ---------------------------------------------------------------------

print("\nPART 4: Shadow Prices & Transmission Bottleneck Analysis")
print("-" * 70)

if hasattr(n, "buses_t") and not n.buses_t.marginal_price.empty:
    mean_lmp = n.buses_t.marginal_price.mean(axis=0).sort_values(ascending=False)
    
    lmp_txt_path = os.path.join(OUT_DIR, "locational_marginal_prices_summary.txt")
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
    save_paper_pdf(fig, ax, "lmp_by_bus")
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
    save_paper_pdf(fig, ax, "lmp_locations_map", is_map=True)
    plt.close(fig)
    print(f"  saved LMP locations map        -> {out_lmp_map_png}")

else:
    print("  [skip] No buses_t.marginal_price data found in solved network.")

# Carbon Shadow Price
co2_txt_path = os.path.join(OUT_DIR, "carbon_shadow_price_summary.txt")
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

line_txt_path = os.path.join(OUT_DIR, "transmission_line_bottlenecks_summary.txt")
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
save_paper_pdf(fig, ax, "transmission_line_bottlenecks")
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
save_paper_pdf(fig, ax, "transmission_line_bottlenecks_map", is_map=True)
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

cost_file_path = os.path.join(OUT_DIR, "system_objective_cost_summary.txt")
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



# =====================================================================
# PART 6 — CO2 Emissions Summary & Mass Balance Accounting
# =====================================================================

print("\n" + "=" * 70)
print("PART 6: CO2 Emissions Summary & Mass Balance Accounting")
print("=" * 70)

co2_records = []

## Default emission factors (tCO2 / MWh_thermal) if missing in network attributes
DEFAULT_CO2_EMISSIONS = {
    "CCGT": 0.187,
    "OCGT": 0.187,
    "oil": 0.248,
    "heavy fuel oil": 0.248,
    "coal": 0.354,
    "lignite": 0.334,
}

if not n.generators.empty and not n.generators_t.p.empty:
    gen_p = n.generators_t.p  # Output in MW
    
    for gen_id, gen_row in n.generators.iterrows():
        carrier = gen_row.carrier
        
        # Determine CO2 emission factor (tCO2 / MWh_th)
        co2_factor = 0.0
        if hasattr(n, "carriers") and not n.carriers.empty and carrier in n.carriers.index and "co2_emissions" in n.carriers.columns:
            co2_factor = float(n.carriers.loc[carrier, "co2_emissions"])
        elif "co2_emissions" in gen_row and pd.notna(gen_row["co2_emissions"]):
            co2_factor = float(gen_row["co2_emissions"])
        else:
            co2_factor = DEFAULT_CO2_EMISSIONS.get(carrier.lower(), DEFAULT_CO2_EMISSIONS.get(carrier, 0.0))
        
        if co2_factor > 0:
            efficiency = float(gen_row.get("efficiency", 1.0))
            if efficiency <= 0 or pd.isna(efficiency):
                efficiency = 1.0
                
            gen_mwh_el = (gen_p[gen_id] * dt_hours).sum()
            fuel_mwh_th = gen_mwh_el / efficiency
            emissions_tco2 = fuel_mwh_th * co2_factor
            
            co2_records.append({
                "generator": gen_id,
                "carrier": carrier,
                "generation_gwh": gen_mwh_el / 1e3,
                "efficiency": efficiency,
                "fuel_gwh_th": fuel_mwh_th / 1e3,
                "co2_factor_tco2_mwhth": co2_factor,
                "emissions_tco2": emissions_tco2,
                "emissions_mtonnes": emissions_tco2 / 1e6
            })

df_co2 = pd.DataFrame(co2_records)

if not df_co2.empty:
    co2_by_carrier = df_co2.groupby("carrier").agg({
        "generation_gwh": "sum",
        "fuel_gwh_th": "sum",
        "emissions_tco2": "sum",
        "emissions_mtonnes": "sum"
    }).reset_index()
    
    total_emissions_tco2 = co2_by_carrier["emissions_tco2"].sum()
    total_emissions_mtonnes = co2_by_carrier["emissions_mtonnes"].sum()
    total_gen_gwh = (n.generators_t.p.sum().sum() * dt_hours) / 1e3 if not n.generators_t.p.empty else 0.0
    
    carbon_intensity_g_kwh = (total_emissions_tco2 / (total_gen_gwh * 1e3)) if total_gen_gwh > 0 else 0.0

    print(f"  Total System CO2 Emissions : {total_emissions_mtonnes:,.3f} Million Tonnes CO2")
    print(f"  Average Carbon Intensity   : {carbon_intensity_g_kwh:,.1f} g CO2/kWh")
    
    # 1. Save Text Summary Report
    co2_report_path = os.path.join(OUT_DIR, "co2_emissions_summary.txt")
    with open(co2_report_path, "w", encoding="utf-8") as f:
        f.write("=" * 85 + "\n")
        f.write(f" PyPSA-Libya — CO2 Emissions Summary & Mass Balance ({duration_label})\n")
        f.write("=" * 85 + "\n\n")
        f.write(f"{'CARRIER':15s} | {'GEN (GWh)':12s} | {'FUEL (GWh_th)':15s} | {'EMISSIONS (tCO2)':18s} | {'EMISSIONS (MtCO2)':18s}\n")
        f.write("-" * 85 + "\n")
        for _, row in co2_by_carrier.iterrows():
            f.write(f"{row['carrier']:15s} | {row['generation_gwh']:12,.1f} | {row['fuel_gwh_th']:15,.1f} | {row['emissions_tco2']:18,.1f} | {row['emissions_mtonnes']:18,.3f}\n")
        f.write("-" * 85 + "\n")
        f.write(f"{'TOTAL':15s} | {total_gen_gwh:12,.1f} | {co2_by_carrier['fuel_gwh_th'].sum():15,.1f} | {total_emissions_tco2:18,.1f} | {total_emissions_mtonnes:18,.3f}\n")
        f.write("-" * 85 + "\n\n")
        f.write(f"  System Carbon Intensity : {carbon_intensity_g_kwh:,.2f} g CO2 / kWh\n")
        
        if hasattr(n, "global_constraints") and not n.global_constraints.empty and "CO2Limit" in n.global_constraints.index:
            limit_val = n.global_constraints.loc["CO2Limit", "constant"]
            f.write(f"  CO2 Cap Target Limit    : {limit_val:,.2f} Tonnes CO2 ({limit_val/1e6:,.3f} MtCO2)\n")
            f.write(f"  Cap Compliance Margin   : {(limit_val - total_emissions_tco2):,.2f} Tonnes CO2\n")
        f.write("=" * 85 + "\n")

    print(f"  Saved CO2 summary text report -> {co2_report_path}")

    # 2. Save CO2 Emissions Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(co2_by_carrier["carrier"], co2_by_carrier["emissions_mtonnes"], 
                  color=[colors.get(c, "#475569") for c in co2_by_carrier["carrier"]], width=0.5)
    ax.set_ylabel("CO2 Emissions (Million Tonnes)")
    ax.set_title("PyPSA-Libya — Total CO2 Emissions by Carrier")
    ax.bar_label(bars, fmt='{:,.2f} Mt', padding=4, fontsize=9)
    ax.margins(y=0.2)
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cbd5e1')
    
    fig.tight_layout()
    co2_png_path = os.path.join(OUT_DIR, "co2_emissions_by_carrier.png")
    fig.savefig(co2_png_path, dpi=150, bbox_inches="tight")
    save_paper_pdf(fig, ax, "co2_emissions_by_carrier")
    plt.close(fig)
    print(f"  Saved CO2 emissions chart     -> {co2_png_path}")
else:
    print("  No active CO2 emitting generators found in the solved network.")
    


# =====================================================================
# PART 7 — DYNAMIC NET PRESENT VALUE (NPV) & REVENUE ANALYSIS
# =====================================================================

print("\n" + "=" * 70)
print("PART 8: Dynamic Net Present Value (NPV), Cash Flow & Revenue Analysis")
print("=" * 70)

# Financial and Operational Parameters
DISCOUNT_RATE = 0.07          # 7% WACC / Discount Rate
PROJECT_LIFETIME = 25         # 25-Year Financial Horizon

# ELECTRICITY_PRICE_EUR_MWH = 65.0  # Assumed Wholesale Market Electricity Price (€/MWh)
ELECTRICITY_PRICE_EUR_MWH = 75.0

# Extract Core Annualized System Cost from Network
annual_capex = 0.0
if hasattr(n, "generators") and not n.generators.empty:
    annual_capex += (n.generators.capital_cost * n.generators.p_nom_opt).sum()
if hasattr(n, "storage_units") and not n.storage_units.empty:
    annual_capex += (n.storage_units.capital_cost * n.storage_units.p_nom_opt).sum()
if hasattr(n, "lines") and not n.lines.empty and "capital_cost" in n.lines.columns:
    annual_capex += (n.lines.capital_cost * (n.lines.s_nom_opt - n.lines.s_nom).clip(lower=0)).sum()

annual_opex = 0.0
weights = n.snapshot_weightings.generators if hasattr(n, "snapshot_weightings") else 1.0
if hasattr(n, "generators_t") and not n.generators_t.p.empty:
    mc = n.generators.marginal_cost
    annual_opex += (n.generators_t.p.mul(mc, axis=1).mul(weights, axis=0)).sum().sum()

annual_total_cost = n.objective # annual_capex + annual_opex

# Total Annual Demand Served (MWh)
if hasattr(n, "loads_t") and not n.loads_t.p.empty:
    annual_demand_mwh = (n.loads_t.p.sum().sum() * dt_hours)
else:
    annual_demand_mwh = 0.0

# 1. Levelized Cost of Electricity (LCOE)
lcoe_eur_mwh_npv = (annual_total_cost / annual_demand_mwh) if annual_demand_mwh > 0 else 0.0

# 2. Capital Expenditure (CAPEX) Estimation from Annualized CAPEX
# Approximating Overnight CAPEX using Capital Recovery Factor (CRF)
crf = (DISCOUNT_RATE * (1 + DISCOUNT_RATE)**PROJECT_LIFETIME) / (((1 + DISCOUNT_RATE)**PROJECT_LIFETIME) - 1)
overnight_capex = annual_capex / crf if crf > 0 else annual_capex * 10.0

# 3. Revenue & Cash Flow Modeling (25-Year Projection)
annual_revenue = annual_demand_mwh * ELECTRICITY_PRICE_EUR_MWH
annual_net_cash_flow = annual_revenue - annual_opex

years = np.arange(0, PROJECT_LIFETIME + 1)
cash_flows = np.zeros(PROJECT_LIFETIME + 1)
discounted_cash_flows = np.zeros(PROJECT_LIFETIME + 1)

# Year 0: Initial Capital Outlay (-CAPEX)
cash_flows[0] = -overnight_capex
discounted_cash_flows[0] = -overnight_capex

# Years 1-25: Net Operational Income / (1 + r)^t
for t in range(1, PROJECT_LIFETIME + 1):
    cash_flows[t] = annual_net_cash_flow
    discounted_cash_flows[t] = annual_net_cash_flow / ((1 + DISCOUNT_RATE)**t)

cumulative_npv = np.cumsum(discounted_cash_flows)
final_npv = cumulative_npv[-1]

# 4. Internal Rate of Return (IRR) Calculation
try:
    import numpy_financial as npf
    project_irr = npf.irr(cash_flows) * 100.0
except ImportError:
    # Fallback numerical IRR root-finder if numpy_financial is not installed
    def calc_npv(rate):
        return sum([cf / ((1 + rate)**t) for t, cf in enumerate(cash_flows)])
    try:
        from scipy.optimize import brentq
        project_irr = brentq(calc_npv, -0.05, 0.50) * 100.0
    except Exception:
        project_irr = np.nan

# 4b. Payback Period (Years) = Overnight CAPEX / Annual Net Cash Flow
if annual_net_cash_flow > 0 and overnight_capex > 0:
    payback_years = overnight_capex / annual_net_cash_flow
else:
    payback_years = np.nan


# 5. Output Financial Summary Text File
npv_txt_path = os.path.join(OUT_DIR, "npv_and_financial_analysis_summary.txt")
with open(npv_txt_path, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write(f" PyPSA-Libya — Net Present Value (NPV) & Financial Analysis ({PROJECT_LIFETIME} Years)\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"  Discount Rate (WACC)        : {DISCOUNT_RATE * 100:.1f} %\n")
    f.write(f"  Project Lifetime            : {PROJECT_LIFETIME} Years\n")
    f.write(f"  Assumed Electricity Price   : €{ELECTRICITY_PRICE_EUR_MWH:.2f} / MWh\n\n")
    f.write("-" * 80 + "\n")
    f.write(f"  Estimated Overnight CAPEX   : €{overnight_capex / 1e9:,.3f} Billion\n")
    f.write(f"  Annualized CAPEX            : €{annual_capex / 1e9:,.3f} Billion / Year\n")
    f.write(f"  Annual OPEX                 : €{annual_opex / 1e6:,.2f} Million / Year\n")
    f.write(f"  Total Annual System Cost    : €{annual_total_cost / 1e9:,.3f} Billion / Year\n")
    f.write("-" * 80 + "\n")
    f.write(f"  Annual System Energy Demand : {annual_demand_mwh / 1e6:,.2f} TWh ({annual_demand_mwh:,.0f} MWh)\n")
    f.write(f"  Levelized Cost (LCOE)       : €{lcoe_eur_mwh_npv:,.2f} / MWh\n")
    f.write(f"  Annual Projected Revenue    : €{annual_revenue / 1e9:,.3f} Billion / Year\n")
    f.write(f"  Annual Net Operational CF   : €{annual_net_cash_flow / 1e9:,.3f} Billion / Year\n")
    f.write("-" * 80 + "\n")
    f.write(f"  NET PRESENT VALUE (NPV)     : €{final_npv / 1e9:,.3f} Billion\n")
    if pd.notna(project_irr):
        f.write(f"  INTERNAL RATE OF RETURN(IRR): {project_irr:.2f} %\n")
    else:
        f.write("  INTERNAL RATE OF RETURN(IRR): N/A\n")
    if pd.notna(payback_years):
        f.write(f"  PAYBACK PERIOD              : {payback_years:.2f} Years\n")
    else:
        f.write("  PAYBACK PERIOD              : N/A\n")
    f.write("-" * 80 + "\n")
    f.write("Notes:\n")
    f.write(" - Payback Period [Years] = Overnight CAPEX / Annual Net Operational Cash Flow.\n")
    f.write("=" * 80 + "\n")

print(f"  Estimated Overnight CAPEX : €{overnight_capex / 1e9:,.3f} Billion")
print(f"  25-Year Net Present Value : €{final_npv / 1e9:,.3f} Billion")
if pd.notna(project_irr):
    print(f"  Project IRR               : {project_irr:.2f} %")
if pd.notna(payback_years):
    print(f"  Payback Period            : {payback_years:.2f} Years")
print(f"  Saved Financial Summary    -> {npv_txt_path}")

# 6. Plot 25-Year Cumulative Cash Flow & NPV Progression Line Chart
fig, ax = plt.subplots(figsize=(10, 5.5))

ax.plot(years, cumulative_npv / 1e9, color="#0284c7", linewidth=2.5, marker="o", markersize=4, label="Cumulative Discounted NPV")
ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

# Highlight Year 0 Outlay
ax.scatter(0, cash_flows[0] / 1e9, color="#dc2626", s=60, zorder=5, label="Initial Overnight CAPEX")

ax.set_xlabel("Project Year", fontsize=10)
ax.set_ylabel("Net Present Value [Billion EUR]", fontsize=10)
ax.set_title(f"PyPSA-Libya — 25-Year Discounted Cash Flow & NPV Profile (WACC = {DISCOUNT_RATE*100:.0f}%)", fontsize=11, fontweight="bold")
ax.set_xticks(np.arange(0, PROJECT_LIFETIME + 1, 2))

ax.set_axisbelow(True)
ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
ax.legend(loc="lower right", fontsize=9)

# Value Callout Box for Cumulative NPV
npv_label_billion = f"Final NPV: €{final_npv / 1e9:,.2f}B"
if pd.notna(project_irr):
    npv_label_billion += f"\nIRR: {project_irr:.1f}%"

ax.annotate(
    npv_label_billion,
    xy=(PROJECT_LIFETIME, final_npv / 1e9),
    xytext=(-65, 25),
    textcoords="offset points",
    fontsize=9,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#e0f2fe", edgecolor="#0284c7", alpha=0.9),
    arrowprops=dict(arrowstyle="->", color="#0284c7", lw=1.2)
)

fig.tight_layout()
npv_png_path = os.path.join(OUT_DIR, "npv_cash_flow_trajectory.png")
fig.savefig(npv_png_path, dpi=150, bbox_inches="tight")
save_paper_pdf(fig, ax, "npv_cash_flow_trajectory")
plt.close(fig)
print(f"  Saved Financial NPV Chart  -> {npv_png_path}")
    



# =====================================================================
# 8. ELECTRICITY-ONLY COST COMPUTATION & COMPARISON (LCO2 vs NoLCO2)
# =====================================================================

HOURS_STR = f"{HOURS}h"

#### 1. Candidate paths construction helper for both cases

def build_elec_candidates(case_key, clusters, hours_str, run_name):
    folder_name = "No_LCO2" if case_key in ["NoLCO2", "No_LCO2"] else "LCO2"
 
    filename = f"elec_s_{clusters}_ec_lcopt_Co2L-{hours_str}.nc"
 
    return [
        os.path.join("Output", run_name, folder_name, hours_str, "networks", filename),
        os.path.join("Output", run_name, case_key, hours_str, "networks", filename),
        os.path.join("viz_outputs", f"libya-year-{clusters}clusters", case_key, hours_str, "networks", filename),
        os.path.join("viz_outputs", f"libya-year-{clusters}clusters", folder_name, hours_str, "networks", filename),
        rf"C:\Users\Mhdella\Documents\pypsa-libya\Output\{run_name}\{folder_name}\{hours_str}\networks\{filename}",
        rf"C:\Users\Mhdella\Documents\pypsa-libya\viz_outputs\libya-year-{clusters}clusters\{folder_name}\{hours_str}\networks\{filename}",
    ]


# Mappings and Color Palette
CAPEX_CARRIER_MAP_ELEC = {
    "solar": "Solar PV",
    "onwind": "Onshore Wind",
    "offwind": "Offshore Wind",
    "CCGT": "CCGT",
    "OCGT": "OCGT",
    "oil": "Oil",
    "heavy fuel oil": "Oil",
    "battery": "Battery Storage",
    "H2 Fuel Cell": "H2 Fuel Cell",
}

CHART_COLORS_ELEC = {
    "Solar PV": "#f5a623",
    "Onshore Wind": "#2ecc71",
    "Offshore Wind": "#1abc9c",
    "CCGT": "#7f8c8d",
    "OCGT": "#e74c3c",
    "Oil": "#8b4513",
    "Battery Storage": "#3498db",
    "Operating Cost": "#d2b48c",
}

# Cost Extraction Helper Functions
def capex_by_category_elec(network: pypsa.Network) -> pd.Series:
    records = []
    def clean_map(carrier_series):
        return carrier_series.astype(str).str.strip().map(lambda c: CAPEX_CARRIER_MAP_ELEC.get(c, c))

    if not network.generators.empty:
        g = network.generators
        cost = g.capital_cost * (g.p_nom_opt if "p_nom_opt" in g.columns else g.p_nom)
        records.append(pd.DataFrame({"category": clean_map(g.carrier), "cost": cost.values}))

    if not network.links.empty:
        l = network.links
        cost = l.capital_cost * (l.p_nom_opt if "p_nom_opt" in l.columns else l.p_nom)
        records.append(pd.DataFrame({"category": clean_map(l.carrier), "cost": cost.values}))

    if not network.stores.empty:
        s = network.stores
        cost = s.capital_cost * (s.e_nom_opt if "e_nom_opt" in s.columns else s.e_nom)
        records.append(pd.DataFrame({"category": clean_map(s.carrier), "cost": cost.values}))

    if not network.storage_units.empty:
        su = network.storage_units
        cost = su.capital_cost * (su.p_nom_opt if "p_nom_opt" in su.columns else su.p_nom)
        records.append(pd.DataFrame({"category": clean_map(su.carrier), "cost": cost.values}))

    if not records:
        return pd.Series(dtype=float)

    df_all = pd.concat(records, ignore_index=True)
    df_mapped = df_all[df_all["category"].isin(CHART_COLORS_ELEC.keys())]
    return df_mapped.groupby("category")["cost"].sum()

def operating_cost_elec(network: pypsa.Network) -> float:
    weights = network.snapshot_weightings.generators if hasattr(network, "snapshot_weightings") else 1.0
    total = 0.0
    if hasattr(network, "generators_t") and not network.generators_t.p.empty:
        mc = network.generators.marginal_cost
        total += (network.generators_t.p.mul(mc, axis=1).mul(weights, axis=0)).sum().sum()
    if hasattr(network, "links_t") and not network.links_t.p0.empty:
        mc = network.links.marginal_cost
        total += (network.links_t.p0.mul(mc, axis=1).mul(weights, axis=0)).sum().sum()
    return total


# --- STAGE 1: Initialize dictionary & dynamically derive cases ---
Dic_Costs_Components = {}

# Set primary case from global configuration variable
Temp_CASE = CASE

# Deriving complementary secondary case (assuming standard two-case model: LCO2 / NoLCO2)
Complement_CASE = "NoLCO2" if Temp_CASE in ["LCO2", "No_LCO2"] else "LCO2"
cases_to_run = [Temp_CASE, Complement_CASE]

print("\n" + "=" * 70)
print(f"Computing Electricity-Only System Costs (LCO2 vs NoLCO2) [{HOURS_STR}]")
print("=" * 70)

# --- STAGE 2: Execute First Run (Temp_CASE) and Second Run (Complement_CASE) ---
for current_run_case in cases_to_run:
    candidate_paths = build_elec_candidates(current_run_case, CLUSTERS, HOURS_STR, RUN_NAME)
    actual_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            actual_path = p
            break

    if not actual_path:
        print(f"  [SKIP] {current_run_case}: No solved network file found in target paths.")
        continue

    try:
        print(f"  Loading network for '{current_run_case}' from '{actual_path}'...")
        n_elec = pypsa.Network(actual_path)

        capex = capex_by_category_elec(n_elec)
        opex = operating_cost_elec(n_elec)

        combined = capex.copy()
        combined["Operating Cost"] = opex
        
        # Save run outputs into dictionary under the corresponding case key
        Dic_Costs_Components[current_run_case] = combined
    except Exception as e:
        print(f"  [ERROR] Failed to process network for {current_run_case}: {e}")


# --- STAGE 3: Aggregate dictionary outcomes, create 2-bar plot and 2-row txt summary ---

CASE_DISPLAY_NAMES = {
    "LCO2": "Limited CO2",
    "NoLCO2": "Unlimited CO2",
}


TARGET_OUT_DIR = OUT_DIR
os.makedirs(TARGET_OUT_DIR, exist_ok=True)

if not Dic_Costs_Components:
    print("  Notice: Could not generate comparison outputs because no network files were loaded.")
else:
    # Convert temporary dictionary to unified DataFrame in Billion EUR
    df_elec = pd.DataFrame(Dic_Costs_Components).T / 1e9
    df_elec = df_elec.fillna(0.0)

    # Ensure fixed row ordering (LCO2, NoLCO2) for consistent representation
    present_cases = [c for c in ["LCO2", "NoLCO2"] if c in df_elec.index]
    df_elec = df_elec.reindex(present_cases)

    # Apply display-name relabeling ONLY now, right before plotting/writing —
    # everything above this point still used the original internal keys.
    df_elec = df_elec.rename(index=CASE_DISPLAY_NAMES)

    category_order_elec = [c for c in ["Solar PV", "Onshore Wind", "Offshore Wind",
                                       "CCGT", "OCGT", "Oil", "Battery Storage", "Operating Cost"]
                          if c in df_elec.columns]
    df_elec = df_elec[category_order_elec]

    # Generate Output 1: Combined Figure with two stacked bars
    fig, ax = plt.subplots(figsize=(8, 6))
    bottom = pd.Series(0.0, index=df_elec.index)

    for cat in category_order_elec:
        values = df_elec[cat]
        ax.bar(df_elec.index, values, bottom=bottom,
               color=CHART_COLORS_ELEC.get(cat, "#95a5a6"), label=cat,
               edgecolor="black", linewidth=0.5, width=0.5)
        bottom += values

    ax.set_ylabel("Annualized System Cost [Billion EUR]", fontsize=11)
    ax.set_title(f"PyPSA-Libya: Electricity-Only System Cost Comparison ({HOURS_STR})\nLimited CO2 vs Unlimited CO2",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    out_png_path = os.path.join(TARGET_OUT_DIR, "fig_cost_by_scenario.png")
    fig.savefig(out_png_path, dpi=200, bbox_inches="tight")
    if "save_paper_pdf" in globals():
        save_paper_pdf(fig, ax, "figure9a_cost_by_scenario")
    plt.close(fig)
    print(f"  Saved comparison figure ({len(df_elec)} bars) -> {out_png_path}")

    # Generate Output 2: Text Breakdown File with two rows (one for each case)
    txt_out_path = os.path.join(TARGET_OUT_DIR, "Breakdown_cost_summary.txt")
    df_totals = df_elec.sum(axis=1)

    with open(txt_out_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"PyPSA-Libya: Electricity-Only Cost Breakdown Summary ({HOURS_STR}) (Billion EUR/year)\n")
        f.write("=" * 100 + "\n\n")
        f.write("--- Itemized Cost Breakdown (Billion EUR/year) ---\n")
        f.write(df_elec.round(4).to_string())
        f.write("\n\n" + "-" * 100 + "\n")
        f.write("--- Total System Costs Summary ---\n")
        for case_k, total_val in df_totals.items():
            f.write(f"  {case_k:14s} Total Cost: {total_val:8.3f} Billion EUR/year\n")
        f.write("=" * 100 + "\n")

    print(f"  Saved text breakdown ({len(df_elec)} rows) -> {txt_out_path}")



# # =====================================================================
# PART 9 — OPEX BREAKDOWN ANALYSIS (STACKED BAR CHART - BOTH SCENARIOS)
# =====================================================================

print("\n" + "=" * 70)
print(f"PART 9: Computing OPEX Breakdown Analysis (LCO2 vs NoLCO2) [{HOURS_STR}]")
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

# Unified Color Palette for OPEX Components
OPEX_CHART_COLORS = {
    "Gas": "#7f8c8d",                 # Grey/Slate fuel color
    "Gas Supply": "#7f8c8d",          # Grey/Slate fuel color
    "Oil": "#8b4513",                 # Brown
    "Solar PV": "#f5a623",            # Yellow/Orange
    "Onshore Wind": "#2ecc71",        # Light Green
    "Offshore Wind": "#1abc9c",       # Dark Teal
    "Electrolysis": "#7b2d8e",        # Purple
    "SMR": "#4f4f2f",                 # Olive/Dark Grey
    "SMR with CC": "#1a5276",         # Dark Blue
    "H2 Storage": "#3498db",          # Bright Blue
    "H2 Fuel Cell": "#e67e22",        # Orange
    "CCGT": "#95a5a6",                # Light Grey
    "OCGT": "#e74c3c",                # Soft Red
    "CO2 sequestration": "#0b1f3a",   # Navy Blue
    "H2 pipeline": "#c0392b",         # Red
    "CO2 Tax": "#000000",             # Black
    "Grid Expansion": "#bdc3c7",      # Platinum Grey
    "Process Emissions": "#d35400",   # Rust
    "Load Shedding": "#e74c3c",       # Red
    "Haber-Bosch": "#16a085"          # Green Teal
}


def opex_by_category(network: pypsa.Network) -> pd.Series:
    """Calculates OPEX grouped by carrier category incorporating marginal operational
    costs, primary fuel supply costs (Gas/Oil), and component dispatches.
    """
    records = []
    weights = network.snapshot_weightings.generators if hasattr(network, "snapshot_weightings") else 1.0

    def clean_map(carrier_series):
        return carrier_series.astype(str).str.strip().map(lambda c: OPEX_CARRIER_MAP.get(c, c))

    # 1. Generators OPEX (Primary Fuel Supply Generators like Gas/Oil)
    if not network.generators.empty and hasattr(network, "generators_t") and not network.generators_t.p.empty:
        g = network.generators
        sel_g = [c for c in g.index if c in network.generators_t.p.columns]
        if sel_g:
            mc = g.loc[sel_g, "marginal_cost"]
            cost = network.generators_t.p[sel_g].mul(mc, axis=1).mul(weights, axis=0).sum(axis=0)
            records.append(pd.DataFrame({
                "category": clean_map(g.loc[sel_g, "carrier"]),
                "cost": cost.values
            }))

    # 2. Links OPEX (Marginal Cost x Primary Input Power p0)
    if not network.links.empty and hasattr(network, "links_t") and not network.links_t.p0.empty:
        l = network.links
        sel_l = [c for c in l.index if c in network.links_t.p0.columns]
        if sel_l:
            mc = l.loc[sel_l, "marginal_cost"]
            cost = network.links_t.p0[sel_l].mul(mc, axis=1).mul(weights, axis=0).sum(axis=0)
            records.append(pd.DataFrame({
                "category": clean_map(l.loc[sel_l, "carrier"]),
                "cost": cost.values
            }))

    # 3. Stores OPEX
    if not network.stores.empty and hasattr(network, "stores_t") and hasattr(network.stores_t, "p") and not network.stores_t.p.empty:
        s = network.stores
        sel_s = [c for c in s.index if c in network.stores_t.p.columns]
        if sel_s and "marginal_cost" in s.columns:
            mc = s.loc[sel_s, "marginal_cost"]
            cost = network.stores_t.p[sel_s].mul(mc, axis=1).mul(weights, axis=0).sum(axis=0)
            records.append(pd.DataFrame({
                "category": clean_map(s.loc[sel_s, "carrier"]),
                "cost": cost.values
            }))

    # 4. Storage Units OPEX
    if not network.storage_units.empty and hasattr(network, "storage_units_t") and not network.storage_units_t.p.empty:
        su = network.storage_units
        sel_su = [c for c in su.index if c in network.storage_units_t.p.columns]
        if sel_su and "marginal_cost" in su.columns:
            mc = su.loc[sel_su, "marginal_cost"]
            cost = network.storage_units_t.p[sel_su].clip(lower=0).mul(mc, axis=1).mul(weights, axis=0).sum(axis=0)
            records.append(pd.DataFrame({
                "category": clean_map(su.loc[sel_su, "carrier"]),
                "cost": cost.values
            }))

    if not records:
        return pd.Series(dtype=float)

    df_all = pd.concat(records, ignore_index=True)
    return df_all.groupby("category")["cost"].sum()


# Process OPEX for both LCO2 and NoLCO2 cases
opex_scenario_costs = {}

for current_run_case in cases_to_run:
    candidate_paths = build_elec_candidates(current_run_case, CLUSTERS, HOURS_STR, RUN_NAME)
    actual_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            actual_path = p
            break

    if not actual_path:
        print(f"  [SKIP OPEX] {current_run_case}: No solved network file found.")
        continue

    try:
        print(f"  Extracting OPEX for '{current_run_case}' from '{actual_path}'...")
        n_opex = pypsa.Network(actual_path)
        opex_breakdown = opex_by_category(n_opex)
        
        # Optional: Add CO2 Tax if defined globally
        co2_tax_rate = globals().get("CO2_TAX_EUR_PER_TON", 0.0)
        if co2_tax_rate > 0:
            vented_mt = globals().get("compute_vented_co2_mt", lambda net: 0.0)(n_opex)
            co2_tax = vented_mt * 1e6 * co2_tax_rate
            if co2_tax > 0:
                opex_breakdown["CO2 Tax"] = co2_tax

        opex_scenario_costs[current_run_case] = opex_breakdown
    except Exception as e:
        print(f"  [ERROR OPEX] Failed to process network for {current_run_case}: {e}")

if opex_scenario_costs:
    # 1. Base Unfiltered DataFrame (in Billion EUR)
    df_opex_orig = pd.DataFrame(opex_scenario_costs).T / 1e9
    df_opex_orig = df_opex_orig.fillna(0.0)

    # Ensure fixed row ordering (LCO2, NoLCO2) and update display labels
    present_cases_opex = [c for c in ["LCO2", "NoLCO2"] if c in df_opex_orig.index]
    df_opex_orig = df_opex_orig.reindex(present_cases_opex)
    df_opex_orig = df_opex_orig.rename(index=CASE_DISPLAY_NAMES)

    # 2. Extract top 5 highest shared OPEX components using the UNFILTERED DataFrame
    top_5_cols = df_opex_orig.max(axis=0).nlargest(5).index.tolist()
    df_opex_top5 = df_opex_orig[top_5_cols]

    # 3. Create filtered DataFrame for plotting and rest of processing (threshold >= 0.001)
    df_opex = df_opex_orig.copy()
    significant_cols = df_opex.columns[df_opex.max(axis=0) >= 0.001]
    if len(significant_cols) > 0:
        df_opex = df_opex[significant_cols]

    preferred_order = [
        "Gas", "Gas Supply", "Oil", "SMR", "H2 Storage", "Grid Expansion",
        "Process Emissions", "Load Shedding", "Haber-Bosch",
        "Solar PV", "Onshore Wind", "Offshore Wind", "CCGT", "OCGT",
        "Electrolysis", "H2 Fuel Cell", "SMR with CC", "CO2 sequestration",
        "H2 pipeline", "CO2 Tax"
    ]
    
    category_order_opex = [c for c in preferred_order if c in df_opex.columns]
    for col in df_opex.columns:
        if col not in category_order_opex:
            category_order_opex.append(col)

    df_opex = df_opex[category_order_opex]

    # Stacked Bar Chart Generation
    fig_opex, ax_opex = plt.subplots(figsize=(8, 6))
    bottom_opex = pd.Series(0.0, index=df_opex.index)

    for cat in category_order_opex:
        values = df_opex[cat]
        ax_opex.bar(
            df_opex.index, values, bottom=bottom_opex,
            color=OPEX_CHART_COLORS.get(cat, "#7f8c8d"), label=cat,
            edgecolor="black", linewidth=0.5, width=0.5
        )
        bottom_opex += values

    ax_opex.set_ylabel("Annual Operational Cost [Billion EUR]", fontsize=11)
    ax_opex.set_title(f"PyPSA-Libya: Operational Expenditure (OPEX) Breakdown Comparison ({HOURS_STR})\nLimited CO2 vs Unlimited CO2", fontsize=11, fontweight="bold")
    ax_opex.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    ax_opex.grid(axis="y", linestyle="--", alpha=0.5)

    fig_opex.tight_layout()

    # Save Figure PNG & PDF
    out_path_opex = os.path.join(OUT_DIR, "fig_opex_cost_by_scenario.png")
    fig_opex.savefig(out_path_opex, dpi=200, bbox_inches="tight")
    if "save_paper_pdf" in globals():
        save_paper_pdf(fig_opex, ax_opex, "figure9d_opex_cost_by_scenario")
    plt.close(fig_opex)
    print(f"  Saved OPEX breakdown chart -> {out_path_opex}")

    # Save Summary Text File using df_opex_orig for totals and top 5
    txt_out_path_opex = os.path.join(OUT_DIR, "Breakdown_opex_cost_summary.txt")
    df_opex_totals = df_opex_orig.sum(axis=1)

    with open(txt_out_path_opex, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"PyPSA-Libya: Operational Expenditure (OPEX) Breakdown Summary ({HOURS_STR}) (Billion EUR/year)\n")
        f.write("=" * 100 + "\n\n")
        f.write("--- Itemized OPEX Breakdown (Billion EUR/year) ---\n")
        f.write(df_opex.round(4).to_string())
        f.write("\n\n" + "-" * 100 + "\n")
        f.write("--- Top 5 Shared OPEX Components (Billion EUR/year) ---\n")
        f.write(df_opex_top5.round(4).to_string())
        f.write("\n\n" + "-" * 100 + "\n")
        f.write("--- Total OPEX Summary ---\n")
        for scenario_k, total_val in df_opex_totals.items():
            f.write(f"  {scenario_k:14s} Total OPEX: {total_val:8.3f} Billion EUR/year\n")
        f.write("=" * 100 + "\n")

    print(f"  Saved OPEX breakdown text report -> {txt_out_path_opex}\n")
else:
    print("  Notice: Could not generate OPEX breakdown because no network files were loaded.")



# =====================================================================
# FINAL PART: ORGANIZE TXT SUMMARIES FOR LATEX
# =====================================================================

print("\n" + "=" * 70)
print("Organizing TXT Summaries for Paper")
print("=" * 70)

import shutil

# Only scan OUT_DIR for this script as there is no carrier directory
folders_to_scan = [OUT_DIR] 
table_count = 0

for folder in folders_to_scan:
    if not os.path.exists(folder):
        continue
        
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        
        # Handle Tables (Copy .txt files containing 'summary' or 'bottlenecks' or 'marginal_prices')
        # This captures transmission_line_bottlenecks.txt even if it doesn't have "summary" in name
        target_keywords = ['summary', 'bottlenecks', 'marginal_prices', 'objective_cost']
        
        if file.lower().endswith(".txt") and any(key in file.lower() for key in target_keywords):
            try:
                shutil.copy(file_path, os.path.join(PAPER_DIR, file))
                print(f" Copied Table: {file}")
                table_count += 1
            except Exception as e:
                print(f" [ERROR] Could not copy {file}: {e}")

print(f"\nSummary of Export:")
print(f" - Paper-ready PDFs: Saved directly during script execution")
print(f" - Summary TXT files copied: {table_count}")
print(f" Output Location: ./{PAPER_DIR}/")
print("=" * 70)


# Replace sys.exit(0) with a control variable:
SOLVED_FILE_EXISTS = os.path.exists(SOLVED_PATH)

if not SOLVED_FILE_EXISTS:
    print("\n" + "!" * 75)
    print(f"[NOTICE] Solved network file does not exist at expected location:")
    print(f"         '{SOLVED_PATH}'")
    print(f"Skipping single-case visualization for CASE='{CASE}' and HOURS={HOURS}h.")
    print("!" * 75 + "\n")
    # Do NOT call sys.exit(0) here
