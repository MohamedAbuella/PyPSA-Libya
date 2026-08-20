import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pypsa

# ------------------------------------------------------------------------------
# BUG FIX: bus names look like "LY0 25" (electricity) or "LY0 25 H2" (paired
# hydrogen bus) — the base electricity bus ID itself contains a space
# (country/subnetwork + cluster number).
# ------------------------------------------------------------------------------
KNOWN_CARRIER_SUFFIXES = [
    " H2", " heat", " gas", " biogas", " oil", " coal", " lignite",
    " urban central heat", " urban decentral heat", " rural heat",
    " low-temperature heat for industry", " high-temperature heat for industry"
]

def strip_carrier_suffix(bus_name: str) -> str:
    for suffix in KNOWN_CARRIER_SUFFIXES:
        if bus_name.endswith(suffix):
            return bus_name[: -len(suffix)]
    return bus_name

# ==============================================================================
# 1. CONFIGURATION & PATH SETUP
# ==============================================================================

POSTNETWORK_PATH = r"C:\Users\Mhdella\Documents\pypsa-libya\Output\libya_h2export_test\postnetworks\elec_s_36_ec_lcopt_Co2L-6h_144h_2030_0.071_AB_70export.nc"

# POSTNETWORK_PATH = r"C:\Users\Mhdella\Documents\pypsa-libya\Output\libya_h2export_test\postnetworks\elec_s_36_ec_lcopt_Co2L-3h_144h_2030_0.071_AB_0.25export.nc"
# POSTNETWORK_PATH = r"C:\Users\Mhdella\Documents\pypsa-libya\Output\libya_h2export_test\postnetworks\elec_s_36_ec_lcopt_3h_144h_2030_0.071_AB_0.25export.nc"

OUT_DIR = r"viz_outputs\libya-h2export-test_elec_s_15"

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print(f"RUN CONFIGURATION: Target Network = '{POSTNETWORK_PATH}'")
print(f"Output Directory: '{OUT_DIR}'")
print("=" * 70)

# Carrier Color Palette
CARRIER_COLORS = {
    "solar": "#f59e0b",
    "solar rooftop": "#fcd34d",
    "onwind": "#10b981",
    "offwind-ac": "#06b6d4",
    "offwind-dc": "#0891b2",
    "OCGT": "#ef4444",
    "CCGT": "#b91c1c",
    "gas": "#dc2626",
    "oil": "#78350f",
    "H2": "#3b82f6",
    "H2 Fuel Cell": "#1d4ed8",
    "H2 Electrolysis": "#60a5fa",
    "battery": "#8b5cf6",
    "battery charger": "#a78bfa",
    "battery discharger": "#c084fc",
    "heat": "#f97316",
    "heat pump": "#fb923c",
    "resistive heater": "#fdba74",
    "biomass": "#15803d",
    "load": "#6b7280",
    "AC": "#64748b"
}

DEFAULT_COLOR = "#9ca3af"

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def safe_savefig(fig, filepath, dpi=200):
    """Saves figure safely by catching drawing errors or invalid paths."""
    try:
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        print(f"  [Saved] -> {filepath}")
    except Exception as e:
        print(f"  [Warning] Failed to save {os.path.basename(filepath)} due to error: {e}")
    finally:
        plt.close(fig)

def get_bus_coordinates(n):
    """Extract bus x, y coordinates ensuring float format and attach to main bus location."""
    buses = n.buses.copy()
    buses["x"] = pd.to_numeric(buses["x"], errors="coerce").fillna(0.0)
    buses["y"] = pd.to_numeric(buses["y"], errors="coerce").fillna(0.0)
    
    for idx, row in buses.iterrows():
        if row["x"] == 0.0 and row["y"] == 0.0:
            location = strip_carrier_suffix(idx)
            if location in buses.index and location != idx:
                buses.loc[idx, "x"] = buses.loc[location, "x"]
                buses.loc[idx, "y"] = buses.loc[location, "y"]
    return buses

# ==============================================================================
# 3. LOAD SOLVED NETWORK
# ==============================================================================
if not os.path.exists(POSTNETWORK_PATH):
    raise FileNotFoundError(f"Solved network file not found at: {POSTNETWORK_PATH}")

print(f"\nLoading solved network from: {POSTNETWORK_PATH}")
n = pypsa.Network(POSTNETWORK_PATH)

snapshots = n.snapshots
if len(snapshots) > 1:
    dt = (snapshots[1] - snapshots[0]).total_seconds() / 3600.0
    total_hours = len(snapshots) * dt
    print(f"Detected simulation window: {total_hours:.1f} hours ({total_hours/24:.1f} days) with {dt}h timestep resolution.")

# ==============================================================================
# 4. VISUALIZATIONS
# ==============================================================================

# ------------------------------------------------------------------------------
# 4.1 Plot Installed Capacity by Carrier (GW)
# ------------------------------------------------------------------------------
print("\n[1/7] Plotting Installed Capacity by Carrier...")
fig, ax = plt.subplots(figsize=(10, 6))
cap_by_carrier = pd.Series(dtype=float)

if not n.generators.empty:
    gen_cap = n.generators.groupby("carrier")["p_nom_opt"].sum()
    cap_by_carrier = pd.concat([cap_by_carrier, gen_cap])

if not n.storage_units.empty:
    su_cap = n.storage_units.groupby("carrier")["p_nom_opt"].sum()
    cap_by_carrier = pd.concat([cap_by_carrier, su_cap])

if not n.stores.empty and "e_nom_opt" in n.stores:
    store_cap = n.stores.groupby("carrier")["e_nom_opt"].sum()
    cap_by_carrier = pd.concat([cap_by_carrier, store_cap])

if not cap_by_carrier.empty:
    cap_by_carrier = cap_by_carrier.groupby(level=0).sum() / 1e3  # Convert to GW
    bar_colors = [CARRIER_COLORS.get(c, DEFAULT_COLOR) for c in cap_by_carrier.index]
    cap_by_carrier.sort_values().plot(kind="barh", ax=ax, color=bar_colors, edgecolor="black")
    ax.set_xlabel("Optimal Capacity (GW / GWh for Stores)")
    ax.set_ylabel("Carrier")
    ax.set_title("Solved Installed Capacity by Carrier (Libya Sector-Coupling)")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

safe_savefig(fig, os.path.join(OUT_DIR, "solved_capacity_by_carrier.png"))

# ------------------------------------------------------------------------------
# 4.2 Plot Spatial Capacity Map (FIXED PIE MAP & FALLBACK)
# ------------------------------------------------------------------------------
print("[2/7] Plotting Detailed Libya Spatial Capacity Map...")
fig, ax = plt.subplots(figsize=(13, 10), subplot_kw={"projection": ccrs.PlateCarree()})

# Base geography setup
ax.add_feature(cfeature.LAND, facecolor="#f8fafc")
ax.add_feature(cfeature.OCEAN, facecolor="#e0f2fe")
ax.add_feature(cfeature.COASTLINE, linewidth=1.0, color="#334155")
ax.add_feature(cfeature.BORDERS, linestyle="--", linewidth=0.8, color="#64748b")
ax.add_feature(cfeature.RIVERS, linewidth=0.5, color="#93c5fd")

buses = get_bus_coordinates(n)

# Filter coordinates focusing on Libya
libya_buses = buses[(buses["x"] >= 9.0) & (buses["x"] <= 25.5) & (buses["y"] >= 19.0) & (buses["y"] <= 33.5)]

if not libya_buses.empty:
    minx, maxx = 9.0, 25.5
    miny, maxy = 19.5, 33.5
else:
    minx, maxx = buses["x"].min() - 1.0, buses["x"].max() + 1.0
    miny, maxy = buses["y"].min() - 1.0, buses["y"].max() + 1.0

ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.PlateCarree())

# Collect Nodal Capacities by Carrier & Location
nodal_cap = pd.DataFrame(index=n.buses.index)

if not n.generators.empty:
    gen_p = n.generators.groupby(["bus", "carrier"])["p_nom_opt"].sum().unstack(fill_value=0.0)
    nodal_cap = nodal_cap.join(gen_p, how="left").fillna(0.0)

if not n.storage_units.empty:
    su_p = n.storage_units.groupby(["bus", "carrier"])["p_nom_opt"].sum().unstack(fill_value=0.0)
    nodal_cap = nodal_cap.join(su_p, how="left", lsuffix="_gen", rsuffix="_su").fillna(0.0)

# Map sub-buses back to primary node locations
nodal_cap["location"] = [strip_carrier_suffix(b) for b in nodal_cap.index]
nodal_cap_grouped = nodal_cap.groupby("location").sum()

# Consolidate duplicate carrier columns using pandas 2.0+ syntax
if not nodal_cap_grouped.empty:
    nodal_cap_grouped = nodal_cap_grouped.T.groupby(by=lambda x: str(x).split("_")[0]).sum().T

# Prepare bus_sizes DataFrame scaled for PyPSA pie charts across all buses
scale_factor = 2e-5
bus_sizes_df = nodal_cap_grouped.reindex(n.buses.index).fillna(0.0) * scale_factor

# Carrier Colors Mapping
carrier_cols = {c: CARRIER_COLORS.get(c, DEFAULT_COLOR) for c in bus_sizes_df.columns}

# Line and link widths
line_widths = pd.Series(0.0, index=n.lines.index)
if not n.lines.empty:
    s_nom = n.lines.get("s_nom_opt", n.lines.get("s_nom", 0.0))
    line_widths = pd.Series(np.where(s_nom > 10, s_nom, 0.0) / 1500.0, index=n.lines.index).fillna(0.0)

link_widths = pd.Series(0.0, index=n.links.index)
if not n.links.empty:
    p_nom = n.links.get("p_nom_opt", n.links.get("p_nom", 0.0))
    link_widths = pd.Series(np.where(p_nom > 10, p_nom, 0.0) / 1500.0, index=n.links.index).fillna(0.0)

try:
    n.plot(
        ax=ax,
        bus_sizes=bus_sizes_df,
        bus_colors=carrier_cols,
        line_widths=line_widths,
        link_widths=link_widths,
        line_colors="#475569",
        link_colors="#0284c7",
        title="Libya Sector-Coupled Solved Capacity & Grid Network",
        boundaries=[minx, maxx, miny, maxy]
    )
except Exception as e:
    print(f"  [Warning] Advanced pie map fallback applied: {e}")
    ac_buses = buses[buses.index.isin(nodal_cap_grouped.index)]
    ac_sizes = nodal_cap_grouped.sum(axis=1).reindex(ac_buses.index).fillna(0.0) * scale_factor * 1000.0
    ax.scatter(ac_buses["x"].values, ac_buses["y"].values, s=ac_sizes.values, c="#0284c7", alpha=0.7, transform=ccrs.PlateCarree())

# Custom Legend Construction
legend_handles = [
    Line2D([0], [0], color="#475569", lw=2, label="AC Lines"),
    Line2D([0], [0], color="#0284c7", lw=2, label="DC Links / Converters"),
]

present_carriers = [c for c in nodal_cap_grouped.columns if nodal_cap_grouped[c].sum() > 1.0]
for c in present_carriers:
    legend_handles.append(mpatches.Patch(color=CARRIER_COLORS.get(c, DEFAULT_COLOR), label=str(c).capitalize()))

ax.legend(handles=legend_handles, loc="lower left", fontsize=8, framealpha=0.9, facecolor="white")
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.3, linestyle=":")

safe_savefig(fig, os.path.join(OUT_DIR, "solved_capacity_map.png"))

# ------------------------------------------------------------------------------
# 4.3 Hourly Generation & Dispatch Timeseries (MW over Time)
# ------------------------------------------------------------------------------
print("[3/7] Plotting Hourly Generation & Dispatch Timeseries...")
try:
    fig, ax = plt.subplots(figsize=(14, 6))
    
    gen_p = n.generators_t.p.T.groupby(n.generators.carrier).sum().T / 1e3 # GW
    load_p = n.loads_t.p.sum(axis=1) / 1e3 # GW
    
    if not gen_p.empty:
        colors = [CARRIER_COLORS.get(c, DEFAULT_COLOR) for c in gen_p.columns]
        gen_p.plot(kind="area", ax=ax, stacked=True, alpha=0.85, color=colors, lw=0)
    
    if not load_p.empty:
        ax.plot(snapshots, load_p, color="black", linewidth=2.5, linestyle="--", label="Total Demand")
        
    ax.set_ylabel("Power (GW)")
    ax.set_xlabel("Time Step")
    ax.set_title("Hourly Power Balance & Dispatch Trajectory (Peak Period)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    safe_savefig(fig, os.path.join(OUT_DIR, "hourly_dispatch_timeseries.png"))
except Exception as e:
    print(f"  [Warning] Dispatch plot failed: {e}")

# ------------------------------------------------------------------------------
# 4.4 Total Energy Generation by Carrier (GWh)
# ------------------------------------------------------------------------------
print("[4/7] Plotting Total Energy Production by Carrier...")
try:
    fig, ax = plt.subplots(figsize=(10, 5))
    
    weights = n.snapshot_weightings.generators
    gen_energy = n.generators_t.p.mul(weights, axis=0).sum()
    energy_by_carrier = gen_energy.groupby(n.generators.carrier).sum() / 1e3  # GWh
    
    if not energy_by_carrier.empty:
        colors = [CARRIER_COLORS.get(c, DEFAULT_COLOR) for c in energy_by_carrier.index]
        energy_by_carrier.sort_values().plot(kind="barh", ax=ax, color=colors, edgecolor="black")
        ax.set_xlabel("Total Energy Supplied (GWh)")
        ax.set_ylabel("Carrier")
        ax.set_title("Total Energy Production by Carrier over Peak Window")
        ax.grid(axis="x", linestyle="--", alpha=0.7)
        
    safe_savefig(fig, os.path.join(OUT_DIR, "energy_production_by_carrier.png"))
except Exception as e:
    print(f"  [Warning] Energy production plot failed: {e}")

# ------------------------------------------------------------------------------
# 4.5 Storage State of Charge (SoC) over Time
# ------------------------------------------------------------------------------
print("[5/7] Plotting Storage State of Charge (SoC)...")
try:
    fig, ax = plt.subplots(figsize=(12, 5))
    has_storage_data = False
    
    if hasattr(n, "storage_units_t") and not n.storage_units_t.state_of_charge.empty:
        soc_su = n.storage_units_t.state_of_charge.T.groupby(n.storage_units.carrier).sum().T / 1e3 # GWh
        soc_su.plot(ax=ax, lw=2)
        has_storage_data = True
        
    if hasattr(n, "stores_t") and not n.stores_t.e.empty:
        soc_stores = n.stores_t.e.T.groupby(n.stores.carrier).sum().T / 1e3 # GWh
        soc_stores.plot(ax=ax, lw=2, linestyle="--")
        has_storage_data = True

    if has_storage_data:
        ax.set_ylabel("Stored Energy (GWh)")
        ax.set_xlabel("Time Step")
        ax.set_title("Storage Trajectory & State of Charge (SoC)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.6)
        safe_savefig(fig, os.path.join(OUT_DIR, "storage_state_of_charge.png"))
    else:
        print("  [Info] No active storage state-of-charge time-series found.")
        plt.close(fig)
except Exception as e:
    print(f"  [Warning] Storage plot failed: {e}")

# ------------------------------------------------------------------------------
# 4.6 Nodal Marginal Electricity Prices ($/MWh)
# ------------------------------------------------------------------------------
print("[6/7] Plotting Nodal Electricity Prices...")
try:
    if hasattr(n, "buses_t") and not n.buses_t.marginal_price.empty:
        prices = n.buses_t.marginal_price
        
        fig, ax = plt.subplots(figsize=(12, 5))
        prices.mean(axis=1).plot(ax=ax, color="crimson", lw=2, label="System Average Price")
        ax.fill_between(prices.index, prices.min(axis=1), prices.max(axis=1), color="crimson", alpha=0.2, label="Nodal Price Range")
        
        ax.set_ylabel("Marginal Price ($/MWh)")
        ax.set_xlabel("Time Step")
        ax.set_title("Nodal Electricity Marginal Price Trajectory")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.6)
        
        safe_savefig(fig, os.path.join(OUT_DIR, "nodal_marginal_prices.png"))
    else:
        print("  [Info] Marginal prices missing or non-converged.")
except Exception as e:
    print(f"  [Warning] Price plot failed: {e}")

# ------------------------------------------------------------------------------
# 4.7 Transmission Line & Link Power Flows
# ------------------------------------------------------------------------------
print("[7/7] Plotting Transmission Line & Link Power Flows...")
try:
    fig, ax = plt.subplots(figsize=(12, 5))
    
    if not n.lines_t.p0.empty:
        line_flows = n.lines_t.p0.abs().mean(axis=0) / 1e3 # GW
        ax.hist(line_flows, bins=20, color="teal", edgecolor="black", alpha=0.7)
        ax.set_xlabel("Average Active Power Flow (GW)")
        ax.set_ylabel("Number of AC Lines")
        ax.set_title("Distribution of Line Active Power Flows")
        ax.grid(True, linestyle=":", alpha=0.6)
        
        safe_savefig(fig, os.path.join(OUT_DIR, "line_power_flows.png"))
    else:
        plt.close(fig)
except Exception as e:
    print(f"  [Warning] Power flow plot failed: {e}")

print("\n" + "=" * 70)
print("Sector-Coupling Visualization Successfully Completed!")
print(f"All outputs saved to: '{OUT_DIR}'")
print("=" * 70)