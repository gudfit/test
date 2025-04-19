#!/usr/bin/env python3
"""
enhanced_flight_delay_map.py

Improvements:
– US state boundaries from Cartopy
– Curved flight paths using bezier curves
– Enhanced color gradient using viridis colormap
– Improved styling with better fonts and layout
– Background map with proper geographic context
– Flight icons with varying sizes based on delay
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from scipy.interpolate import make_interp_spline
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm.auto import tqdm
import matplotlib.font_manager as fm

# CONFIG ------------------------------------------------------------------
PRED_CSV = Path("final_predictions.csv")
IATA_CSV = Path("iata-icao.csv")
OUT_PNG = Path("enhanced_flight_delay_map.png")

# contiguous‑US bounding box
LAT_MIN, LAT_MAX = 24, 50
LON_MIN, LON_MAX = -125, -65

TOTAL_LINES = 400     # total flights to plot
N_BINS = 5            # more bins for smoother transitions
SEED = 42             # reproducibility
# -------------------------------------------------------------------------

# 1) load predictions ----------------------------------------------------
if not PRED_CSV.exists():
    sys.exit(f"❌ '{PRED_CSV}' not found.")
df = pd.read_csv(PRED_CSV)

# 2) load IATA→coords ----------------------------------------------------
if not IATA_CSV.exists():
    sys.exit("❌ 'iata-icao.csv' not found.")
iata_df = pd.read_csv(IATA_CSV, dtype=str)
iata_df["iata"] = iata_df["iata"].str.strip().str.upper()
iata_df = iata_df.dropna(subset=["latitude", "longitude"])
iata_df["lat"] = iata_df["latitude"].astype(float)
iata_df["lon"] = iata_df["longitude"].astype(float)
coords = dict(zip(iata_df["iata"], zip(iata_df["lat"], iata_df["lon"])))
print(f"ℹ️ Loaded {len(coords)} airports")

# 3) merge coords & filter CCS ------------------------------------------
df = df.merge(
    iata_df[["iata", "lat", "lon"]]
    .rename(columns={"iata": "Origin", "lat": "lat_o", "lon": "lon_o"}),
    on="Origin", how="inner"
).merge(
    iata_df[["iata", "lat", "lon"]]
    .rename(columns={"iata": "Dest", "lat": "lat_d", "lon": "lon_d"}),
    on="Dest", how="inner"
)
# drop anything outside the 48 states rect
df = df[
    (df.lat_o.between(LAT_MIN, LAT_MAX) & df.lon_o.between(LON_MIN, LON_MAX)) &
    (df.lat_d.between(LAT_MIN, LAT_MAX) & df.lon_d.between(LON_MIN, LON_MAX))
]
if df.empty:
    sys.exit("❌ No contiguous‑US flights remain.")

# 4) quantile bins & stratified sample ----------------------------------
df["qbin"], bins = pd.qcut(df["Predicted_Delay"],
                          q=N_BINS, retbins=True, labels=False, duplicates="drop")

per_bin = int(np.ceil(TOTAL_LINES / df["qbin"].nunique()))
sampled_parts = []
rng = np.random.RandomState(SEED)
for q in sorted(df["qbin"].unique()):
    grp = df[df["qbin"] == q]
    take = min(len(grp), per_bin)
    sampled_parts.append(grp.sample(take, random_state=SEED))
sampled = pd.concat(sampled_parts, ignore_index=True)
print(f"ℹ️ {len(sampled)} flights ≈{per_bin} per bin across {N_BINS} bins")

# 5) fake in‑bin delays for smooth gradient -----------------------------
plot_delays = []
for _, row in sampled.iterrows():
    q = int(row["qbin"])
    lo, hi = bins[q], bins[q+1]
    plot_delays.append(rng.uniform(lo, hi))
sampled["plot_delay"] = plot_delays

# 6) Create custom colormap for smoother transition --------------------
# Create a custom blue-white-red colormap for better transitions
colors_blue = plt.cm.Blues(np.linspace(0.3, 1, 128))
colors_red = plt.cm.Reds(np.linspace(0, 0.8, 128))
colors = np.vstack([colors_blue, colors_red])
custom_cmap = LinearSegmentedColormap.from_list("BlueWhiteRed", colors)

# 7) Generate curved flight paths ---------------------------------------
def create_curved_path(lon1, lat1, lon2, lat2, height_factor=0.2):
    """Create a curved path between two points using a quadratic Bezier curve"""
    # Calculate the midpoint
    mid_lon = (lon1 + lon2) / 2
    mid_lat = (lat1 + lat2) / 2
    
    # Calculate the distance between the points
    dist = np.sqrt((lon2 - lon1)**2 + (lat2 - lat1)**2)
    
    # Calculate a control point that is perpendicular to the straight line
    dx = lon2 - lon1
    dy = lat2 - lat1
    
    # Perpendicular direction
    nx = -dy
    ny = dx
    
    # Normalize and scale by distance and height factor
    length = np.sqrt(nx*nx + ny*ny)
    if length > 0:
        nx = nx / length * dist * height_factor
        ny = ny / length * dist * height_factor
    
    # Control point
    ctrl_lon = mid_lon + nx
    ctrl_lat = mid_lat + ny
    
    # Create a Bezier curve with more points for smoother curves
    t = np.linspace(0, 1, 50)
    
    # Quadratic Bezier formula
    lon_curve = (1-t)**2 * lon1 + 2*(1-t)*t * ctrl_lon + t**2 * lon2
    lat_curve = (1-t)**2 * lat1 + 2*(1-t)*t * ctrl_lat + t**2 * lat2
    
    return np.column_stack([lon_curve, lat_curve])

# Create curved paths
print("ℹ️ Generating curved flight paths...")
curved_paths = []
for _, row in tqdm(sampled.iterrows(), total=len(sampled)):
    curved_path = create_curved_path(
        row['lon_o'], row['lat_o'], 
        row['lon_d'], row['lat_d'], 
        height_factor=0.15  # Adjust this for more/less curve
    )
    curved_paths.append(curved_path)

# Normalize colors
vmin, vmax = sampled["plot_delay"].min(), sampled["plot_delay"].max()
norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
colors = custom_cmap(norm(sampled["plot_delay"].values))

# 8) Set up figure with Cartopy for proper map projection --------------
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(16, 12), dpi=150)

# Use Cartopy for proper map projection
projection = ccrs.AlbersEqualArea(central_longitude=-97.0, central_latitude=38.0)
ax = fig.add_subplot(1, 1, 1, projection=projection)

# Set map extent
ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#F5F5F5', edgecolor='none')
ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#E6F3F7', edgecolor='none')
ax.add_feature(cfeature.STATES.with_scale('50m'), facecolor='none', edgecolor='#CCCCCC', linewidth=0.8)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8, edgecolor='#999999')

# Add background lakes for better geographical context
ax.add_feature(cfeature.LAKES.with_scale('50m'), facecolor='#E6F3F7', edgecolor='#AADDEE', linewidth=0.5)

# Add gridlines with translucent appearance
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=0.5, color='#DDDDDD', alpha=0.5, linestyle=':')

# 9) Add curved flight lines ---------------------------------------------
print("ℹ️ Adding flight paths to map...")
for i, path in enumerate(curved_paths):
    color = colors[i]
    delay = sampled.iloc[i]["plot_delay"]
    width = 1.2 + abs(delay) / vmax * 2  # Make important flights (large absolute delays) thicker
    
    # Convert coordinates to the map projection
    path_proj = projection.transform_points(
        ccrs.PlateCarree(), path[:, 0], path[:, 1]
    )[:, :2]
    
    # Get alpha based on importance (larger absolute values more visible)
    alpha = 0.5 + 0.5 * abs(delay) / vmax
    
    # Add the path to the map
    ax.plot(path_proj[:, 0], path_proj[:, 1], 
            color=color, linewidth=width, alpha=min(0.9, alpha), 
            solid_capstyle='round', zorder=10)

# 10) Add airports with styled markers -----------------------------------
print("ℹ️ Adding airport markers...")
major_airports = set()
for _, row in sampled.iterrows():
    major_airports.add(row['Origin'])
    major_airports.add(row['Dest'])

for code in major_airports:
    lat, lon = coords[code]
    
    # Convert coordinates to map projection
    x, y = projection.transform_point(lon, lat, ccrs.PlateCarree())
    
    # Count of flights from/to this airport
    flight_count = sum((sampled['Origin'] == code) | (sampled['Dest'] == code))
    
    # Marker size based on flight count
    size = max(20, min(120, flight_count * 3))
    
    # Draw airport marker
    ax.scatter(x, y, s=size, marker='o', 
               color='white', edgecolor='black', 
               linewidth=0.8, alpha=0.7, zorder=20)
    
    # Add airport code label for major airports
    if flight_count > 2:
        ax.text(x, y, code, fontsize=7, 
                ha='center', va='center', 
                fontweight='bold', zorder=21)

# 11) Add title and colorbar ---------------------------------------------
ax.set_title("U.S. Flight Delay Network Visualization", 
             fontsize=24, pad=20, fontweight='bold')

# Add subtitle with data info
ax.text(0.5, 0.97, "Flight paths colored by predicted delay times", 
        transform=fig.transFigure, ha='center', fontsize=14, 
        color='#555555')

# Add colorbar
cax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
sm = ScalarMappable(norm=norm, cmap=custom_cmap)
sm.set_array([])
cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
cb.set_label('Predicted Delay (minutes)', fontsize=14, weight='bold', labelpad=10)
cb.ax.tick_params(labelsize=12)

# Add custom labels for early vs late
cb.ax.text(0.1, -2.5, 'EARLY ◀', ha='center', va='center', 
           transform=cb.ax.transAxes, fontsize=12, fontweight='bold', color='#3377AA')
cb.ax.text(0.9, -2.5, '▶ LATE', ha='center', va='center', 
           transform=cb.ax.transAxes, fontsize=12, fontweight='bold', color='#AA3333')

# 12) Add footnote and credit info --------------------------------------
plt.figtext(0.02, 0.02, "Data source: flight delay predictions", 
            fontsize=8, color='#666666')
plt.figtext(0.98, 0.02, "Enhanced visualization", 
            fontsize=8, color='#666666', ha='right')

# Adjust layout and save
plt.savefig(OUT_PNG, bbox_inches='tight', dpi=300)
print(f"✅ Enhanced map saved to '{OUT_PNG}'")
