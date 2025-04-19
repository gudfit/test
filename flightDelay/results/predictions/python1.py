#!/usr/bin/env python3
"""
enhanced_flight_delay_map.py

Improvements:
– US state boundaries from Cartopy
– Curved flight paths using bezier curves
– Enhanced color gradient using custom colormap
– Improved styling with better fonts and layout
– Background map with proper geographic context
– Focus on top 10 busiest airports
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
from matplotlib.patheffects import withStroke
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

TOP_AIRPORTS = 10      # Focus on top N busiest airports
N_BINS = 5            # more bins for smoother transitions
SEED = 42             # reproducibility
# -------------------------------------------------------------------------

# Create random number generator
rng = np.random.RandomState(SEED)

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

# 4) Find the top N busiest airports ------------------------------------
# Count flights per airport (both origin and destination)
origin_counts = df['Origin'].value_counts()
dest_counts = df['Dest'].value_counts()
all_counts = origin_counts.add(dest_counts, fill_value=0)

# Get the top N busiest airports
top_airports = all_counts.nlargest(TOP_AIRPORTS).index.tolist()
print(f"ℹ️ Top {TOP_AIRPORTS} busiest airports: {', '.join(top_airports)}")

# Filter flights to only include those between top airports
df_top = df[(df['Origin'].isin(top_airports)) & (df['Dest'].isin(top_airports))]
print(f"ℹ️ {len(df_top)} flights between top {TOP_AIRPORTS} airports")

# 6) Aggregate flights by airport pairs ---------------------------------
print("ℹ️ Aggregating flights between airport pairs...")
# Group flights by Origin-Destination pairs
grouped = df_top.groupby(['Origin', 'Dest']).agg({
    'Predicted_Delay': 'mean',  # Average delay for color
    'Flight_Delay': 'mean',     # Actual delay for comparison
    'Origin': 'count'           # Count of flights for line thickness
}).rename(columns={'Origin': 'flight_count'}).reset_index()

# Sort by flight count to draw busiest routes on top
grouped = grouped.sort_values('flight_count')

print(f"ℹ️ Aggregated to {len(grouped)} unique airport pairs")

# 7) Create custom colormap for smoother transition --------------------
# Create a custom blue-white-red colormap for better transitions
colors_blue = plt.cm.Blues(np.linspace(0.3, 1, 128))
colors_red = plt.cm.Reds(np.linspace(0, 0.8, 128))
colors = np.vstack([colors_blue, colors_red])
custom_cmap = LinearSegmentedColormap.from_list("BlueWhiteRed", colors)

# 8) Generate curved flight paths for aggregated routes ---------------
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
    t = np.linspace(0, 1, 30)  # Reduced number of points for better performance
    
    # Quadratic Bezier formula
    lon_curve = (1-t)**2 * lon1 + 2*(1-t)*t * ctrl_lon + t**2 * lon2
    lat_curve = (1-t)**2 * lat1 + 2*(1-t)*t * ctrl_lat + t**2 * lat2
    
    return np.column_stack([lon_curve, lat_curve])

# Create curved paths for each airport pair
print("ℹ️ Generating curved flight paths...")
curved_paths = []
for _, row in tqdm(grouped.iterrows(), total=len(grouped)):
    # Get coordinates for origin and destination
    origin = row['Origin']
    dest = row['Dest']
    
    # Skip if we don't have coordinates for either airport
    if origin not in coords or dest not in coords:
        continue
        
    lat1, lon1 = coords[origin]
    lat2, lon2 = coords[dest]
    
    curved_path = create_curved_path(
        lon1, lat1, 
        lon2, lat2, 
        height_factor=0.15  # Adjust this for more/less curve
    )
    curved_paths.append(curved_path)

# Normalize colors based on delay values
vmin, vmax = grouped["Predicted_Delay"].min(), grouped["Predicted_Delay"].max()
norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
colors = custom_cmap(norm(grouped["Predicted_Delay"].values))

# 9) Set up figure with Cartopy for proper map projection --------------
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

# 10) Add curved flight lines ---------------------------------------------
print("ℹ️ Adding flight paths to map...")
for i, path in enumerate(curved_paths):
    if i >= len(grouped):  # Safety check
        continue
        
    row = grouped.iloc[i]
    color = colors[i]
    delay = row["Predicted_Delay"]
    
    # Line width based on flight count (log scale to prevent extremely thick lines)
    count = row["flight_count"]
    width = 0.8 + np.log1p(count) * 0.5
    
    # Convert coordinates to the map projection
    path_proj = projection.transform_points(
        ccrs.PlateCarree(), path[:, 0], path[:, 1]
    )[:, :2]
    
    # Get alpha based on importance
    alpha = min(0.9, 0.4 + 0.5 * (count / grouped["flight_count"].max()))
    
    # Add the path to the map
    ax.plot(path_proj[:, 0], path_proj[:, 1], 
            color=color, linewidth=width, alpha=alpha, 
            solid_capstyle='round', zorder=10)
    
    # Optional: Add flight count as text in the middle of the path
    if count > grouped["flight_count"].quantile(0.75):  # Only for busiest routes
        mid_idx = len(path) // 2
        mid_x, mid_y = path_proj[mid_idx]
        
        # Add small white background for better readability
        ax.text(mid_x, mid_y, f"{int(count)}", fontsize=7,
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                zorder=11)

# 11) Add airports with styled markers -----------------------------------
print("ℹ️ Adding airport markers...")
for code in top_airports:
    lat, lon = coords[code]
    
    # Convert coordinates to map projection
    x, y = projection.transform_point(lon, lat, ccrs.PlateCarree())
    
    # Count of flights from/to this airport
    flight_count = sum((grouped['Origin'] == code) | (grouped['Dest'] == code))
    
    # Get airport name if available (fallback to code)
    airport_name = code
    airport_row = iata_df[iata_df['iata'] == code]
    if not airport_row.empty and 'airport' in airport_row.columns:
        airport_name = airport_row['airport'].iloc[0]
        if isinstance(airport_name, str) and airport_name:
            airport_name = airport_name.split(" International")[0]  # Simplify long names
    
    # Marker size based on flight count
    size = max(100, min(500, flight_count * 20))
    
    # Draw airport marker with highlight effect
    ax.scatter(x, y, s=size, marker='o', 
               color='white', edgecolor='black', 
               linewidth=1.5, alpha=0.85, zorder=20)
    ax.scatter(x, y, s=size*0.7, marker='o', 
               color='#f0f0f0', edgecolor='#999999', 
               linewidth=0.5, alpha=0.6, zorder=21)
    
    # Add airport code label
    ax.text(x, y-size/50, code, fontsize=11, 
            ha='center', va='center', 
            fontweight='bold', color='#333333', zorder=22,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))
    
    # Add smaller airport name below
    if code != airport_name:
        ax.text(x, y+size/50, airport_name, fontsize=8,
                ha='center', va='center', color='#666666', zorder=22,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

# 12) Add title and colorbar ---------------------------------------------
ax.set_title(f"Flight Connections Between Top {TOP_AIRPORTS} U.S. Airports", 
             fontsize=24, pad=20, fontweight='bold')

# Add subtitle with data info
unique_pairs = len(grouped)
total_flights = grouped["flight_count"].sum()
subtitle = f"Showing {unique_pairs} routes with {int(total_flights)} total flights | Line thickness = flight frequency | Color = average delay"
ax.text(0.5, 0.97, subtitle, 
        transform=fig.transFigure, ha='center', fontsize=12, 
        color='#555555')

# Add colorbar
cax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
sm = ScalarMappable(norm=norm, cmap=custom_cmap)
sm.set_array([])
cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
cb.set_label('Average Predicted Delay (minutes)', fontsize=14, weight='bold', labelpad=10)
cb.ax.tick_params(labelsize=12)

# Add custom labels for early vs late
cb.ax.text(0.1, -2.5, 'EARLY ◀', ha='center', va='center', 
           transform=cb.ax.transAxes, fontsize=12, fontweight='bold', color='#3377AA')
cb.ax.text(0.9, -2.5, '▶ LATE', ha='center', va='center', 
           transform=cb.ax.transAxes, fontsize=12, fontweight='bold', color='#AA3333')

# 13) Add legend for line thickness ------------------------------------
# Create custom handles for the legend
legend_counts = [10, 100, 500, 1000]
legend_handles = []
legend_labels = []

for count in legend_counts:
    width = 0.8 + np.log1p(count) * 0.5
    line = plt.Line2D([0], [0], color='gray', linewidth=width, alpha=0.7)
    legend_handles.append(line)
    legend_labels.append(f"{count} flights")

# Place the legend below the colorbar
leg = fig.legend(legend_handles, legend_labels, 
                 loc='lower center', 
                 bbox_to_anchor=(0.5, 0.02),
                 ncol=len(legend_counts),
                 title="Line Thickness = Flight Frequency",
                 frameon=True,
                 fancybox=True,
                 shadow=True)
leg.get_title().set_fontsize(10)

# 14) Add footnote and credit info --------------------------------------
plt.figtext(0.02, 0.02, "Data source: flight delay predictions", 
            fontsize=8, color='#666666')
plt.figtext(0.98, 0.02, "Enhanced visualization", 
            fontsize=8, color='#666666', ha='right')

# Adjust layout and save
plt.savefig(OUT_PNG, bbox_inches='tight', dpi=300)
print(f"✅ Enhanced map saved to '{OUT_PNG}'")
