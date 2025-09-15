# src/00_prep.py
"""
Preprocessing script to compute slope (degrees) from a DEM and attach slope values
to a CSV of points (latitude, longitude). This version:

- Reprojects geographic DEMs (EPSG:4326) to an appropriate UTM CRS (meters).
- Computes slope using pixel spacings in meters.
- Reprojects input lat/lon to the DEM CRS before sampling slope.
- Handles nodata (attempts to fill nodata where present).
- Saves updated CSV with a corrected 'slope' column.

Dependencies:
- rasterio
- numpy
- pandas
- pyproj

Usage:
    python src/00_prep.py
"""

import os
from typing import Tuple
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.fill import fillnodata
from pyproj import Transformer
from rasterio.crs import CRS

# -------------------------------
# Paths (change as needed)
# -------------------------------
DATA_PATH = "data/jharia_synthetic_dataset_balanced.csv"
DEM_PATH = "data/jharia_synthetic_dem_fixed.tif"
OUTPUT_PATH = "data/jharia_dataset_with_dem_fixed.csv"

# -------------------------------
# Utilities
# -------------------------------


def _choose_utm_crs_for_lonlat(lon: float, lat: float) -> CRS:
    """
    Choose an appropriate UTM EPSG code for a lon/lat point.
    Northern hemisphere -> EPSG:326 + zone
    Southern hemisphere -> EPSG:327 + zone
    """
    zone = int((lon + 180) // 6) + 1
    if lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone
    return CRS.from_epsg(epsg)


def reproject_dem_to_meters(dem_path: str) -> Tuple[np.ndarray, rasterio.Affine, CRS]:
    """
    Load DEM and (if needed) reproject it to a projected CRS with units in meters.
    Returns: (dem_array, transform, dem_crs)
    - dem_array is a numpy array of dtype float64
    - transform is the affine transform for the returned array
    - dem_crs is a rasterio CRS object for the returned array
    """
    with rasterio.open(dem_path) as src:
        src_crs = src.crs
        src_transform = src.transform
        src_bounds = src.bounds
        src_dtype = src.meta.get("dtype", "float32")
        src_nodata = src.nodata
        src_arr = src.read(1)

        # If CRS already projected (units likely meters) - don't reproject
        if src_crs is not None and not src_crs.is_geographic:
            dem_arr = src_arr.astype("float64")
            dem_transform = src_transform
            dem_crs = src_crs
            return dem_arr, dem_transform, dem_crs

        # Else: choose UTM zone based on DEM centroid and reproject to that CRS
        center_lon = (src_bounds.left + src_bounds.right) / 2.0
        center_lat = (src_bounds.bottom + src_bounds.top) / 2.0
        dst_crs = _choose_utm_crs_for_lonlat(center_lon, center_lat)

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds
        )

        # Prepare destination array and reproject (bilinear)
        dst_arr = np.empty((dst_height, dst_width), dtype=src_dtype)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_arr,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

        dem_arr = dst_arr.astype("float64")
        dem_transform = dst_transform
        dem_crs = dst_crs

        # Set nodata -> np.nan if src_nodata is set
        if src_nodata is not None:
            dem_arr[dem_arr == src_nodata] = np.nan

        # If there are NaNs, try to fill them using rasterio.fill.fillnodata (best-effort)
        if np.isnan(dem_arr).any():
            try:
                # fillnodata expects float array; mask can be boolean array where True indicates nodata
                mask = np.isnan(dem_arr)
                # We limit max_search_distance - smaller values are faster; tune if needed
                dem_arr = fillnodata(dem_arr, mask=mask, max_search_distance=100, smoothing_iterations=0)
            except Exception as e:
                warnings.warn(f"fillnodata failed ({e}). Falling back to mean-fill for NaNs.")
                # fallback: replace NaN with global finite mean (not ideal, but avoids NaN propagation)
                finite_mean = np.nanmean(dem_arr)
                dem_arr = np.where(np.isfinite(dem_arr), dem_arr, finite_mean)

        return dem_arr, dem_transform, dem_crs


def compute_slope_meters(dem_array: np.ndarray, transform: rasterio.Affine) -> np.ndarray:
    """
    Compute slope in degrees from a DEM array where transform pixel sizes are in meters.
    Uses numpy.gradient with explicit spacing (yres, xres in meters).
    Returns slope_array (same shape) in degrees.
    """
    # Ensure float dtype
    arr = dem_array.astype("float64")

    # Pixel size (meters)
    xres = transform.a
    yres = -transform.e
    if xres == 0 or yres == 0:
        raise ValueError(f"Invalid transform pixel sizes (xres={xres}, yres={yres}).")

    # Compute gradient: dz/dy (rows) , dz/dx (cols)
    dz_dy, dz_dx = np.gradient(arr, yres, xres)

    # slope in degrees
    slope = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))

    # Clip unrealistic values (numerical safety)
    slope = np.clip(slope, 0.0, 90.0)

    return slope


def latlon_to_dem_xy(lon: float, lat: float, dem_crs: CRS) -> Tuple[float, float]:
    """
    Convert lon/lat (EPSG:4326) to DEM CRS coordinates (x, y).
    If the DEM CRS is geographic (rare here), returns lon, lat unchanged.
    """
    if dem_crs is None:
        raise ValueError("dem_crs is required for coordinate transformation.")
    if dem_crs.is_geographic:
        # DEM in lat/lon already
        return lon, lat
    transformer = Transformer.from_crs("EPSG:4326", dem_crs.to_string(), always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y


def sample_slope_for_point(lat: float, lon: float, slope_array: np.ndarray, dem_transform: rasterio.Affine, dem_crs: CRS) -> float:
    """
    Given a lat/lon (EPSG:4326) sample slope value from slope_array (which uses dem_transform & dem_crs).
    Returns float slope in degrees or np.nan if outside bounds or invalid.
    """
    try:
        x, y = latlon_to_dem_xy(lon, lat, dem_crs)  # note order (lon, lat) -> (x, y)
    except Exception:
        return np.nan

    # Convert world coords -> column, row (float)
    col_f, row_f = ~dem_transform * (x, y)
    # Use floor to ensure integer index; int() truncates toward zero which can be problematic for negatives
    col = int(np.floor(col_f))
    row = int(np.floor(row_f))

    if 0 <= row < slope_array.shape[0] and 0 <= col < slope_array.shape[1]:
        val = slope_array[row, col]
        if np.isfinite(val):
            return float(val)
        else:
            return np.nan
    else:
        return np.nan


# -------------------------------
# Main flow
# -------------------------------

def main():
    print("📂 Loading dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"CSV not found: {DATA_PATH}")
    if not os.path.exists(DEM_PATH):
        raise FileNotFoundError(f"DEM not found: {DEM_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f" - Loaded {len(df)} rows from {DATA_PATH}")

    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise KeyError("CSV must contain 'latitude' and 'longitude' columns (decimal degrees).")

    # Backup existing slope column if present
    if "slope" in df.columns:
        backup_col = "slope_orig"
        # avoid overwriting previous backup
        i = 0
        while backup_col in df.columns:
            i += 1
            backup_col = f"slope_orig_{i}"
        df[backup_col] = df["slope"]
        print(f" - Existing 'slope' column backed up to '{backup_col}'")

    # Load and (if needed) reproject DEM to meters
    print("📥 Loading DEM and preparing (reproject if geographic)...")
    dem_arr, dem_transform, dem_crs = reproject_dem_to_meters(DEM_PATH)
    print(f" - DEM shape: {dem_arr.shape}, DEM CRS: {dem_crs.to_string()}, transform: {dem_transform}")

    # Compute slope (degrees)
    print("⚙️ Computing slope map (degrees) using meter-based pixel spacing...")
    slope_array = compute_slope_meters(dem_arr, dem_transform)
    print(" - Slope computed. Stats: "
          f"min={float(np.nanmin(slope_array)):.3f}, "
          f"max={float(np.nanmax(slope_array)):.3f}, "
          f"mean={float(np.nanmean(slope_array)):.3f}")

    # Extract slope for each row
    print("📌 Extracting slope value for each data point (lat/lon -> DEM CRS -> raster index)...")
    # vectorized-ish loop (still per-row; dataset sizes ~k rows so acceptable)
    slopes = []
    for idx, row in df.iterrows():
        lat = row.get("latitude", None)
        lon = row.get("longitude", None)
        if pd.isna(lat) or pd.isna(lon):
            slopes.append(np.nan)
            continue
        s = sample_slope_for_point(float(lat), float(lon), slope_array, dem_transform, dem_crs)
        slopes.append(s)

    df["slope"] = slopes

    # Basic sanity printing
    print(" - Slope column added. Summary stats:")
    print(df["slope"].describe())

    # Save CSV
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"💾 Saved updated dataset with slope -> {OUTPUT_PATH}")
    print("✅ Preprocessing complete.")


if __name__ == "__main__":
    main()
