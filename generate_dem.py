import numpy as np
import rasterio
from rasterio.transform import from_origin

width, height = 200, 200     # pixels
resolution = 5               # meters per pixel

def generate_rugged_dem(width, height, scale=40, octaves=4, persistence=0.5, lacunarity=2.0, seed=42):
    np.random.seed(seed)
    dem = np.zeros((height, width))
    for o in range(octaves):
        noise = np.random.rand(height, width)
        dem += (persistence ** o) * noise
        scale /= lacunarity
    dem = (dem - dem.min()) / (dem.max() - dem.min())  # normalize 0–1
    dem *= 300  # set elevation range ~0–300 m
    return dem

dem = generate_rugged_dem(width, height)

# Add a pit to mimic an open-pit mine
cx, cy = width // 2, height // 2
X, Y = np.meshgrid(np.arange(width), np.arange(height))
dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
pit = 100 * np.exp(-dist**2 / (2*(30**2)))
dem -= pit
dem = np.clip(dem, 0, None)

profile = {
    'driver': 'GTiff',
    'height': height,
    'width': width,
    'count': 1,
    'dtype': rasterio.float32,
    'crs': 'EPSG:4326',
    'transform': from_origin(86.35, 23.8, 0.0005, 0.0005)  # approx geographic extent
}

with rasterio.open("jharia_synthetic_dem_fixed.tif", "w", **profile) as dst:
    dst.write(dem.astype(np.float32), 1)

print("✅ New DEM saved as jharia_synthetic_dem_fixed.tif")
