import numpy as np
from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_dilation
import pickle, time

DATA_DIR = Path("/mnt/workspace/training_data")
OUT_DIR = Path("/mnt/workspace/training_3d")
OUT_DIR.mkdir(exist_ok=True)
RESOLUTION = 64

def create_visual_hull(views, resolution=64):
    azimuths = [30, 90, 150, 210, 270, 330]
    coords = np.linspace(-1, 1, resolution)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    voxels = np.ones(X.shape, dtype=bool)

    for view, az in zip(views, azimuths):
        img = np.array(view.resize((resolution, resolution))).astype(np.float32)
        gray = np.mean(img, axis=2)
        std_img = np.std(img, axis=2)
        fg = ~((gray > 180) | ((gray > 140) & (std_img < 15)))
        fg = binary_fill_holes(fg)
        fg = binary_dilation(fg, iterations=1)

        rad = np.radians(az)
        el_rad = np.radians(20)
        px = X * np.cos(rad) + Z * np.sin(rad)
        pz = -X * np.sin(rad) + Z * np.cos(rad)
        py = Y * np.cos(el_rad) - pz * np.sin(el_rad)
        ix = np.clip(((px + 1) / 2 * (resolution - 1)).astype(int), 0, resolution - 1)
        iy = np.clip(((1 - (py + 1) / 2) * (resolution - 1)).astype(int), 0, resolution - 1)
        voxels &= fg[iy, ix]

    voxels_smooth = gaussian_filter(voxels.astype(float), sigma=0.8) > 0.3
    return voxels_smooth.astype(np.float32)

samples = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
print(f"Processing {len(samples)} chairs into 3D voxels ({RESOLUTION}^3)...")

t0 = time.time()
count = 0
for i, d in enumerate(samples):
    inp = d / "input.png"
    view_files = [d / f"view_{j}.png" for j in range(6)]
    if not inp.exists() or not all(v.exists() for v in view_files):
        continue

    views = [Image.open(v).convert("RGB") for v in view_files]
    voxels = create_visual_hull(views, RESOLUTION)

    # Save
    out_path = OUT_DIR / f"{d.name}.npz"
    np.savez_compressed(out_path, voxels=voxels, input_name=str(inp))
    count += 1

    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(samples)} done ({time.time()-t0:.0f}s)")

print(f"\nDone! {count} voxel grids saved to {OUT_DIR}")
print(f"Time: {time.time()-t0:.0f}s")
