# data/download_jhtdb.py
import os
import h5py
from pyJHTDB import isotropic
from pyJHTDB.lib import configure_token
from tqdm import tqdm
import numpy as np
import time

def download_2d_slices(root_dir='data/raw', num_slices=100, resolution=1024):
    os.makedirs(root_dir, exist_ok=True)
    h5_path = os.path.join(root_dir, 'jhtdb_2d_slices.h5')
    if os.path.exists(h5_path):
        print(f"Dataset already exists at {h5_path}")
        return

    # Use default token for testing; replace with your token for production
    configure_token('edu.jhu.pha.turbulence.testing-201406')

    with h5py.File(h5_path, 'w') as f:
        u_group = f.create_group('u')
        v_group = f.create_group('v')
        magnitude_group = f.create_group('magnitude')  # For velocity magnitude

        print(f"Downloading {num_slices} 2D slices from isotropic turbulence...")
        for i in tqdm(range(num_slices)):
            t = i * 0.002  # Time step (0 to 0.2, covering dataset range)
            x = np.linspace(0, 2*np.pi, resolution)
            y = np.linspace(0, 2*np.pi, resolution)
            z = np.full_like(x, np.pi)  # Fixed z-plane for 2D
            X, Y, Z = np.meshgrid(x, y, [np.pi], indexing='ij')
            coords = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)

            try:
                vel = isotropic.get_velocity(coords, t=t)
                u_slice = vel[:, 0].reshape(resolution, resolution)
                v_slice = vel[:, 1].reshape(resolution, resolution)
                magnitude = np.sqrt(u_slice**2 + v_slice**2)  # Scalar field for SR
                u_group.create_dataset(str(i), data=u_slice)
                v_group.create_dataset(str(i), data=v_slice)
                magnitude_group.create_dataset(str(i), data=magnitude)
            except Exception as e:
                print(f"Error downloading slice {i}: {e}")
                continue

            time.sleep(0.5)  # Avoid server throttling

    size_gb = os.path.getsize(h5_path) / (1024**3)
    print(f"Downloaded {h5_path} ({size_gb:.2f} GB). Contains {num_slices} 2D slices.")

if __name__ == "__main__":
    download_2d_slices()