import h5py
import glob
from PIL import Image
import numpy as np
import os
from tqdm import tqdm # Optional, for progress bar (pip install tqdm)

# CONFIGURATION
DATA_DIR = "dataset"          # Your folder of images
OUTPUT_FILE = "pokemon.h5"    # The file tinyworlds needs
IMAGE_SIZE = 128              # Must match your training config

print(f"Looking for images in {DATA_DIR}...")
# Sort is CRITICAL to keep time order valid!
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.png"))) 

if not files:
    print("Error: No images found! Check the DATA_DIR path.")
    exit()

print(f"Found {len(files)} frames. Packing into {OUTPUT_FILE}...")

with h5py.File(OUTPUT_FILE, 'w') as f:
    # Create the dataset container
    dset = f.create_dataset("images", (len(files), IMAGE_SIZE, IMAGE_SIZE, 3), dtype='uint8')
    
    for i, file in enumerate(tqdm(files)):
        try:
            # Load and ensure RGB
            img = Image.open(file).convert('RGB')
            # Resize ensures compatibility even if your generation was slightly off
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            dset[i] = np.array(img)
        except Exception as e:
            print(f"Skipping bad frame {file}: {e}")

print("Done! You can now delete the 'dataset' folder to save space.")