import h5py
import glob
import numpy as np
import os
import re
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = "Dataset"          # Path to your folder containing batch subfolders
OUTPUT_FILE = "pokemon.h5"    # Output file
IMAGE_SIZE = 128              # Target size (must match training config)
CHUNK_SIZE = 100              # How many images to process at once (Low memory usage)

# --- HELPER: NATURAL SORT ---
# This ensures "frame_10.png" comes after "frame_9.png", not "frame_1.png"
def natural_keys(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

print(f"Looking for images in {DATA_DIR} recursively...")

# 1. Find all PNGs recursively in subfolders
#    Dataset/batch_1/frame_0.png, Dataset/batch_2/frame_0.png, etc.
files = glob.glob(os.path.join(DATA_DIR, "**", "*.png"), recursive=True)

# 2. Sort them to ensure time consistency
#    We sort by the full path to ensure batch_1 comes before batch_2, 
#    and frame_1 comes before frame_2.
files.sort(key=natural_keys)

if not files:
    print(f"Error: No images found in {DATA_DIR} or its subfolders!")
    exit()

total_files = len(files)
print(f"Found {total_files} frames. Packing into {OUTPUT_FILE}...")

# 3. Create H5 file with efficient Chunking
with h5py.File(OUTPUT_FILE, 'w') as f:
    # Create a dataset that can be resized if needed, using chunks for efficiency.
    # 'chunks' allows the file to be read/written in blocks, crucial for performance.
    dset = f.create_dataset(
        "images", 
        (total_files, IMAGE_SIZE, IMAGE_SIZE, 3), 
        dtype='uint8',
        chunks=(1, IMAGE_SIZE, IMAGE_SIZE, 3) # Optimize for reading 1 frame at a time
    )
    
    # 4. Write in Batches (Memory Optimization)
    #    Instead of loading 50,000 images into RAM (which crashes 16GB Macs),
    #    we load 100 images, write them to disk, and clear RAM.
    for i in tqdm(range(0, total_files, CHUNK_SIZE)):
        batch_files = files[i : i + CHUNK_SIZE]
        batch_images = []
        
        for file_path in batch_files:
            try:
                # Load -> Resize -> Array
                img = Image.open(file_path).convert('RGB')
                if img.size != (IMAGE_SIZE, IMAGE_SIZE):
                    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BOX)
                
                batch_images.append(np.array(img))
            except Exception as e:
                print(f"Warning: Skipping bad file {file_path}: {e}")
                # If a file fails, we append a black frame to keep indexing safe
                batch_images.append(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype='uint8'))

        # Stack into a numpy array (Batch_Size, 128, 128, 3)
        batch_data = np.stack(batch_images, axis=0)
        
        # Write directly to the H5 file on disk
        dset[i : i + len(batch_files)] = batch_data

print(f"Success! {total_files} frames packed into '{OUTPUT_FILE}'.")
print("You can now run the training script.")