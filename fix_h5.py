import h5py
import os

# Path to your file
FILE_PATH = "datasets/pokemon.h5"

if not os.path.exists(FILE_PATH):
    print(f"Error: File not found at {FILE_PATH}")
    exit()

with h5py.File(FILE_PATH, 'r+') as f:
    print(f"Current keys: {list(f.keys())}")
    
    if 'images' in f.keys():
        print("Found key 'images'. Renaming to 'frames'...")
        f.move('images', 'frames')
        print("Success! Key renamed.")
    elif 'frames' in f.keys():
        print("Key 'frames' already exists. No changes needed.")
    else:
        print("Error: Could not find 'images' key.")
        
    print(f"Final keys: {list(f.keys())}")