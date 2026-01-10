import os
import random
from pyboy import PyBoy
# FIX: WindowEvent is located in utils in newer PyBoy versions
from pyboy.utils import WindowEvent 

# --- CONFIGURATION ---
ROM_PATH = "rom.gb"         # MUST be a .gb or .gbc file (NOT .gba)
OUTPUT_DIR = "dataset"
TARGET_FPS = 6
GAME_SPEED = 0              # 0 = Max speed
TOTAL_HOURS = 50

# --- CALCULATIONS ---
# Game Boy runs at 60 FPS. We save every 10th frame to get 6 FPS.
SAVE_INTERVAL = 60 // TARGET_FPS 
TOTAL_FRAMES = TOTAL_HOURS * 60 * 60 * 60 

os.makedirs(OUTPUT_DIR, exist_ok=True)

# window="null" is fastest for headless data generation on M2
# If you want to see it running, change to window="SDL2" (slower)
print(f"Loading {ROM_PATH}...")
pyboy = PyBoy(ROM_PATH, window="null")
pyboy.set_emulation_speed(GAME_SPEED)

print(f"Starting generation. Target: {TOTAL_HOURS} hours gameplay.")

# Valid actions for Pokemon Red
valid_actions = [
    WindowEvent.PRESS_ARROW_UP,
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B,
    WindowEvent.PRESS_BUTTON_START, 
]

release_actions = {
    WindowEvent.PRESS_ARROW_UP: WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.PRESS_ARROW_DOWN: WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_LEFT: WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT: WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.PRESS_BUTTON_A: WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B: WindowEvent.RELEASE_BUTTON_B,
    WindowEvent.PRESS_BUTTON_START: WindowEvent.RELEASE_BUTTON_START,
}

frame_count = 0
saved_count = 0
current_action = None

try:
    while frame_count < TOTAL_FRAMES:
        # --- RANDOM AGENT LOGIC ---
        # Press button every 15 frames
        if frame_count % 15 == 0:
            current_action = random.choice(valid_actions)
            pyboy.send_input(current_action)
        
        # Release button 5 frames later
        if frame_count % 15 == 5 and current_action:
            release = release_actions.get(current_action)
            if release:
                pyboy.send_input(release)
            current_action = None

        # --- ADVANCE FRAME ---
        pyboy.tick()
        frame_count += 1

        # --- SAVE DATA ---
        if frame_count % SAVE_INTERVAL == 0:
            filename = f"{OUTPUT_DIR}/frame_{saved_count:07d}.png"
            # PyBoy screen image is PIL format
            pyboy.screen.image.save(filename)
            
            saved_count += 1
            if saved_count % 1000 == 0:
                print(f"Progress: {saved_count} frames saved.")

except KeyboardInterrupt:
    print("\nGeneration paused by user.")
except Exception as e:
    print(f"\nError: {e}")
finally:
    pyboy.stop()
    print(f"Done. Saved {saved_count} frames to '{OUTPUT_DIR}/'")