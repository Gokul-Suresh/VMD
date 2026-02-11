import os
import numpy as np
from tracker import get_pose_data
from converter import save_vmd

# --- SETTINGS ---
INPUT_VIDEO = "dance.mp4"
OUTPUT_NAME = "motion_output.vmd"
# ----------------

def run_pipeline():
    print("--- Phase 1: Tracking ---")
    # 1. Run the tracker
    motion_data = get_pose_data(INPUT_VIDEO)
    
    if motion_data is None or len(motion_data) == 0:
        print("Pipeline stopped: No motion data captured.")
        return

    print("\n--- Phase 2: Converting ---")
    # 2. Convert directly to VMD
    save_vmd(motion_data, OUTPUT_NAME)
    
    print("\n--- DONE ---")
    print(f"Your animation is ready in the 'output' folder as: {OUTPUT_NAME}")

if __name__ == "__main__":
    run_pipeline()