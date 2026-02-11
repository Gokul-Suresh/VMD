import os
from tracker import get_pose_data
from converter import save_vmd

# --- SETTINGS ---
INPUT_VIDEO = "dance.mp4"
OUTPUT_NAME = "motion_output.vmd"
# ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOTION_DATA_FILE = os.path.join(BASE_DIR, "motion_data.npy")


def _load_existing_motion_data(path):
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        print("ERROR: Missing dependency: numpy. Install with: pip install numpy")
        print(f"Details: {exc}")
        return None

    return np.load(path)


def run_pipeline(input_video: str = INPUT_VIDEO, output_name: str = OUTPUT_NAME):
    print("--- Phase 1: Tracking ---")
    motion_data = get_pose_data(input_video)

    if motion_data is None or len(motion_data) == 0:
        if os.path.exists(MOTION_DATA_FILE):
            print("Tracking unavailable/empty. Falling back to existing motion_data.npy...")
            motion_data = _load_existing_motion_data(MOTION_DATA_FILE)
        else:
            print("Pipeline stopped: No motion data captured.")
            return None

    if motion_data is None:
        return None

    print("\n--- Phase 2: Converting ---")
    try:
        output_path = save_vmd(motion_data, output_name)
    except RuntimeError as err:
        print(f"ERROR: {err}")
        return None

    print("\n--- DONE ---")
    print(f"Your animation is ready at: {output_path}")
    return output_path


if __name__ == "__main__":
    run_pipeline()
