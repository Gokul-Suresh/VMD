import numpy as np
import struct
import os

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "motion_data.npy")
OUTPUT_PATH = os.path.join(BASE_DIR, "output", "head_motion.vmd")

def save_head_vmd():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found. Run tracker.py first!")
        return

    motion_data = np.load(DATA_PATH)
    os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)

    # VMD Header (MMD version 2)
    header = b'Vocaloid Motion Data 0002\x00'.ljust(30, b'\x00')
    model_name = "PythonTracker".encode('shift_jis').ljust(20, b'\x00')
    
    # Bone: "頭" (Head) encoded in Japanese Shift-JIS
    bone_name = "頭".encode('shift_jis').ljust(15, b'\x00')
    
    bone_frames = []

    print(f"Converting {len(motion_data)} frames for Head tracking...")

    for frame_idx, landmarks in enumerate(motion_data):
        # MediaPipe Head Landmarks: 0=Nose, 7=L Ear, 8=R Ear
        nose = landmarks[0]
        l_ear = landmarks[7]
        r_ear = landmarks[8]
        
        # --- SIMPLE ROTATION CALCULATION ---
        # Yaw (Left/Right): Z-depth difference between ears
        yaw = np.arctan2(l_ear[2] - r_ear[2], l_ear[0] - r_ear[0])
        
        # Pitch (Up/Down): Nose Y relative to ear height
        ear_mid_y = (l_ear[1] + r_ear[1]) / 2
        pitch = (nose[1] - ear_mid_y) * 2 # Sensitivity multiplier
        
        # Convert to a basic Quaternion [x, y, z, w]
        # This is a simplified version for head-only testing
        cy = np.cos(yaw * 0.5); sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5); sp = np.sin(pitch * 0.5)
        
        q = [
            sp * cy,  # X (Pitch)
            cp * sy,  # Y (Yaw)
            0.0,      # Z (Roll - ignored for now)
            cp * cy   # W
        ]
        
        # Head usually doesn't "translate" (move position), it just rotates
        pos = [0, 0, 0] 

        # Pack binary: frame (I), pos (3f), rot (4f), interp (64 bytes)
        frame_data = struct.pack('<Ifffffff', frame_idx, *pos, *q)
        bone_frames.append(bone_name + frame_data + (b'\x00' * 64))

    with open(OUTPUT_PATH, 'wb') as f:
        f.write(header)
        f.write(model_name)
        f.write(struct.pack('<I', len(bone_frames)))
        for bf in bone_frames:
            f.write(bf)

    print(f"Success! Open this in MMD: {OUTPUT_PATH}")

if __name__ == "__main__":
    save_head_vmd()