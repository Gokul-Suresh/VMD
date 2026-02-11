import cv2
import mediapipe as mp
import numpy as np
import struct
import os

# --- PATH CONFIGURATION ---
# We use os.path.join to make sure it works on Windows and Mac
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(CURRENT_DIR, 'Input')
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'Output')

VIDEO_FILENAME = 'dance.mp4'
OUTPUT_FILENAME = 'motion.vmd'

VIDEO_PATH = os.path.join(INPUT_DIR, VIDEO_FILENAME)
OUTPUT_VMD = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

SCALE_FACTOR = 80.0

# --- CHECK PATHS BEFORE STARTING ---
if not os.path.exists(INPUT_DIR):
    print(f"ERROR: The folder '{INPUT_DIR}' does not exist.")
    print("Please create a folder named 'Input' and put your video inside.")
    exit()

if not os.path.exists(VIDEO_PATH):
    print(f"ERROR: Could not find '{VIDEO_FILENAME}' inside the Input folder.")
    print(f"Looking for: {VIDEO_PATH}")
    exit()

if not os.path.exists(OUTPUT_DIR):
    print(f"Creating missing Output folder: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR)

# --- MMD BONE NAMES ---
BONE_MAP = {
    # --- LEFT SIDE ---
    "左腕": (11, 13, [1, 0, 0]),   # Left Arm (Shoulder -> Elbow)
    "左ひじ": (13, 15, [1, 0, 0]),  # Left Elbow (Elbow -> Wrist)
    
    # --- RIGHT SIDE ---
    "右腕": (12, 14, [-1, 0, 0]),  # Right Arm (Shoulder -> Elbow)
    "右ひじ": (14, 16, [-1, 0, 0]), # Right Elbow (Elbow -> Wrist)
    
    # --- LEGS ---
    "左足": (23, 25, [0, -1, 0]),  # Left Leg (Hip -> Knee)
    "左ひざ": (25, 27, [0, -1, 0]), # Left Knee (Knee -> Ankle)
    "右足": (24, 26, [0, -1, 0]),  # Right Leg (Hip -> Knee)
    "右ひざ": (26, 28, [0, -1, 0])  # Right Knee (Knee -> Ankle)
}

def get_quaternion_between_vectors(u, v):
    u = np.array(u)
    v = np.array(v)
    if np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0: return [0, 0, 0, 1]
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    dot = np.dot(u, v)
    if dot > 0.99999: return [0, 0, 0, 1]
    if dot < -0.99999: return [0, 1, 0, 0]
    axis = np.cross(u, v)
    w = 1.0 + dot
    q = np.array([axis[0], axis[1], axis[2], w])
    return q / np.linalg.norm(q)

def write_vmd(filename, bone_frames):
    print(f"Writing {len(bone_frames)} frames to {filename}...")
    with open(filename, 'wb') as f:
        f.write(b'Vocaloid Motion Data 0002' + b'\x00' * 5)
        f.write(b'Camera_Motion' + b'\x00' * 7)
        f.write(struct.pack('<I', len(bone_frames)))
        for bf in bone_frames:
            name_bytes = bf['name'].encode('shift-jis', errors='replace')
            name_bytes = name_bytes[:15].ljust(15, b'\x00')
            f.write(name_bytes)
            f.write(struct.pack('<I', bf['frame']))
            f.write(struct.pack('<3f', *bf['pos']))
            f.write(struct.pack('<4f', *bf['rot']))
            f.write(b'\x00' * 64)
        for _ in range(4): f.write(struct.pack('<I', 0))

def process():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    bone_data = []
    frame_idx = 0

    print(f"Processing {VIDEO_FILENAME}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_world_landmarks.landmark
            
            # Root Motion
            l_hip = np.array([landmarks[23].x, landmarks[23].y, landmarks[23].z])
            r_hip = np.array([landmarks[24].x, landmarks[24].y, landmarks[24].z])
            center_pos = (l_hip + r_hip) / 2.0
            # Try changing +10 to -5 or -10 to bring her down
            final_pos = [center_pos[0] * SCALE_FACTOR, -center_pos[1] * SCALE_FACTOR - 2, -center_pos[2] * SCALE_FACTOR]
            bone_data.append({"name": "センター", "frame": frame_idx, "pos": final_pos, "rot": [0, 0, 0, 1]})

            # Limb Rotation
            for mmd_name, (p_idx, c_idx, ref_vec) in BONE_MAP.items():
                p = np.array([landmarks[p_idx].x, landmarks[p_idx].y, landmarks[p_idx].z])
                c = np.array([landmarks[c_idx].x, landmarks[c_idx].y, landmarks[c_idx].z])
                current_vec = c - p
                current_vec[1] = -current_vec[1] 
                q = get_quaternion_between_vectors(ref_vec, current_vec)
                bone_data.append({"name": mmd_name, "frame": frame_idx, "pos": [0, 0, 0], "rot": q})

        frame_idx += 1
        if frame_idx % 30 == 0: print(f"Processed {frame_idx} frames...")

    cap.release()
    write_vmd(OUTPUT_VMD, bone_data)
    print(f"Success! Saved to Output/{OUTPUT_FILENAME}")

if __name__ == "__main__":
    process()