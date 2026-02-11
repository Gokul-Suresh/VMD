import os
import struct

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "motion_data.npy")
OUTPUT_DIR = os.path.join(BASE_DIR, "Output")


def _import_numpy():
    try:
        import numpy as np
        return np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: numpy. Install with: pip install numpy"
        ) from exc


def _quat_from_euler_xyz(pitch: float, yaw: float, roll: float):
    """Build quaternion [x, y, z, w] from Euler XYZ angles in radians."""
    np = _import_numpy()
    cx = np.cos(pitch * 0.5)
    sx = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cz = np.cos(roll * 0.5)
    sz = np.sin(roll * 0.5)

    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz

    return [float(x), float(y), float(z), float(w)]


def save_vmd(motion_data, output_name: str = "head_motion.vmd"):
    """Convert MediaPipe pose world landmarks to a simple head-bone VMD animation."""
    np = _import_numpy()

    if motion_data is None or len(motion_data) == 0:
        raise ValueError("motion_data is empty; run tracking first.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_name)

    header = b"Vocaloid Motion Data 0002\x00".ljust(30, b"\x00")
    model_name = "PythonTracker".encode("shift_jis", errors="ignore").ljust(20, b"\x00")
    bone_name = "頭".encode("shift_jis").ljust(15, b"\x00")

    bone_frames = []
    print(f"Converting {len(motion_data)} frames to VMD ({output_name})...")

    for frame_idx, landmarks in enumerate(motion_data):
        nose = landmarks[0]
        left_ear = landmarks[7]
        right_ear = landmarks[8]

        ear_dx = left_ear[0] - right_ear[0]
        ear_dz = left_ear[2] - right_ear[2]
        yaw = np.arctan2(ear_dz, ear_dx)

        ear_mid_y = (left_ear[1] + right_ear[1]) * 0.5
        pitch = (nose[1] - ear_mid_y) * 1.8

        roll = np.arctan2(left_ear[1] - right_ear[1], ear_dx if abs(ear_dx) > 1e-6 else 1e-6)
        roll *= 0.5

        q = _quat_from_euler_xyz(float(pitch), float(yaw), float(roll))
        pos = [0.0, 0.0, 0.0]

        frame_data = struct.pack("<Ifffffff", frame_idx, *pos, *q)
        bone_frames.append(bone_name + frame_data + (b"\x00" * 64))

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(model_name)
        f.write(struct.pack("<I", len(bone_frames)))
        for frame in bone_frames:
            f.write(frame)

        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))

    print(f"Success! VMD saved to: {output_path}")
    return output_path


def save_head_vmd():
    """Backward-compatible entrypoint using saved NPY motion data."""
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found. Run tracker.py first!")
        return None

    np = _import_numpy()
    motion_data = np.load(DATA_PATH)
    return save_vmd(motion_data, "head_motion.vmd")


if __name__ == "__main__":
    try:
        save_head_vmd()
    except RuntimeError as err:
        print(f"ERROR: {err}")
