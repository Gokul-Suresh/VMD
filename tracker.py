import os

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR_CANDIDATES = [
    os.path.join(BASE_DIR, "Input"),
    os.path.join(BASE_DIR, "input"),
    BASE_DIR,
]
SAVE_FILE = os.path.join(BASE_DIR, "motion_data.npy")


def _resolve_video_path(video_filename: str) -> str | None:
    """Resolve a video filename across common project input locations."""
    if os.path.isabs(video_filename) and os.path.exists(video_filename):
        return video_filename

    direct_path = os.path.join(BASE_DIR, video_filename)
    if os.path.exists(direct_path):
        return direct_path

    for folder in INPUT_DIR_CANDIDATES:
        candidate = os.path.join(folder, video_filename)
        if os.path.exists(candidate):
            return candidate

    return None


def get_pose_data(video_filename: str):
    try:
        import cv2
        import mediapipe as mp
    except ModuleNotFoundError as exc:
        print(
            "ERROR: Missing tracking dependency. Install with: "
            "pip install opencv-python mediapipe"
        )
        print(f"Details: {exc}")
        return None

    video_path = _resolve_video_path(video_filename)

    if video_path is None:
        searched = "\n - ".join([os.path.join(d, video_filename) for d in INPUT_DIR_CANDIDATES])
        print(f"ERROR: Could not find video '{video_filename}'. Searched:\n - {searched}")
        return None

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,       # better head/neck tracking
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    all_frames_data = []

    print(f"Tracking head & body motion for: {video_path}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark

            frame_points = []
            for lm in landmarks:
                frame_points.append([lm.x, -lm.y, lm.z])

            all_frames_data.append(frame_points)

    cap.release()

    if len(all_frames_data) > 0:
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            print("ERROR: Missing dependency: numpy. Install with: pip install numpy")
            print(f"Details: {exc}")
            return None

        data = np.array(all_frames_data, dtype=np.float32)
        np.save(SAVE_FILE, data)
        print(f"Successfully captured {len(all_frames_data)} frames.")
        print(f"Data saved to: {SAVE_FILE}")
        return data

    print("No motion detected.")
    return None


if __name__ == "__main__":
    get_pose_data("dance.mp4")
