import cv2
import mediapipe as mp
import numpy as np
import os

# 1. SETUP - Configure MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,       # Highest accuracy for bone tracking
    smooth_landmarks=True,
    min_detection_confidence=0.35,
    min_tracking_confidence=0.35
)


def _resolve_video_path(video_filename):
    """Resolve a video path from common folder name variants."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_dirs = ["input", "Input"]

    for folder in candidate_dirs:
        video_path = os.path.join(script_dir, folder, video_filename)
        if os.path.exists(video_path):
            return video_path

    # Fallback to the default expected path so the error message is predictable.
    return os.path.join(script_dir, "input", video_filename)


def _normalize_frame(frame_points):
    """
    Normalize a skeleton frame for more stable rig retargeting.

    - Root at midpoint of left/right hips.
    - Scale by shoulder width to reduce camera-distance variation.
    """
    left_hip = np.array(frame_points[23], dtype=np.float32)
    right_hip = np.array(frame_points[24], dtype=np.float32)
    root = (left_hip + right_hip) / 2.0

    centered = np.array(frame_points, dtype=np.float32) - root

    left_shoulder = centered[11]
    right_shoulder = centered[12]
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)

    if shoulder_width > 1e-6:
        centered /= shoulder_width

    return centered.tolist()


def get_pose_data(video_filename):
    """
    Reads a video from the input folder and extracts 3D bone coordinates.

    Important behavior for VMD/retargeting workflows:
    - Keeps frame count aligned with the source video.
    - Reuses previous frame values for temporarily occluded landmarks.
    - Returns normalized skeleton coordinates for steadier playback.
    """
    video_path = _resolve_video_path(video_filename)

    # 2. VALIDATION: Check if the file exists
    if not os.path.exists(video_path):
        print("--- ERROR ---")
        print(f"Could not find: {video_filename} inside the 'input' or 'Input' folder.")
        print(f"Looked in: {video_path}")
        print(f"FIX: Ensure your video is located at: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    all_frames_data = []
    previous_points = None

    print(f"Starting tracking for: {video_path}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe requires RGB images
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        frame_points = None

        # Use world landmarks for 3D coordinates in meters.
        # Also use 2D landmark visibility to avoid noisy jumps.
        if results.pose_world_landmarks and results.pose_landmarks:
            world_landmarks = results.pose_world_landmarks.landmark
            image_landmarks = results.pose_landmarks.landmark

            frame_points = []
            for i, lm in enumerate(world_landmarks):
                vis = image_landmarks[i].visibility

                # Flip Y-axis: MediaPipe +Y is down, most 3D rigs use +Y up.
                point = [lm.x, -lm.y, lm.z]

                # If a point is not visible enough, reuse previous value.
                if previous_points is not None and vis < 0.5:
                    point = previous_points[i]

                frame_points.append(point)

            frame_points = _normalize_frame(frame_points)

        if frame_points is None:
            # Keep video frame alignment to avoid broken keyframe timing.
            if previous_points is not None:
                frame_points = previous_points
            else:
                frame_points = [[0.0, 0.0, 0.0] for _ in range(33)]

        all_frames_data.append(frame_points)
        previous_points = frame_points

    cap.release()

    if len(all_frames_data) == 0:
        print("Done. No motion detected.")
    else:
        print(f"Done. Captured {len(all_frames_data)} frames of motion.")

    return np.array(all_frames_data, dtype=np.float32)


if __name__ == "__main__":
    # The name of your video file inside the input folder
    VIDEO_NAME = "dance.mp4"

    motion_data = get_pose_data(VIDEO_NAME)

    if motion_data is not None:
        print(f"Data ready for export. Shape: {motion_data.shape}")
