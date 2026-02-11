import cv2
import mediapipe as mp
import numpy as np
import os

# --- PATH SETUP ---
# This ensures we always find files relative to where this script is saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "input")
SAVE_FILE = os.path.join(BASE_DIR, "motion_data.npy")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,       # Set to 2 for better head/neck tracking
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_pose_data(video_filename):
    video_path = os.path.join(INPUT_PATH, video_filename)
    
    if not os.path.exists(video_path):
        print(f"ERROR: Could not find {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    all_frames_data = []

    print(f"Tracking head & body motion for: {video_filename}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # FIX: Use 'pose_world_landmarks' for 3D metric coordinates
        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark
            
            frame_points = []
            for lm in landmarks:
                # FIX: Invert Y so character isn't upside down (+Y is UP in MMD)
                frame_points.append([lm.x, -lm.y, lm.z])
            
            all_frames_data.append(frame_points)

    cap.release()
    
    if len(all_frames_data) > 0:
        data = np.array(all_frames_data)
        np.save(SAVE_FILE, data) # Saves to the VMD folder automatically
        print(f"Successfully captured {len(all_frames_data)} frames.")
        print(f"Data saved to: {SAVE_FILE}")
        return data
    else:
        print("No motion detected.")
        return None

if __name__ == "__main__":
    # Change this to match your video name
    get_pose_data("dance.mp4")