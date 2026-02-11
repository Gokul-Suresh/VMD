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
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_pose_data(video_filename):
    """
    Reads a video from the 'input' folder and extracts 3D bone coordinates.
    """
    # PATH FIX: Look inside the 'input' folder relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, "input", video_filename)

    # 2. VALIDATION: Check if the file exists
    if not os.path.exists(video_path):
        print(f"--- ERROR ---")
        print(f"Could not find: {video_filename} inside the 'input' folder.")
        print(f"Looked in: {video_path}")
        print(f"FIX: Ensure your video is located at: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    all_frames_data = []

    print(f"Starting tracking for: {video_path}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # MediaPipe requires RGB images
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # 3. COORDINATE TRACKING FIX
        # Use 'pose_world_landmarks' for 3D coordinates in meters
        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark
            
            frame_points = []
            for lm in landmarks:
                # 4. TRANSLATION FIX
                # Flip the Y-axis: MediaPipe's +Y is DOWN, 3D engines use +Y as UP
                frame_points.append([lm.x, -lm.y, lm.z])
            
            all_frames_data.append(frame_points)

    cap.release()
    
    if len(all_frames_data) == 0:
        print("Done. No motion detected.")
    else:
        print(f"Done. Captured {len(all_frames_data)} frames of motion.")
        
    return np.array(all_frames_data)

if __name__ == "__main__":
    # The name of your video file inside the 'input' folder
    VIDEO_NAME = 'dance.mp4'
    
    motion_data = get_pose_data(VIDEO_NAME)
    
    if motion_data is not None:
        print(f"Data ready for export. Shape: {motion_data.shape}")