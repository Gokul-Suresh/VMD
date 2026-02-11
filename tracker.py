import cv2
import mediapipe as mp
import numpy as np
import os

def get_pose_data(video_path):
    # 1. Check if file exists first
    if not os.path.exists(video_path):
        print(f"ERROR: The file '{video_path}' was not found.")
        print(f"Current working directory is: {os.getcwd()}")
        return np.array([])

    cap = cv2.VideoCapture(video_path)
    
    # 2. Check if OpenCV can open it
    if not cap.isOpened():
        print(f"ERROR: OpenCV could not open '{video_path}'.")
        print("Make sure the path is correct and you have the right codecs installed.")
        return np.array([])

    all_frames_data = []
    print(f"Successfully opened {video_path}. Reading video...")



# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_pose_data(video_path):
    cap = cv2.VideoCapture(video_path)
    all_frames_data = []

    print("Reading video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # MediaPipe needs RGB, OpenCV gives BGR
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            # We want 'world_landmarks' for 3D coordinates (meters)
            # plain 'landmarks' are just screen coordinates (pixels)
            landmarks = results.pose_landmarks.landmark
            
            # Store just the points we care about (e.g., 33 points)
            frame_points = []
            for lm in landmarks:
                frame_points.append([lm.x, lm.y, lm.z])
            
            all_frames_data.append(frame_points)

    cap.release()
    print(f"Captured {len(all_frames_data)} frames of motion.")
    return np.array(all_frames_data)

# Test it
data = get_pose_data("dance.mp4")
print(data.shape) # Should be (frames, 33, 3)