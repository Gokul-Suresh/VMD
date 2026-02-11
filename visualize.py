import cv2
import mediapipe as mp

# Setup MediaPipe Visuals
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- INPUT YOUR VIDEO HERE ---
VIDEO_PATH = 'dance.mp4' 
# -----------------------------

cap = cv2.VideoCapture(VIDEO_PATH)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

print("Press 'q' to quit the window.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    # Recolor to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False # Improves performance
    
    # Make detection
    results = pose.process(image)
    
    # Recolor back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Draw Landmarks (The Skeleton)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), # Joints
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)  # Bones
        )
        
    cv2.imshow('MediaPipe Feed', image)

    # Exit on 'q' key
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()