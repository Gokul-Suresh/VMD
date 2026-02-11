import cv2
import mediapipe as mp

# 1. Setup MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. Open the video file
video_path = 'dance.mp4'  # Make sure this matches your file name
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

print("Processing video... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. Convert BGR (OpenCV format) to RGB (MediaPipe format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 4. Process the frame to find the pose
    results = pose.process(frame_rgb)

    # 5. Draw the landmarks on the frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )
        
        # OPTIONAL: Print the nose coordinates to the terminal to prove we have data
        # nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        # print(f"Nose X: {nose.x}, Nose Y: {nose.y}")

    # 6. Show the video
    cv2.imshow('MediaPipe Skeleton Preview', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()