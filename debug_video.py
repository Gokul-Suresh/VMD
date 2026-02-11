import cv2
import os

video_file = "dance.mp4"

# 1. Check where Python is actually looking
print(f"Current Working Directory: {os.getcwd()}")

# 2. Check if the file actually exists there
if os.path.exists(video_file):
    print(f"✅ File '{video_file}' found.")
else:
    print(f"❌ File '{video_file}' NOT found in the current directory.")
    print("   -> Please move the video here or use the full path (e.g., C:/Users/Name/Videos/input.mp4)")

# 3. Try to open it with OpenCV
cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    print("❌ OpenCV failed to open the video. The file might be corrupt or use an unsupported codec.")
else:
    success, frame = cap.read()
    if success:
        print(f"✅ Success! Video opened. Resolution: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print("❌ OpenCV opened the file but couldn't read the first frame.")

cap.release()