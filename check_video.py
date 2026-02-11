import cv2
import os

# --- REPLACE THIS WITH YOUR VIDEO FILENAME ---
VIDEO_FILE = 'dance.mp4' 
# ---------------------------------------------

print(f"--- DIAGNOSTIC START ---")
print(f"Current Working Directory: {os.getcwd()}")

# 1. Check if the file path is correct
if not os.path.exists(VIDEO_FILE):
    print(f"\n[FAIL] File not found: '{VIDEO_FILE}'")
    print(f"Step to fix: Move '{VIDEO_FILE}' into the folder displayed above.")
    print(f"OR: Use the full path (e.g., C:/Users/Name/Downloads/{VIDEO_FILE})")
else:
    print(f"\n[PASS] File found at: {os.path.abspath(VIDEO_FILE)}")

    # 2. Check if OpenCV can open it
    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        print(f"[FAIL] OpenCV could not open the video.")
        print("Possible causes: Corrupted file or missing codec (install ffmpeg).")
    else:
        print(f"[PASS] OpenCV opened the video successfully.")
        
        # 3. Try to read one frame
        ret, frame = cap.read()
        if ret:
            print(f"[PASS] Successfully read a frame. Resolution: {frame.shape[1]}x{frame.shape[0]}")
            print("\nYour video is working! The issue was likely the path in the previous script.")
        else:
            print(f"[FAIL] Opened video, but could not read any frames.")

cap.release()
print("--- DIAGNOSTIC END ---")