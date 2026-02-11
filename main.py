import cv2
import mediapipe as mp
import numpy as np
from vmd_writer import VmdWriter
from pose_math import get_quaternion_between_vectors, quaternion_multiply, quaternion_inverse, get_mp_vector

# --- CONFIGURATION ---
VIDEO_PATH = 'dance.mp4'  # Ensure full path if needed
OUTPUT_PATH = 'output1.vmd'

# --- SETUP ---
mp_pose = mp.solutions.pose
# We use COMPLEXITY=2 for better accuracy on the "flopping"
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2)
writer = VmdWriter()

# --- BONE DEFINITIONS ---
# MMD Standard Rest Pose Vectors (approximate directions in T-Pose)
# X+ is Left, Y+ is Up, Z+ is Back (relative to model)
VEC_VERTICAL_DOWN = np.array([0, -1, 0])
VEC_VERTICAL_UP   = np.array([0,  1, 0])
VEC_HORIZONTAL_L  = np.array([1,  0, 0])  # Left Arm
VEC_HORIZONTAL_R  = np.array([-1, 0, 0])  # Right Arm
VEC_LEG_DOWN      = np.array([0, -1, 0])

cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0

print(f"Processing {VIDEO_PATH}...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_world_landmarks:
        # We use pose_WORLD_landmarks for physics-accurate meters, not pixels
        lm = results.pose_world_landmarks.landmark
        
        # --- 1. ROOT / CENTER (HIPS) ---
        # Calculate global hip rotation first
        # MP: 23=Left Hip, 24=Right Hip
        # We want the vector pointing from Right Hip to Left Hip
        hip_vec_mp = get_mp_vector(lm, 24, 23)
        target_hip_vec = np.array([1, 0, 0]) # MMD Hips represent "Left"
        
        # Calculate Root Rotation (Center Bone)
        rot_center = get_quaternion_between_vectors(target_hip_vec, hip_vec_mp)
        
        # The Inverse is needed to cancel out root rotation for the upper body
        rot_center_inv = quaternion_inverse(rot_center)

        # --- 2. UPPER BODY ---
        
        # Function to calculate local bone rotation
        def calc_bone_rot(start_idx, end_idx, base_vector, parent_rot_inv):
            # 1. Get current physical vector
            curr_vec = get_mp_vector(lm, start_idx, end_idx)
            
            # 2. Rotate it "back" by the parent's inverse rotation
            # This makes the vector local to the parent bone
            # (e.g. If chest is bent 45deg, ignore that 45deg for the arm calculation)
            # Note: This is a simplified approach. Full IK is harder.
            
            # We calculate global rotation first
            global_rot = get_quaternion_between_vectors(base_vector, curr_vec)
            
            # Then remove parent rotation: Local = Parent_Inv * Global
            local_rot = quaternion_multiply(parent_rot_inv, global_rot)
            return local_rot, global_rot

        # Define Bone Map: (Name, StartMP, EndMP, BaseVector, ParentInverse)
        
        # SPINE / CHEST
        # MP: 11/12 Shoulders, 23/24 Hips.
        # Vector from Mid-Hip to Mid-Shoulder
        mid_hip_x = (lm[23].x + lm[24].x) / 2
        mid_hip_y = (lm[23].y + lm[24].y) / 2
        mid_hip_z = (lm[23].z + lm[24].z) / 2
        
        mid_shou_x = (lm[11].x + lm[12].x) / 2
        mid_shou_y = (lm[11].y + lm[12].y) / 2
        mid_shou_z = (lm[11].z + lm[12].z) / 2
        
        # Custom vector construction for Spine
        spine_vec_mp = np.array([mid_shou_x - mid_hip_x, -(mid_shou_y - mid_hip_y), -(mid_shou_z - mid_hip_z)])
        rot_spine = get_quaternion_between_vectors(VEC_VERTICAL_UP, spine_vec_mp)
        # Make spine local to Center (hips)
        rot_spine_local = quaternion_multiply(rot_center_inv, rot_spine)
        rot_spine_inv = quaternion_inverse(rot_spine) # For children of spine

        # ARMS
        # Left Arm (11->13)
        rot_l_arm, rot_l_arm_global = calc_bone_rot(11, 13, VEC_HORIZONTAL_L, rot_spine_inv)
        # Left Elbow (13->15)
        rot_l_elbow, _ = calc_bone_rot(13, 15, VEC_HORIZONTAL_L, quaternion_inverse(rot_l_arm_global))
        
        # Right Arm (12->14)
        rot_r_arm, rot_r_arm_global = calc_bone_rot(12, 14, VEC_HORIZONTAL_R, rot_spine_inv)
        # Right Elbow (14->16)
        rot_r_elbow, _ = calc_bone_rot(14, 16, VEC_HORIZONTAL_R, quaternion_inverse(rot_r_arm_global))

        # LEGS
        # Left Leg (23->25)
        rot_l_leg, rot_l_leg_global = calc_bone_rot(23, 25, VEC_LEG_DOWN, rot_center_inv)
        # Left Knee (25->27)
        rot_l_knee, _ = calc_bone_rot(25, 27, VEC_LEG_DOWN, quaternion_inverse(rot_l_leg_global))
        
        # Right Leg (24->26)
        rot_r_leg, rot_r_leg_global = calc_bone_rot(24, 26, VEC_LEG_DOWN, rot_center_inv)
        # Right Knee (26->28)
        rot_r_knee, _ = calc_bone_rot(26, 28, VEC_LEG_DOWN, quaternion_inverse(rot_r_leg_global))
        
        # HEAD (Mid-Shoulder -> Nose 0)
        # Approximating neck/head rotation
        neck_vec = get_mp_vector(lm, 0, 0) # Dummy
        # Head vector is messy in MP. We use Shoulder Midpoint to Nose.
        nose_vec = np.array([lm[0].x - mid_shou_x, -(lm[0].y - mid_shou_y), -(lm[0].z - mid_shou_z)])
        rot_head = get_quaternion_between_vectors(VEC_VERTICAL_UP, nose_vec)
        rot_head_local = quaternion_multiply(rot_spine_inv, rot_head)


        # --- 3. WRITE TO VMD ---
        # Note: Some bones might need "damping" or locking if they jitter too much.
        
        writer.add_bone_frame("センター", frame_count, (0, 0, 0), rot_center) # Center
        writer.add_bone_frame("上半身", frame_count, (0, 0, 0), rot_spine_local) # Upper Body
        writer.add_bone_frame("頭", frame_count, (0, 0, 0), rot_head_local) # Head
        
        writer.add_bone_frame("左腕", frame_count, (0, 0, 0), rot_l_arm) # Left Arm
        writer.add_bone_frame("左ひじ", frame_count, (0, 0, 0), rot_l_elbow) # Left Elbow
        writer.add_bone_frame("右腕", frame_count, (0, 0, 0), rot_r_arm) # Right Arm
        writer.add_bone_frame("右ひじ", frame_count, (0, 0, 0), rot_r_elbow) # Right Elbow
        
        writer.add_bone_frame("左足", frame_count, (0, 0, 0), rot_l_leg) # Left Leg
        writer.add_bone_frame("左ひざ", frame_count, (0, 0, 0), rot_l_knee) # Left Knee
        writer.add_bone_frame("右足", frame_count, (0, 0, 0), rot_r_leg) # Right Leg
        writer.add_bone_frame("右ひざ", frame_count, (0, 0, 0), rot_r_knee) # Right Knee


    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
writer.write(OUTPUT_PATH)
print("Done!")