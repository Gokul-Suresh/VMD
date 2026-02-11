# Video2VMD (simple + debuggable)

This repo provides a **simple, modular workflow** for converting tracked video poses into a `.vmd` animation file.

## Goals

- Track all major body bones (torso, head, arms, legs)
- Track rotations robustly with quaternion math
- Keep each step isolated so errors are easy to inspect/fix

## Pipeline

1. **Pose source** (`pose_sources.py`)
   - `MediaPipePoseSource`: runs inference on video (requires `opencv-python` + `mediapipe`)
   - `JsonPoseSource`: reads precomputed landmarks from JSON for deterministic debugging
2. **Skeleton retargeting** (`skeleton.py`)
   - Maps landmarks to MMD-like bone names
   - Computes bone rotation from rest direction to tracked direction
   - Smooths rotations over time to reduce jitter
3. **VMD writing** (`vmd_writer.py`)
   - Writes motion frames into VMD 0002 structure

## Install

```bash
pip install -e .
```

## Run

### From a video (normal mode)

```bash
python -m video2vmd.cli input.mp4 output.vmd
```

### From debug JSON (repeatable troubleshooting)

```bash
python -m video2vmd.cli input.mp4 output.vmd --pose-json sample_pose.json
```

## JSON format for debug mode

```json
{
  "frames": [
    {
      "frame_index": 0,
      "landmarks": {
        "hips_center": [0.0, 0.0, 0.0],
        "spine": [0.0, 0.2, 0.0],
        "neck": [0.0, 0.4, 0.0],
        "head": [0.0, 0.55, 0.0],
        "left_shoulder": [0.1, 0.4, 0.0],
        "left_elbow": [0.25, 0.35, 0.0],
        "left_wrist": [0.4, 0.3, 0.0],
        "right_shoulder": [-0.1, 0.4, 0.0],
        "right_elbow": [-0.25, 0.35, 0.0],
        "right_wrist": [-0.4, 0.3, 0.0],
        "left_hip": [0.08, 0.0, 0.0],
        "left_knee": [0.08, -0.3, 0.0],
        "left_ankle": [0.08, -0.6, 0.0],
        "right_hip": [-0.08, 0.0, 0.0],
        "right_knee": [-0.08, -0.3, 0.0],
        "right_ankle": [-0.08, -0.6, 0.0]
      }
    }
  ]
}
```


## Troubleshooting

- If your model starts upside-down or mirrored on frame 0, verify your input video is upright and from a single frontal camera.
- MediaPipe landmarks are camera-space (`+y` downward), while MMD-style rigs usually expect `+y` upward. This project converts landmark axes before retargeting to reduce flipped first-frame poses.

## Why this is easy to debug

- Swap media inference out for `JsonPoseSource`
- Unit-test math independently (`rotation.py`)
- Unit-test retarget logic separately (`skeleton.py`)
- Keep VMD writer separate from tracking
