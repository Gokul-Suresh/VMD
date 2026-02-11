from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import VideoToVmdPipeline
from .pose_sources import JsonPoseSource, MediaPipePoseSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track a video into a VMD motion file")
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("output", type=Path, help="Output .vmd path")
    parser.add_argument(
        "--pose-json",
        type=Path,
        help="Optional precomputed pose json file for deterministic debugging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pose_source = (
        JsonPoseSource(args.pose_json)
        if args.pose_json
        else MediaPipePoseSource()
    )
    pipeline = VideoToVmdPipeline(pose_source=pose_source)
    motion = pipeline.convert(args.video, args.output)
    print(f"Wrote {len(motion.bone_frames)} bone frames to {args.output}")


if __name__ == "__main__":
    main()
