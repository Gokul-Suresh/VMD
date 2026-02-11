from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, Iterator

from .models import PoseFrame, Vec3


class PoseSource(ABC):
    @abstractmethod
    def frames(self, video_path: Path) -> Iterable[PoseFrame]:
        raise NotImplementedError


class JsonPoseSource(PoseSource):
    """Reads precomputed pose landmarks for deterministic debugging."""

    def __init__(self, pose_json_path: Path) -> None:
        self.pose_json_path = pose_json_path

    def frames(self, video_path: Path) -> Iterable[PoseFrame]:
        _ = video_path
        payload = json.loads(self.pose_json_path.read_text(encoding="utf-8"))
        for frame in payload["frames"]:
            landmarks = {
                name: Vec3(*coords)
                for name, coords in frame["landmarks"].items()
            }
            yield PoseFrame(frame_index=frame["frame_index"], landmarks=landmarks)


class MediaPipePoseSource(PoseSource):
    """Adapter around MediaPipe pose. Optional dependency at runtime."""

    INDEX_TO_NAME: Dict[int, str] = {
        0: "head",
        11: "left_shoulder",
        12: "right_shoulder",
        13: "left_elbow",
        14: "right_elbow",
        15: "left_wrist",
        16: "right_wrist",
        23: "left_hip",
        24: "right_hip",
        25: "left_knee",
        26: "right_knee",
        27: "left_ankle",
        28: "right_ankle",
    }

    def frames(self, video_path: Path) -> Iterable[PoseFrame]:
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipePoseSource requires opencv-python and mediapipe installed"
            ) from exc

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        pose = mp.solutions.pose.Pose(static_image_mode=False)
        frame_index = 0
        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if result.pose_landmarks is None:
                    frame_index += 1
                    continue

                lmk = result.pose_landmarks.landmark
                landmarks: Dict[str, Vec3] = {}
                for idx, name in self.INDEX_TO_NAME.items():
                    p = lmk[idx]
                    landmarks[name] = Vec3(p.x, p.y, p.z)

                if "left_hip" in landmarks and "right_hip" in landmarks:
                    landmarks["hips_center"] = (
                        landmarks["left_hip"] + landmarks["right_hip"]
                    ) * 0.5
                if "left_shoulder" in landmarks and "right_shoulder" in landmarks:
                    landmarks["spine"] = (
                        landmarks["left_hip"] + landmarks["right_hip"]
                    ) * 0.25 + (
                        landmarks["left_shoulder"] + landmarks["right_shoulder"]
                    ) * 0.25
                    landmarks["neck"] = (
                        landmarks["left_shoulder"] + landmarks["right_shoulder"]
                    ) * 0.5
                yield PoseFrame(frame_index=frame_index, landmarks=landmarks)
                frame_index += 1
        finally:
            cap.release()
            pose.close()
