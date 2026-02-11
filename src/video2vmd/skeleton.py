from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .models import BoneFrame, PoseFrame, Quat, Vec3
from .rotation import quat_from_two_vectors, smooth_quat

# Bone name, parent landmark, child landmark, rest direction.
BONE_SPECS: List[Tuple[str, str, str, Vec3]] = [
    ("センター", "hips_center", "spine", Vec3(0.0, 1.0, 0.0)),
    ("上半身", "spine", "neck", Vec3(0.0, 1.0, 0.0)),
    ("頭", "neck", "head", Vec3(0.0, 1.0, 0.0)),
    ("左腕", "left_shoulder", "left_elbow", Vec3(1.0, 0.0, 0.0)),
    ("左ひじ", "left_elbow", "left_wrist", Vec3(1.0, 0.0, 0.0)),
    ("右腕", "right_shoulder", "right_elbow", Vec3(-1.0, 0.0, 0.0)),
    ("右ひじ", "right_elbow", "right_wrist", Vec3(-1.0, 0.0, 0.0)),
    ("左足", "left_hip", "left_knee", Vec3(0.0, -1.0, 0.0)),
    ("左ひざ", "left_knee", "left_ankle", Vec3(0.0, -1.0, 0.0)),
    ("右足", "right_hip", "right_knee", Vec3(0.0, -1.0, 0.0)),
    ("右ひざ", "right_knee", "right_ankle", Vec3(0.0, -1.0, 0.0)),
]


@dataclass
class SkeletonRetargeter:
    rotation_smoothing: float = 0.4
    center_scale: float = 15.0

    def __post_init__(self) -> None:
        self._last_rotation: Dict[str, Quat] = {}
        self._center_origin: Vec3 | None = None

    def convert(self, pose_frame: PoseFrame) -> Iterable[BoneFrame]:
        out: List[BoneFrame] = []
        for bone_name, parent_key, child_key, rest_dir in BONE_SPECS:
            parent = pose_frame.landmarks.get(parent_key)
            child = pose_frame.landmarks.get(child_key)
            if parent is None or child is None:
                continue

            tracked_dir = (child - parent).normalized()
            raw_rotation = quat_from_two_vectors(rest_dir, tracked_dir)
            rotation = smooth_quat(
                self._last_rotation.get(bone_name),
                raw_rotation,
                self.rotation_smoothing,
            )
            self._last_rotation[bone_name] = rotation
            if bone_name == "センター":
                if self._center_origin is None:
                    self._center_origin = parent
                position = (parent - self._center_origin) * self.center_scale
            else:
                # Bone keyframes should primarily carry rotation. Writing raw
                # tracked positions for every bone distorts the rig because VMD
                # positions are local offsets from model bind pose.
                position = Vec3(0.0, 0.0, 0.0)

            out.append(
                BoneFrame(
                    frame_index=pose_frame.frame_index,
                    bone_name=bone_name,
                    position=position,
                    rotation=rotation,
                )
            )
        return out
