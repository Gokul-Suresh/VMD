from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable

from .models import BoneFrame, MotionTrack


def _encode_shift_jis(value: str, size: int) -> bytes:
    encoded = value.encode("shift_jis", errors="replace")
    return encoded[:size].ljust(size, b"\x00")


class VmdWriter:
    def __init__(self, model_name: str = "Video2VMD") -> None:
        self.model_name = model_name

    def write(self, path: Path, motion: MotionTrack) -> None:
        with path.open("wb") as f:
            f.write(_encode_shift_jis("Vocaloid Motion Data 0002", 30))
            f.write(_encode_shift_jis(self.model_name, 20))

            frames = sorted(motion.bone_frames, key=lambda x: (x.frame_index, x.bone_name))
            f.write(struct.pack("<I", len(frames)))
            for bf in frames:
                f.write(_encode_shift_jis(bf.bone_name, 15))
                f.write(struct.pack("<I", bf.frame_index))
                f.write(struct.pack("<fff", bf.position.x, bf.position.y, bf.position.z))
                f.write(struct.pack("<ffff", bf.rotation.x, bf.rotation.y, bf.rotation.z, bf.rotation.w))
                f.write(bytes([20] * 64))

            # Other keyframe sections (face/camera/light/shadow/IK) left empty.
            for _ in range(5):
                f.write(struct.pack("<I", 0))
