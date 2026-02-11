from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scale: float) -> "Vec3":
        return Vec3(self.x * scale, self.y * scale, self.z * scale)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def magnitude(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

    def normalized(self) -> "Vec3":
        mag = self.magnitude()
        if mag == 0:
            return Vec3(0.0, 0.0, 0.0)
        return self * (1.0 / mag)


@dataclass(frozen=True)
class Quat:
    x: float
    y: float
    z: float
    w: float


@dataclass(frozen=True)
class PoseFrame:
    frame_index: int
    landmarks: Dict[str, Vec3]


@dataclass(frozen=True)
class BoneFrame:
    frame_index: int
    bone_name: str
    position: Vec3
    rotation: Quat


@dataclass
class MotionTrack:
    bone_frames: List[BoneFrame] = field(default_factory=list)

    def extend(self, frames: Iterable[BoneFrame]) -> None:
        self.bone_frames.extend(frames)
