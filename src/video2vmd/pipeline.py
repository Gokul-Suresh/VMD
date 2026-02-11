from __future__ import annotations

from pathlib import Path

from .models import MotionTrack
from .pose_sources import PoseSource
from .skeleton import SkeletonRetargeter
from .vmd_writer import VmdWriter


class VideoToVmdPipeline:
    def __init__(
        self,
        pose_source: PoseSource,
        retargeter: SkeletonRetargeter | None = None,
        writer: VmdWriter | None = None,
    ) -> None:
        self.pose_source = pose_source
        self.retargeter = retargeter or SkeletonRetargeter()
        self.writer = writer or VmdWriter()

    def convert(self, video_path: Path, output_path: Path) -> MotionTrack:
        motion = MotionTrack()
        for pose_frame in self.pose_source.frames(video_path):
            motion.extend(self.retargeter.convert(pose_frame))
        self.writer.write(output_path, motion)
        return motion
