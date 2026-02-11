from pathlib import Path

from video2vmd.pipeline import VideoToVmdPipeline
from video2vmd.pose_sources import JsonPoseSource


def test_pipeline_generates_vmd(tmp_path: Path):
    pose_json = tmp_path / "pose.json"
    pose_json.write_text(
        """
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
                "left_elbow": [0.25, 0.4, 0.0],
                "left_wrist": [0.4, 0.4, 0.0],
                "right_shoulder": [-0.1, 0.4, 0.0],
                "right_elbow": [-0.25, 0.4, 0.0],
                "right_wrist": [-0.4, 0.4, 0.0],
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
        """,
        encoding="utf-8",
    )
    output = tmp_path / "out.vmd"

    pipeline = VideoToVmdPipeline(pose_source=JsonPoseSource(pose_json))
    motion = pipeline.convert(tmp_path / "dummy.mp4", output)

    assert output.exists()
    assert len(motion.bone_frames) >= 10
