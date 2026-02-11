from types import SimpleNamespace

from video2vmd.models import Vec3
from video2vmd.pose_sources import MediaPipePoseSource


def test_landmark_to_vec3_flips_y_and_z_axes():
    landmark = SimpleNamespace(x=0.25, y=0.75, z=-0.5)

    got = MediaPipePoseSource._landmark_to_vec3(landmark)

    assert got == Vec3(0.25, -0.75, 0.5)
