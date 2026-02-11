from video2vmd.models import Vec3
from video2vmd.rotation import quat_from_two_vectors


def test_quat_from_two_vectors_identity():
    q = quat_from_two_vectors(Vec3(0, 1, 0), Vec3(0, 1, 0))
    assert abs(q.w - 1.0) < 1e-6


def test_quat_from_two_vectors_90deg():
    q = quat_from_two_vectors(Vec3(1, 0, 0), Vec3(0, 1, 0))
    assert abs(q.w - 0.7071) < 1e-2
