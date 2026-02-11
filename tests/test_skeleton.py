from video2vmd.models import PoseFrame, Vec3
from video2vmd.skeleton import SkeletonRetargeter


def _upright_pose(frame_index: int, hips_x: float = 0.0) -> PoseFrame:
    return PoseFrame(
        frame_index=frame_index,
        landmarks={
            "hips_center": Vec3(hips_x, 0.0, 0.0),
            "spine": Vec3(hips_x, 0.2, 0.0),
            "neck": Vec3(hips_x, 0.4, 0.0),
            "head": Vec3(hips_x, 0.55, 0.0),
            "left_shoulder": Vec3(hips_x + 0.1, 0.4, 0.0),
            "left_elbow": Vec3(hips_x + 0.25, 0.4, 0.0),
            "left_wrist": Vec3(hips_x + 0.4, 0.4, 0.0),
            "right_shoulder": Vec3(hips_x - 0.1, 0.4, 0.0),
            "right_elbow": Vec3(hips_x - 0.25, 0.4, 0.0),
            "right_wrist": Vec3(hips_x - 0.4, 0.4, 0.0),
            "left_hip": Vec3(hips_x + 0.08, 0.0, 0.0),
            "left_knee": Vec3(hips_x + 0.08, -0.3, 0.0),
            "left_ankle": Vec3(hips_x + 0.08, -0.6, 0.0),
            "right_hip": Vec3(hips_x - 0.08, 0.0, 0.0),
            "right_knee": Vec3(hips_x - 0.08, -0.3, 0.0),
            "right_ankle": Vec3(hips_x - 0.08, -0.6, 0.0),
        },
    )


def test_only_center_writes_position_offsets():
    retargeter = SkeletonRetargeter(center_scale=10.0)

    f0 = list(retargeter.convert(_upright_pose(0, hips_x=0.0)))
    f1 = list(retargeter.convert(_upright_pose(1, hips_x=0.2)))

    center_0 = next(b for b in f0 if b.bone_name == "センター")
    center_1 = next(b for b in f1 if b.bone_name == "センター")

    assert center_0.position == Vec3(0.0, 0.0, 0.0)
    assert abs(center_1.position.x - 2.0) < 1e-6

    for bone in f1:
        if bone.bone_name != "センター":
            assert bone.position == Vec3(0.0, 0.0, 0.0)
