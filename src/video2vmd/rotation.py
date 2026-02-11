from __future__ import annotations

from .models import Quat, Vec3


def quat_normalize(q: Quat) -> Quat:
    mag = (q.x**2 + q.y**2 + q.z**2 + q.w**2) ** 0.5
    if mag == 0:
        return Quat(0.0, 0.0, 0.0, 1.0)
    return Quat(q.x / mag, q.y / mag, q.z / mag, q.w / mag)


def quat_from_two_vectors(src: Vec3, dst: Vec3) -> Quat:
    """Build a quaternion rotating src to dst.

    Uses numerically stable half-angle method.
    """
    a = src.normalized()
    b = dst.normalized()
    dot = a.dot(b)

    if dot < -0.999999:
        axis = Vec3(1.0, 0.0, 0.0).cross(a)
        if axis.magnitude() < 1e-6:
            axis = Vec3(0.0, 1.0, 0.0).cross(a)
        axis = axis.normalized()
        return Quat(axis.x, axis.y, axis.z, 0.0)

    cross = a.cross(b)
    quat = Quat(cross.x, cross.y, cross.z, 1.0 + dot)
    return quat_normalize(quat)


def smooth_quat(prev: Quat | None, current: Quat, alpha: float) -> Quat:
    if prev is None:
        return current
    blended = Quat(
        x=prev.x * (1.0 - alpha) + current.x * alpha,
        y=prev.y * (1.0 - alpha) + current.y * alpha,
        z=prev.z * (1.0 - alpha) + current.z * alpha,
        w=prev.w * (1.0 - alpha) + current.w * alpha,
    )
    return quat_normalize(blended)
