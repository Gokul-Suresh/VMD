import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def get_quaternion_between_vectors(u, v):
    """
    Calculate the rotation (quaternion) needed to rotate vector u to align with vector v.
    """
    u = normalize(u)
    v = normalize(v)

    # Dot product gives the cosine of the angle
    dot = np.dot(u, v)

    # If vectors are parallel (dot == 1), no rotation needed
    if dot > 0.999999:
        return np.array([0, 0, 0, 1]) # Identity quaternion (x,y,z,w)
    
    # If vectors are opposite (dot == -1), 180 degree rotation
    elif dot < -0.999999:
        axis = np.cross(np.array([1, 0, 0]), u)
        if np.linalg.norm(axis) < 0.00001:
            axis = np.cross(np.array([0, 1, 0]), u)
        axis = normalize(axis)
        return np.array([axis[0], axis[1], axis[2], 0])

    # Standard case
    axis = np.cross(u, v)
    w = 1 + dot
    
    # Normalize the quaternion
    q = np.array([axis[0], axis[1], axis[2], w])
    return normalize(q)

def quaternion_multiply(q1, q0):
    """
    Multiplies two quaternions q1 * q0
    Expected format: [x, y, z, w]
    """
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    
    return np.array([
        w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0,
        w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0,
        w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0,
        w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0
    ])

def quaternion_inverse(q):
    """
    Returns the inverse of a quaternion. 
    For unit quaternions, this is just [-x, -y, -z, w]
    """
    return np.array([-q[0], -q[1], -q[2], q[3]])

def get_mp_vector(landmarks, idx_start, idx_end):
    """
    Gets the raw vector from MediaPipe World Landmarks.
    MediaPipe World: X=Right, Y=Down, Z=Camera Depth
    """
    s = landmarks[idx_start]
    e = landmarks[idx_end]
    
    # We must flip Y and Z to match MMD's coordinate system (Right Handed Y-Up)
    # MMD: X=Left(-)/Right(+), Y=Down(-)/Up(+), Z=Front(-)/Back(+)
    start_vec = np.array([s.x, -s.y, -s.z])
    end_vec   = np.array([e.x, -e.y, -e.z])
    
    return end_vec - start_vec