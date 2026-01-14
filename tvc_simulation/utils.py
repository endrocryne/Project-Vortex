"""
Utility functions for quaternion mathematics and coordinate transformations.
"""

import numpy as np


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions (scalar-first convention: [w, x, y, z]).
    
    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]
    
    Returns:
        Resulting quaternion q1 ⊗ q2
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quaternion_normalize(q):
    """
    Normalize a quaternion to unit length.
    
    Args:
        q: Quaternion [w, x, y, z]
    
    Returns:
        Normalized quaternion
    """
    # Check for non-finite values (NaN or Inf)
    if not np.all(np.isfinite(q)):
        # Return identity quaternion if input is invalid
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to 3x3 rotation matrix (Inertial to Body).
    
    Args:
        q: Quaternion [w, x, y, z] mapping Inertial to Body
    
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = q
    
    return np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x**2 + z**2),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])


def quaternion_conjugate(q):
    """
    Compute quaternion conjugate.
    
    Args:
        q: Quaternion [w, x, y, z]
    
    Returns:
        Conjugate quaternion [w, -x, -y, -z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_derivative(q, omega):
    """
    Compute quaternion derivative given angular velocity in body frame.
    Implements: dq/dt = 0.5 * q ⊗ [0, ω]
    
    Args:
        q: Current quaternion [w, x, y, z]
        omega: Angular velocity in body frame [ωx, ωy, ωz]
    
    Returns:
        Quaternion derivative dq/dt
    """
    # Create pure quaternion from omega: [0, ωx, ωy, ωz]
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
    
    # dq/dt = 0.5 * q ⊗ omega_quat
    return 0.5 * quaternion_multiply(q, omega_quat)


def quaternion_to_euler(q):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw) in radians.
    
    Args:
        q: Quaternion [w, x, y, z]
    
    Returns:
        Array [roll, pitch, yaw] in radians
    """
    w, x, y, z = q
    
    # Roll (φ)
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    
    # Pitch (θ)
    sin_pitch = 2*(w*y - z*x)
    sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    
    # Yaw (ψ)
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    
    return np.array([roll, pitch, yaw])


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion.
    
    Args:
        roll: Roll angle in radians
        pitch: Pitch angle in radians
        yaw: Yaw angle in radians
    
    Returns:
        Quaternion [w, x, y, z]
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])


def rotate_vector_by_quaternion(v, q):
    """
    Rotate a vector from Inertial frame to Body frame using quaternion.
    
    Args:
        v: Vector in Inertial frame [x, y, z]
        q: Quaternion [w, x, y, z] mapping Inertial to Body
    
    Returns:
        Rotated vector in Body frame
    """
    # Convert to rotation matrix and apply
    R = quaternion_to_rotation_matrix(q)
    return R @ v


def cross_product_matrix(v):
    """
    Create skew-symmetric matrix for cross product operation.
    [v]× such that [v]×u = v × u
    
    Args:
        v: Vector [x, y, z]
    
    Returns:
        3x3 skew-symmetric matrix
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
