import numpy as np
import matplotlib.pyplot as plt

def read_xyz_q_positions(file_path):
    """
    Reads positions from a file where each line is:
        qw qx qy qz x y z
    Returns Nx3 positions (no filtering).
    """
    positions = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            # positions are last 3 entries
            x, y, z = map(float, parts[4:7])
            positions.append([x, y, z])
    return np.array(positions)

# Path to your quaternion+position file
positions_file = "E1_xyz_q.txt"
subsample = 2  # take every 10th point

# Read positions (no filtering)
positions = read_xyz_q_positions(positions_file)

positions = positions
# Subsample for trajectory
trajectory = positions[::subsample]

# Plot trajectory in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(trajectory[:,0], trajectory[:,1], trajectory[:,2],
          c='blue', linewidth=1.5, label='Trajectory (E14_xyz_q)')

ax.set_xlabel('X (horizontal)')
ax.set_ylabel('Y (horizontal)')
ax.set_zlabel('Z (vertical)')
ax.set_title('Trajectory from E14_xyz_q.txt (Z = vertical, subsample=10)')
ax.legend()
ax.grid(True)

# Equal scaling for X/Y/Z
max_range = np.array([trajectory[:,0].max()-trajectory[:,0].min(),
                      trajectory[:,1].max()-trajectory[:,1].min(),
                      trajectory[:,2].max()-trajectory[:,2].min()]).max() / 2.0

mid_x = (trajectory[:,0].max()+trajectory[:,0].min()) * 0.5
mid_y = (trajectory[:,1].max()+trajectory[:,1].min()) * 0.5
mid_z = (trajectory[:,2].max()+trajectory[:,2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()

trajectory_colmap=trajectory
import numpy as np
import matplotlib.pyplot as plt

def read_E13_tf(file_path):
    trajectory = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if "Translation:" in line:
                try:
                    t_str = line.split("Translation:")[1].split("|")[0]
                    t_parts = t_str.strip().split(",")
                    x = float(t_parts[0].split("=")[1])
                    y = float(t_parts[1].split("=")[1])
                    z = float(t_parts[2].split("=")[1])
                    trajectory.append([x, y, z])
                except:
                    continue
    return np.array(trajectory)

# Read trajectory
trajectory_file = "E1_tf.txt"
trajectory = read_E13_tf(trajectory_file)

# Rotate 90 degrees around Y-axis
Rx_90 = np.array([[1, 0, 0],
                  [0, 0, -1],
                  [0, 1, 0]])
trajectory_rot_tf = (Rx_90 @ trajectory.T).T

# Center over Z
mid_z = (trajectory_rot_tf[:,2].max() + trajectory_rot_tf[:,2].min()) * 0.5
trajectory_rot_tf[:,2] -= mid_z

# Plot trajectory
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory_rot_tf[:,0], trajectory_rot_tf[:,1], trajectory_rot_tf[:,2],
        label='Estimated Trajectory (rotated & Z-centered)', color='red', linewidth=2)
ax.plot3D(trajectory_colmap[:,0], trajectory_colmap[:,1], trajectory_colmap[:,2],
          c='blue', linewidth=1.5, label='Trajectory (E13_xyz_q)')

ax.set_xlabel('X (horizontal)')
ax.set_ylabel('Y (horizontal)')
ax.set_zlabel('Z (vertical)')
ax.set_title('Trajectory from E13_tf.txt (rotated 90° around Y, centered over Z)')
ax.legend()
ax.grid(True)

# Equal scaling
max_range = np.array([trajectory_rot_tf[:,0].max()-trajectory_rot_tf[:,0].min(),
                      trajectory_rot_tf[:,1].max()-trajectory_rot_tf[:,1].min(),
                      trajectory_rot_tf[:,2].max()-trajectory_rot_tf[:,2].min()]).max() / 2.0

mid_x = (trajectory_rot_tf[:,0].max()+trajectory_rot_tf[:,0].min()) * 0.5
mid_y = (trajectory_rot_tf[:,1].max()+trajectory_rot_tf[:,1].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(-max_range, max_range)  # Z centered at 0

plt.show()


import numpy as np

# -----------------------------
# Umeyama alignment
# -----------------------------
def umeyama_alignment(src, dst, with_scaling=True):
    """
    Align src to dst using Umeyama method.
    Returns: scale, rotation matrix, translation vector
    """
    assert src.shape == dst.shape
    N = src.shape[0]
    mean_src = src.mean(axis=0)
    mean_dst = dst.mean(axis=0)
    src_centered = src - mean_src
    dst_centered = dst - mean_dst

    cov = np.dot(dst_centered.T, src_centered) / N
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    scale = 1.0
    if with_scaling:
        var_src = np.sum(src_centered**2) / N
        scale = np.trace(np.diag(D) @ S) / var_src
    t = mean_dst - scale * R @ mean_src
    return scale, R, t

def apply_umeyama(src, dst):
    scale, R, t = umeyama_alignment(src, dst)
    aligned = scale * (R @ src.T).T + t
    return aligned, scale, R, t

# -----------------------------
# Compute ATE and RPE
# -----------------------------
def compute_ate(aligned, reference):
    error = np.linalg.norm(aligned - reference, axis=1)
    rmse = np.sqrt(np.mean(error**2))
    return rmse

def compute_rpe(aligned, reference):
    rpe = []
    for i in range(len(aligned)-1):
        delta_est = aligned[i+1] - aligned[i]
        delta_ref = reference[i+1] - reference[i]
        rpe.append(np.linalg.norm(delta_est - delta_ref))
    return np.array(rpe)


# -----------------------------
# Align trajectory_colmap to trajectory_rot_tf using Umeyama
# -----------------------------
# Make sure both trajectories have same length for alignment
n = min(len(trajectory_colmap), len(trajectory_rot_tf))
traj_src = trajectory_colmap[:n]       # colmap trajectory (source)
traj_dst = trajectory_rot_tf[:n]       # TF trajectory (target/reference)

print(len(trajectory_colmap), len(trajectory_rot_tf))
# Apply Umeyama alignment
traj_aligned, scale, R, t = apply_umeyama(traj_src, traj_dst)

print(f"Scale applied to colmap trajectory: {scale:.6f}")

# Center both trajectories around Z = 0
mid_z_dst = (traj_dst[:,2].max() + traj_dst[:,2].min()) * 0.5
traj_dst_centered = traj_dst.copy()
traj_dst_centered[:,2] -= mid_z_dst

mid_z_aligned = (traj_aligned[:,2].max() + traj_aligned[:,2].min()) * 0.5
traj_aligned[:,2] -= mid_z_aligned

# -----------------------------
# Plot aligned trajectories
# -----------------------------
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# TF trajectory (reference)
ax.plot(traj_dst_centered[:,0], traj_dst_centered[:,1], traj_dst_centered[:,2],
        label='Trajectory TF (reference)', color='red', linewidth=2)

# Colmap trajectory (aligned)
ax.plot(traj_aligned[:,0], traj_aligned[:,1], traj_aligned[:,2],
        label='Trajectory Colmap (aligned)', color='blue', linewidth=2, alpha=0.8)

# Labels and title
ax.set_xlabel('X (horizontal)')
ax.set_ylabel('Y (horizontal)')
ax.set_zlabel('Z (vertical)')
ax.set_title('Aligned Trajectories (Z = vertical)')
#ax.legend()
ax.grid(True)

# Equal scaling
all_points = np.vstack([traj_aligned, traj_dst_centered])
max_range = (all_points.max(axis=0) - all_points.min(axis=0)).max() / 2.0
mid_x = (all_points[:,0].max() + all_points[:,0].min()) * 0.5
mid_y = (all_points[:,1].max() + all_points[:,1].min()) * 0.5
mid_z = (all_points[:,2].max() + all_points[:,2].min()) * 0.5

margin = 0.05  # 5% margin around the points
all_points = np.vstack([traj_aligned, traj_dst_centered])

min_vals = all_points.min(axis=0)
max_vals = all_points.max(axis=0)
ranges = max_vals - min_vals

ax.set_xlim(min_vals[0] - margin*ranges[0], max_vals[0] + margin*ranges[0])
ax.set_ylim(min_vals[1] - margin*ranges[1], max_vals[1] + margin*ranges[1])
ax.set_zlim(min_vals[2] - margin*ranges[2], max_vals[2] + margin*ranges[2])

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()

traj_aligned, scale, R, t = apply_umeyama(traj_src, traj_dst)
print(f"Scale applied to colmap trajectory: {scale:.6f}")

# -----------------------------
# Compute ATE and RPE
# -----------------------------
ate = compute_ate(traj_aligned, traj_dst)
rpe_values = compute_rpe(traj_aligned, traj_dst)
mean_rpe = np.mean(rpe_values)

print(f"Absolute Trajectory Error (ATE): {ate:.6f}")
print(f"Mean Relative Pose Error (RPE): {mean_rpe:.6f}")




