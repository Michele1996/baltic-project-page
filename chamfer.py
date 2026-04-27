import os
import numpy as np
import struct
import open3d as o3d
import csv
from scipy.spatial import cKDTree


# ------------------------------------------------------------
# COLMAP points3D.bin reader
# ------------------------------------------------------------
def read_points3d_binary(path):
    points = []
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            f.read(8)  # point id
            xyz = struct.unpack("<ddd", f.read(24))
            f.read(3)  # rgb
            f.read(8)  # error
            track_length = struct.unpack("<Q", f.read(8))[0]
            f.read(track_length * 8)
            points.append(xyz)
    return np.array(points)


# ------------------------------------------------------------
# Bounding box scale estimation
# ------------------------------------------------------------
def estimate_scale_bbox(src_pts, ref_pts):
    src_min = np.min(src_pts, axis=0)
    src_max = np.max(src_pts, axis=0)
    ref_min = np.min(ref_pts, axis=0)
    ref_max = np.max(ref_pts, axis=0)

    src_diag = np.linalg.norm(src_max - src_min)
    ref_diag = np.linalg.norm(ref_max - ref_min)

    return ref_diag / src_diag


# ------------------------------------------------------------
# RMS scale estimation (more robust)
# ------------------------------------------------------------
def estimate_scale_rms(src_pts, ref_pts):
    src_centered = src_pts - np.mean(src_pts, axis=0)
    ref_centered = ref_pts - np.mean(ref_pts, axis=0)

    src_rms = np.sqrt(np.mean(np.sum(src_centered**2, axis=1)))
    ref_rms = np.sqrt(np.mean(np.sum(ref_centered**2, axis=1)))

    return ref_rms / src_rms


# ------------------------------------------------------------
# Chamfer Distance
# ------------------------------------------------------------
def chamfer_distance(p1, p2):
    tree1 = cKDTree(p1)
    tree2 = cKDTree(p2)

    d1, _ = tree1.query(p2)
    d2, _ = tree2.query(p1)

    return np.mean(d1**2) + np.mean(d2**2)


# ------------------------------------------------------------
# Surface Roughness
# ------------------------------------------------------------
def compute_surface_roughness(pcd, radius=0.01):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=30
        )
    )

    kdtree = o3d.geometry.KDTreeFlann(pcd)
    pts = np.asarray(pcd.points)
    residuals = []

    for i in range(len(pts)):
        [_, idx, _] = kdtree.search_radius_vector_3d(pts[i], radius)
        if len(idx) < 5:
            continue

        neighbors = pts[idx]
        centroid = np.mean(neighbors, axis=0)
        cov = np.cov((neighbors - centroid).T)
        eigvals, _ = np.linalg.eigh(cov)

        residuals.append(np.min(eigvals))

    return np.mean(residuals)


# ------------------------------------------------------------
# Mean NN distance
# ------------------------------------------------------------
def compute_nn_distance(points):
    tree = cKDTree(points)
    dist, _ = tree.query(points, k=2)
    return np.mean(dist[:, 1])


# ------------------------------------------------------------
# Load and downsample point cloud
# ------------------------------------------------------------
def load_pointcloud(folder, voxel_size=0.005):
    pts = read_points3d_binary(os.path.join(folder, "points3D.bin"))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    return pcd.voxel_down_sample(voxel_size)


# ------------------------------------------------------------
# Batch comparison
# ------------------------------------------------------------
def batch_compare(base_folder=".", ref_dataset="E1"):

    ref_path = os.path.join(base_folder, ref_dataset, "sparse/0")
    ref_pcd = load_pointcloud(ref_path)
    ref_pts = np.asarray(ref_pcd.points)

    results = []

    for i in range(1, 15):

        dataset = f"E{i}"
        if dataset == ref_dataset:
            continue

        print(f"\nProcessing {dataset}")

        folder = os.path.join(base_folder, dataset, "sparse/0")
        if not os.path.exists(folder):
            print("Folder missing — skipping")
            continue

        pcd = load_pointcloud(folder)
        pts = np.asarray(pcd.points)

        # ---------------------------
        # SCALE ESTIMATION
        # ---------------------------
        scale_bbox = estimate_scale_bbox(pts, ref_pts)
        scale_rms = estimate_scale_rms(pts, ref_pts)

        # Apply RMS scale (more stable)
        pts_scaled = pts * scale_rms
        pcd.points = o3d.utility.Vector3dVector(pts_scaled)

        # ---------------------------
        # ICP refinement (rigid only)
        # ---------------------------
        reg = o3d.pipelines.registration.registration_icp(
            pcd,
            ref_pcd,
            0.05,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        pcd.transform(reg.transformation)
        pts_aligned = np.asarray(pcd.points)

        # ---------------------------
        # METRICS
        # ---------------------------
        cd = chamfer_distance(ref_pts, pts_aligned)
        mean_dev_mm = np.sqrt(cd) * 1000

        roughness = compute_surface_roughness(pcd)
        nn_mm = compute_nn_distance(pts_aligned) * 1000

        results.append([
            dataset,
            scale_bbox,
            scale_rms,
            abs(1 - scale_bbox) * 100,
            abs(1 - scale_rms) * 100,
            reg.fitness,
            reg.inlier_rmse,
            cd,
            mean_dev_mm,
            roughness,
            nn_mm
        ])

    # ---------------------------
    # SAVE CSV
    # ---------------------------
    with open("reconstruction_metrics_sim3_corrected.csv", "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Dataset",
            "Scale_BBox",
            "Scale_RMS",
            "Scale_Drift_BBox_%",
            "Scale_Drift_RMS_%",
            "ICP_Fitness",
            "ICP_RMSE",
            "Chamfer_Distance_m",
            "Chamfer_RMS_mm",
            "Surface_Roughness",
            "Mean_NN_Distance_mm"
        ])

        writer.writerows(results)

    print("\nSaved to reconstruction_metrics_sim3_corrected.csv")


# ------------------------------------------------------------
if __name__ == "__main__":
    batch_compare()
