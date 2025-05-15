import numpy as np
import cv2
from sklearn.cluster import DBSCAN

def pixel_to_ray(pixel_coords, intrinsics):
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u, v = pixel_coords
    x = (u - cx) / fx
    y = (v - cy) / fy
    ray = np.array([x, y, 1.0])
    return ray / np.linalg.norm(ray)

def transform_points(points, transform_matrix):
    n = points.shape[0]
    homo_points = np.hstack((points, np.ones((n, 1))))
    transformed = (transform_matrix @ homo_points.T).T
    return transformed[:, :3]

def project_lidar_to_image(lidar_points, intrinsics, extrinsics):
    lidar_in_cam = transform_points(lidar_points, extrinsics)
    in_front = lidar_in_cam[:, 2] > 0
    lidar_in_cam = lidar_in_cam[in_front]
    pts_2d = intrinsics @ lidar_in_cam.T
    pts_2d = (pts_2d[:2] / pts_2d[2]).T
    return pts_2d, lidar_in_cam

def get_points_in_bbox(projected_pts, original_pts, bbox):
    x, y, w, h = bbox
    mask = (projected_pts[:, 0] > x) & (projected_pts[:, 0] < x + w) & \
           (projected_pts[:, 1] > y) & (projected_pts[:, 1] < y + h)
    return original_pts[mask]

def get_lidar_bbox_from_points(points, cluster_eps=0.5, min_samples=5):
    if len(points) < min_samples:
        return []
    db = DBSCAN(eps=cluster_eps, min_samples=min_samples).fit(points)
    labels = db.labels_
    bboxes = []
    for lbl in set(labels):
        if lbl == -1:
            continue
        cluster = points[labels == lbl]
        min_pt = np.min(cluster, axis=0)
        max_pt = np.max(cluster, axis=0)
        bbox = np.hstack((min_pt, max_pt - min_pt))  # x, y, z, dx, dy, dz
        bboxes.append(bbox)
    return bboxes

def process_single_frame(lidar_points, image_bboxes, intrinsics, cam_to_lidar):
    pts_2d, pts_3d_cam = project_lidar_to_image(lidar_points, intrinsics, cam_to_lidar)
    all_lidar_bboxes = []
    for bbox in image_bboxes:
        points_in_box = get_points_in_bbox(pts_2d, pts_3d_cam, bbox)
        lidar_bboxes = get_lidar_bbox_from_points(points_in_box)
        all_lidar_bboxes.extend(lidar_bboxes)
    return np.array(all_lidar_bboxes)
