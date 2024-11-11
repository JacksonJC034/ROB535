"""
Code written by Joey Wilson, 2023.
"""

import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans


# Given two Nx3 and Mx3 point clouds and initial SE(3) matrix
# Register using open3d and return SE(3) registration matrix
# Refer to open3d geometry.PointCloud
# As well as open3d registration.registration_icp
# See the following tutorial for help
# http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
# Grade will be based on the fitness score compared to our solution
def register_clouds(xyz_source, xyz_target, trans_init=None):
  if trans_init is None:
    trans_init = np.eye(4)
  threshold = 0.1
  max_iters = 300

  # TODO: create o3d pointclouds for the source and target
  source = o3d.geometry.PointCloud()
  source.points = o3d.utility.Vector3dVector(xyz_source)
  target = o3d.geometry.PointCloud()
  target.points = o3d.utility.Vector3dVector(xyz_target)

  # Pre-registration similarity
  evaluation_pre = o3d.pipelines.registration.evaluate_registration(
    source, target, threshold, trans_init)
  print("Before registration:", evaluation_pre)

  # TODO: register the point clouds using registration_icp
  reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iters))

  evaluation_post = o3d.pipelines.registration.evaluate_registration(
    source, target, threshold, reg_p2p.transformation)
  print("After registration:", evaluation_post)

  # TODO: obtain the transformation matrix from reg_p2p
  reg_mat = reg_p2p.transformation
  return reg_mat


# Given two point clouds Nx3, corresponding N labels, and registration matrix
# Transform the points of the source (time T-1) to the pose at frame T
# Return combined points (N+M)x3 and labels (N+M)
def combine_clouds(xyz_source, xyz_target, labels_source, labels_target, reg_mat):
  # TODO: Apply transformation matrix on xyz_source
  xyz_source_hom = np.hstack((xyz_source, np.ones((xyz_source.shape[0], 1))))
  xyz_transformed_hom = (reg_mat @ xyz_source_hom.T).T
  xyz_transformed = xyz_transformed_hom[:, :3]
  # TODO: Concatenate xyz and labels
  # Note: Concatenate transformed followed by target
  xyz_all = np.vstack((xyz_transformed, xyz_target))
  label_all = np.hstack((labels_source, labels_target))
  return xyz_all, label_all


# Mask to return points in only the vehicle and bus class
# Check the yaml file to identify which labels correspond to vehicle and bus
# Note that labels are mapped to training set, meaning in range 0 to 20
def mask_dynamic(xyz, label):
  # TODO: Create a mask such that label == vehicle or label == bus
  dynamic_labels = [1, 5]
  dynamic_mask = np.isin(label, dynamic_labels)
  return xyz[dynamic_mask, :], label[dynamic_mask]


# Similarly, mask out the vehicle and bus class to return only the static points
def mask_static(xyz, label):
  # TODO: Create a mask opposite of above
  dynamic_labels = [1, 5]
  static_mask = ~np.isin(label, dynamic_labels)
  return xyz[static_mask, :], label[static_mask]


# Given an Nx3 matrix of points and a Cx3 matrix of clusters
# Return for each point the index of the nearest cluster
# For efficiency, useful functions include np.tile, np.linalg.norm, and np.argmin  
def cluster_dists(xyz, clusters):
  N, __ = xyz.shape
  xyz = xyz.reshape(N, 1, 3)
  C = clusters.shape[0]
  clusters = clusters.reshape(1, C, 3)
  # TODO: Create assignments between each point and the closest cluster
  diffs = xyz - clusters  # Shape: (N, C, 3)
  sq_dists = np.sum(diffs ** 2, axis=2)  # Shape: (N, C)
  closest_clusts = np.argmin(sq_dists, axis=1)  # Shape: (N,)
  return closest_clusts


# Given Nx3 points and N assignments (for each point, the index of the cluster)
# Calculate the new centroids of each cluster
# Return centroids shape Cx3
def new_centroids(xyz, assignments, C):
  new_instances = np.zeros((C, 3))
  # TODO: Calculate new clusters by the assignments
  for c in range(C):
        # Get all points assigned to cluster c
        assigned_points = xyz[assignments == c]
        if assigned_points.shape[0] > 0:
            # Compute the mean of these points
            new_instances[c] = np.mean(assigned_points, axis=0)
        else:
            # Handle empty clusters if necessary (e.g., reinitialize)
            new_instances[c] = xyz[np.random.randint(0, xyz.shape[0])]
  return new_instances


# Returns an integer corresponding to the number of instances of dynamic vehicles
# Shown in the point cloud
def num_instances():
  # TODO: Return the number of instances
  return 5


# K means algorithm. Given points, calculates and returns clusters.
def cluster(xyz):
  C = num_instances()
  rng = np.random.default_rng(seed=1)
  instances = xyz[rng.choice(xyz.shape[0], size=C, replace=False), :]
  prev_assignments = rng.choice(C, size=xyz.shape[0])
  while True:
    assignments = cluster_dists(xyz, instances)
    instances = new_centroids(xyz, assignments, C)
    if (assignments == prev_assignments).all():
      return instances, assignments
    prev_assignments = assignments


# Sci-kit learn implementation which is more advanced. 
# Try changing the random seed in cluster (will not be tested) and observe 
# how the clusters change. 
# Then try running scikit learn clustering with different random states. 
def cluster_sci(xyz):
  kmeans = KMeans(n_clusters=num_instances(), random_state=5, n_init="auto").fit(xyz)
  clustered_labels = kmeans.predict(xyz)
  return kmeans.cluster_centers_, clustered_labels

# Given Nx3 point cloud, 3x3 camera intrinsic matrix, and 4x4 LiDAR to Camera 
# Compute image points Nx2 and depth in camera-frame per point
def to_pixels(xyz, P, RT):
  N = xyz.shape[0]
  # Convert to camera frame
  xyz_homogeneous = np.hstack((xyz, np.ones((N, 1))))  # Shape: (N, 4)
  transformed_points_homogeneous = (RT @ xyz_homogeneous.T).T  # Shape: (N, 4)
  transformed_points = transformed_points_homogeneous[:, :3]   # Shape: (N, 3)

  # Depth in camera frame
  d = transformed_points[:, 2]  # Shape: (N,)

  # Use intrinsic matrix
  image_points_homogeneous = (P @ transformed_points.T).T  # Shape: (N, 3)
  image_x = image_points_homogeneous[:, 0] / image_points_homogeneous[:, 2]
  image_y = image_points_homogeneous[:, 1] / image_points_homogeneous[:, 2]

  imgpoints = np.stack([image_x, image_y], axis=1)
  return imgpoints, d