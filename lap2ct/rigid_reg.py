import open3d as o3d
import SimpleITK as sitk
import numpy as np
from skimage import measure
import cv2
import copy
import nibabel as nib

def draw_registration_result(source, target):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # o3d.visualization.draw_geometries([source_temp, target_temp],
                                    #   zoom=0.4559,
                                    #   front=[0.6452, -0.3036, -0.7011],
                                    #   lookat=[1.9892, 2.0208, 1.8945],
                                    #   up=[-0.2779, -0.9482, 0.1556])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(source=None, target=None, voxel_size=0.05):
    draw_registration_result(source, target)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),  # Enable scaling
        4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(5000000, 0.999))
    return result

def init_transformation(source, target):
    """
    Initialize the transformation matrix for registration.
    
    Parameters:
    - source: Source point cloud (Open3D PointCloud).
    - target: Target point cloud (Open3D PointCloud).
    
    Returns:
    - transformation: Initial transformation matrix.
    """
    source_center = np.mean(np.asarray(source.points), axis=0)
    target_center = np.mean(np.asarray(target.points), axis=0)
    
    translation = target_center - source_center
    transformation = np.eye(4)
    transformation[:3, 3] = translation

    source_min = np.min(np.asarray(source.points), axis=0)
    target_min = np.min(np.asarray(target.points), axis=0)
    source_max = np.max(np.asarray(source.points), axis=0)
    target_max = np.max(np.asarray(target.points), axis=0)
    source_scale = np.linalg.norm(source_max - source_min)
    target_scale = np.linalg.norm(target_max - target_min) 
    scale = target_scale / source_scale
    transformation[:3, :3] *= scale
    source.transform(transformation)
    
    return transformation


def rigid_reg(source, target, voxel_size=1.817):
    """
    Perform rigid registration of two point clouds.
    
    Parameters:
    - source: Source point cloud (Open3D PointCloud).
    - target: Target point cloud (Open3D PointCloud).
    - voxel_size: Voxel size for downsampling and feature computation.
    
    Returns:
    - result: Registration result containing the transformation matrix.
    """
    # draw_registration_result(source, target, np.identity(4))
    init_transform = init_transformation(source, target)
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source, target,
        voxel_size)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print(result_ransac)
    source_transformed = source_down.transform(result_ransac.transformation)
    # draw_registration_result(source_down, target_down)
    return source_transformed, init_transform, result_ransac