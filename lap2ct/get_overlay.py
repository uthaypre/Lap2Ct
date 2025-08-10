import open3d as o3d
import numpy as np
import copy


def get_overlay(organ_meshes):
    """
    Create an overlay point cloud from the organ meshes.
    
    Args:
        organ_meshes (dict): Dictionary of organ meshes with organ names as keys.

    Returns:
        o3d.geometry.PointCloud: Point cloud containing the overlay of all organs.
    """
    overlay_pcd = o3d.geometry.PointCloud()
    for organ, mesh in organ_meshes.items():
        single_mesh = copy.deepcopy(mesh)  # Clone the mesh to avoid modifying the original
        single_mesh.paint_uniform_color(np.array([0.9, 0.9, 0.9]))  # Set ct pc color to gray
        single_pcd = single_mesh.sample_points_uniformly(number_of_points=10000)
        overlay_pcd += single_pcd

    return overlay_pcd