import open3d as o3d
import numpy as np
from pathlib import Path
import os
import cv2

def get_3dlap(lap_path, label_classes=None, label_colors=None):
    # Load the segmentation mask
    lap_mask = cv2.imread(lap_path)

    lap_mask = cv2.cvtColor(lap_mask, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if needed
    
    # Construct depth image path and load it
    base_dir = os.path.dirname(lap_path)
    filename = os.path.basename(lap_path)
    name_without_ext = os.path.splitext(filename)[0]
    # Remove one more extension if it exists (e.g., file.nii.gz -> file)
    name_without_ext = os.path.splitext(name_without_ext)[0]
    depth_path = os.path.join(base_dir, name_without_ext + "_0000_depth.png")
    print(f"Looking for depth image at: {depth_path}")
    depth_image = cv2.imread(depth_path)
    if depth_image is None:
        raise FileNotFoundError(f"Could not read depth file: {depth_path}")
    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if needed
    depth_float = depth_image.astype(np.float32)

    colored_mask = np.zeros((lap_mask.shape[0], lap_mask.shape[1], 3))
    for label, color in label_colors.items():
        if int(label) == 0:
            continue  # Skip background
        colored_mask[lap_mask == int(label)] = color
    colored_mask = (colored_mask * 255).astype(np.uint16) 

    width, height = depth_float.shape[1], depth_float.shape[0]
    fov = 55.0  # Field of view in degrees
    fx = 0.5 * width / np.tan(0.5 * np.deg2rad(fov))
    fy = 0.5 * height / np.tan(0.5 * np.deg2rad(fov))
    cx = width / 2.0
    cy = height / 2.0

    # Filter out each organ point cloud
    organ_pc_collection = {}
    organ_bool_lap = []
    for label, organ in label_classes.items():
        organ_mask = (lap_mask == int(label)).astype(np.uint8)  # Create a mask for the specific organ
        # Check if organ exists in the data
        if np.sum(organ_mask) == 0:
            print(f"Skipping {organ} (label {label}) - not present in this scan")
            organ_bool_lap.append(False)
            organ_pc_collection[label] = None
            continue
        organ_bool_lap.append(True)

        single_colored_mask = np.zeros((organ_mask.shape[0], organ_mask.shape[1], 3))
        single_colored_mask[organ_mask == 1] = label_colors[label]  # Assign color to the organ mask
        single_colored_mask = (single_colored_mask * 255).astype(np.uint8)
        organ_depth = depth_float * organ_mask  # Apply the organ mask to the depth image
        single_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        color=o3d.geometry.Image(single_colored_mask),
                        depth=o3d.geometry.Image(organ_depth),
                        depth_trunc=1000.0,
                        convert_rgb_to_intensity=False)
        single_seg_point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(single_rgbd_image,
                        intrinsic=o3d.camera.PinholeCameraIntrinsic(
                        width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy))
        # Add the organ point cloud to the collection
        organ_pc_collection[label] = single_seg_point_cloud

    # Visualize the point clouds for each organ
    o3d.visualization.draw_geometries([pc for pc in organ_pc_collection.values() if pc is not None], mesh_show_back_face=True)

    # Save individual organ point clouds or combine them
    valid_point_clouds = [pc for pc in organ_pc_collection.values() if pc is not None]
    if valid_point_clouds:
        # Get base directory and filename without extension
        base_dir = os.path.dirname(lap_path)
        filename = os.path.basename(lap_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        # Option 1: Save combined point cloud
        combined_pc = valid_point_clouds[0]
        for pc in valid_point_clouds[1:]:
            combined_pc += pc
        combined_path = os.path.join(base_dir, name_without_ext + "_combined.ply")
        o3d.io.write_point_cloud(combined_path, combined_pc)
        
        # # Option 2: Save individual organ point clouds
        # for label, pc in organ_pc_collection.items():
        #     if pc is not None:
        #         organ_name = label_classes.get(label, f"organ_{label}")
        #         organ_path = os.path.join(base_dir, name_without_ext + f"_{organ_name}.ply")
        #         o3d.io.write_point_cloud(organ_path, pc)

    return organ_pc_collection, organ_bool_lap