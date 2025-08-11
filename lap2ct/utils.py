import open3d as o3d
import numpy as np
from skimage import measure
import nibabel as nib
import open3d as o3d
import cv2
import copy
import torch
import os
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
from models.DepthAnything.depth_anything.dpt import DepthAnything
from models.DepthAnything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

def get_depthmap(img_path, outdir='./vis_depth', encoder='vitl', grayscale=False):
    """
    Run Depth Anything inference on images.
    
    Args:
        img_path (str): Path to image file or directory
        outdir (str): Output directory for results
        encoder (str): Encoder type ('vits', 'vitb', 'vitl')
        grayscale (bool): Output grayscale depth maps
        
    Returns:
        depth (np.ndarray): Depth map of the input image (for single image) or last processed image
    """
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE).eval()
    
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    if os.path.isfile(img_path):
        if str(img_path).endswith('txt'):
            with open(img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [img_path]
    else:
        print("filenames:", filenames, "Type:", type(filenames))
        filenames = os.listdir(img_path)
        print("Filenames:", filenames, "Type:", type(filenames))
        filenames = [os.path.join(img_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    
    os.makedirs(outdir, exist_ok=True)
    
    depth_result = None  # Initialize result variable
    
    for filename in tqdm(filenames):
        raw_image = cv2.imread(str(filename))
        if raw_image is None:
            print(f"Warning: Could not read image {filename}")
            continue
            
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        h, w = image.shape[:2]
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            depth = depth_anything(image)
        
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        
        depth_result = depth.cpu().numpy().astype(np.uint8)  # Store the depth result
        
        if grayscale:
            depth_for_save = np.repeat(depth_result[..., np.newaxis], 3, axis=-1)
        else:
            depth_for_save = cv2.applyColorMap(depth_result, cv2.COLORMAP_INFERNO)
        
        filename_base = os.path.basename(filename)
        cv2.imwrite(os.path.join(outdir, filename_base[:filename_base.rfind('.')] + '_depth.png'), depth_for_save)

    return depth_result


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
        print(f"Processing organ: {organ} with label: {label} and color: {label_colors[label]}")
        organ_mask = (lap_mask == int(label)).astype(np.uint8)  # Create a mask for the specific organ
        # Check if organ exists in the data
        if np.sum(organ_mask) == 0:
            print(f"Skipping {organ} (label {label}) - not present in this scan")
            organ_bool_lap.append(False)
            # organ_pc_collection[organ] = None
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
        organ_pc_collection[organ] = single_seg_point_cloud

    # # Visualize the point clouds for each organ
    # try:
    #     o3d.visualization.draw_geometries([pc for pc in organ_pc_collection.values() if pc is not None], mesh_show_back_face=True)
    # except Exception as e:
    #     print(f"Error visualizing point clouds: {e}")
    #     return organ_pc_collection, organ_bool_lap
    
    # Save individual organ point clouds or combine them
    valid_point_clouds = [pc for pc in organ_pc_collection.values() if pc is not None]
    if valid_point_clouds:
        # Get base directory and filename without extension
        base_dir = os.path.dirname(lap_path)
        filename = os.path.basename(lap_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        # # Option 1: Save combined point cloud
        # combined_pc = valid_point_clouds[0]
        # for pc in valid_point_clouds[1:]:
        #     combined_pc += pc
        # combined_path = os.path.join(base_dir, name_without_ext + "_combined.ply")
        # o3d.io.write_point_cloud(combined_path, combined_pc)
        
        # # Option 2: Save individual organ point clouds
        # for organ_name, pc in organ_pc_collection.items():
        #     if pc is not None:
        # #         organ_name = label_classes.get(label, f"organ_{label}")
        #         organ_path = os.path.join(base_dir, name_without_ext + f"_{organ_name}.ply")
        #         o3d.io.write_point_cloud(organ_path, pc)
        print(organ_pc_collection)
    return organ_pc_collection, organ_bool_lap
def get_3dct(ct_path, label_classes=None, label_colors=None):
    ct_mask = nib.load(ct_path).get_fdata()

    organ_collection = {}
    organ_bool = []
    for label, organ in label_classes.items():
        organ_mask = (ct_mask == int(label)).astype(np.uint8)  # Create a mask for the specific organ
        # Check if organ exists in the data
        if np.sum(organ_mask) == 0:
            print(f"Skipping {organ} (label {label}) - not present in this scan")
            organ_bool.append(False)
            organ_collection[organ] = None
            continue
        organ_bool.append(True)
        verts, faces, _, _ = measure.marching_cubes(organ_mask, level=0.5)  # Extract the mesh for the organ

        organ_mesh = o3d.geometry.TriangleMesh()
        organ_mesh.vertices = o3d.utility.Vector3dVector(verts)
        organ_mesh.triangles = o3d.utility.Vector3iVector(faces)
        organ_mesh.compute_vertex_normals()
        organ_mesh.paint_uniform_color(label_colors[label])  # Assign a random color to each organ
        organ_collection[organ] = organ_mesh  # Store the organ name and mesh
        print(f"Organ: {organ}, label: {label}")

    # Visualize the organ meshes
    # o3d.visualization.draw_geometries([organ_collection[organ] for organ in organ_collection])
    ## save point cloud
    # o3d.io.write_triangle_mesh(ct_path.parent / (Path(ct_path.stem).stem + ".obj"), [organ_collection[organ] for organ in organ_collection])  # Save the liver mesh as an example
    return organ_collection, organ_bool # both have length = 25