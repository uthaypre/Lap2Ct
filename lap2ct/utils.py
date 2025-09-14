import open3d as o3d
import numpy as np
from skimage import measure
import nibabel as nib
import open3d as o3d
import cv2
import math
import copy
import torch
import os
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
from models.DepthAnything.depth_anything.dpt import DepthAnything
from models.DepthAnything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import logging
import matplotlib
import sys
from pathlib import Path
# from models.DepthAnythingV2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from models.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2
from scipy.ndimage import binary_fill_holes
def setup_logger(log_level="INFO", log_file=False):
    """
    Set up logging configuration.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (str): Optional log file path. If None, logs to console only.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger('Lap2Ct')
    
    # Only configure if logger doesn't already have handlers
    if not logger.handlers:
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_file:
            # Create unique log file name with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_log_file = Path(f"logs/lap2ct_log_{timestamp}.log")
            unique_log_file.parent.mkdir(exist_ok=True)

            file_handler = logging.FileHandler(unique_log_file, mode='a')  # Append mode
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger

def get_color_from_name(name, label_name, label_colors):
    name_label = {n: l for l, n in label_name.items()}
    return label_colors[name_label[name]] 

def get_ball_marker(center, radius=5, n_points=2000, color=(1.0, 0.0, 0.0)):
    # 1) make a small sphere mesh
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)

    # 2) sample mesh → point cloud
    ball = mesh.sample_points_uniformly(number_of_points=n_points)

    # 3) move to desired center
    ball.translate(np.asarray(center), relative=True)
    return ball
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

def get_metric_depth(org_path):
    # model_configs = {
    #     'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    #     'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    #     'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    # }
    model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    encoder = 'vitl' # or 'vits', 'vitb'
    # dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    # max_depth = 1 # 20 for indoor model, 80 for outdoor model

    # model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    # model.load_state_dict(torch.load(f'models/DepthAnythingV2/metric_depth/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'models/DepthAnythingV2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(device).eval()

    raw_img = cv2.imread(str(org_path))
    depth = model.infer_image(raw_img) # HxW depth map in meters in numpy
    h, w = raw_img.shape[:2]
    
    # # Convert numpy depth to tensor for interpolation
    # depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    # depth_tensor = F.interpolate(depth_tensor, (h, w), mode='bilinear', align_corners=False)
    # depth = depth_tensor.squeeze().numpy()  # Remove dims and convert back to numpy
    
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    cmap = matplotlib.colormaps.get_cmap('magma') # Spectral_r
    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    # cv2.imwrite(os.path.join('/mnt/d/projectsD/datasets/LAP2CT/laparoscopy/', org_path.stem + '_depth.png'), depth * 255.0 / max_depth)  # Save depth map as PNG
    cv2.imwrite(os.path.join('/mnt/d/projectsD/datasets/LAP2CT/laparoscopy/', org_path.stem + '_depth.png'), depth)  # Save depth map as PNG
    return depth
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
        # single_mesh.paint_uniform_color(np.array([0.9, 0.9, 0.9]))  # Set ct pc color to gray
        single_pcd = single_mesh.sample_points_uniformly(number_of_points=10000)
        overlay_pcd += single_pcd

    return overlay_pcd
def get_3dlap(lap_path, label_classes=None, label_colors=None, logger=None):
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
    logger.info(f"depth values org: {min(depth_float.flatten())} - {max(depth_float.flatten())}")
    # max_depth = 300
    # depth_scale = max_depth / (depth_float.max() - depth_float.min())  # Scale depth values to [0, 300]
    # depth_float = (depth_float) * depth_scale
    logger.info(f"depth values scaled: {min(depth_float.flatten())} - {max(depth_float.flatten())}")
    colored_mask = np.zeros((lap_mask.shape[0], lap_mask.shape[1], 3))
    for label, color in label_colors.items():
        if int(label) == 0:
            continue  # Skip background
        colored_mask[lap_mask == int(label)] = color
    colored_mask = (colored_mask * 255).astype(np.uint8) 
    cv2.imwrite(os.path.join(base_dir, name_without_ext + "_colored_mask.png"), colored_mask)
    logger.debug(f"Colored mask contains these colors: {np.unique(colored_mask.reshape(-1, 3), axis=0)}")

    width, height = depth_float.shape[1], depth_float.shape[0]
    fov = 70.0  # Field of view in degrees
    fx = 0.5 * width / np.tan(0.5 * np.deg2rad(fov))
    fy = 0.5 * height / np.tan(0.5 * np.deg2rad(fov))
    cx = width / 2.0
    cy = height / 2.0
    logger.debug(f"Camera intrinsic parameters: fx={fx}, fy={fy}, cx={cx}, cy={cy}, fov={fov}, width={width}, height={height}")
    logger.debug(f"Depth image shape: {depth_float.shape}, Mask shape: {lap_mask.shape}")

    # Filter out each organ point cloud
    organ_pc_collection = {}
    organ_bool_lap = []
    for label, organ in label_classes.items():
        print(f"Processing organ: {organ} with label: {label} and color: {label_colors[label]}")
        organ_mask = (lap_mask == int(label)).astype(np.uint8)  # Create a mask for the specific organ
        # postprocess
        organ_mask = mask_postprocessing(organ_mask)
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
                        # depth_scale=0.10,  # No scaling needed since depth is already in millimeters
                        # depth_trunc=1000.0,
                        convert_rgb_to_intensity=False)
        single_seg_point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(single_rgbd_image,
                        intrinsic=o3d.camera.PinholeCameraIntrinsic(
                        width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy))
        
        # Apply coordinate system transformation to match medical imaging conventions
        # Open3D creates point clouds with Y pointing down (image coords)
        # We want Y pointing up for proper 3D visualization and registration
        # flip_transform = np.array([
        #     [1,  0,  0, 0],
        #     [0,  0, -1, 0],  # Flip Y-axis
        #     [0,  1,  0, 0],
        #     [0,  0,  0, 1]
        # ])
        single_seg_point_cloud.transform([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        
        # Add the organ point cloud to the collection
        organ_pc_collection[organ] = single_seg_point_cloud
        # logger.info(f"points: {np.asarray(single_seg_point_cloud.points[:30])}")


    
    # Visualize the point clouds for each organ
    try:
        # o3d.visualization.draw([pc for organ_name, pc in organ_pc_collection.items() if pc is not None and organ_name != "background"]+[o3d.geometry.TriangleMesh.create_coordinate_frame()])
        o3d.visualization.draw([pc for organ_name, pc in organ_pc_collection.items() if pc is not None and organ_name != "background"])
    except Exception as e:
        print(f"Error visualizing point clouds: {e}")
        return organ_pc_collection, organ_bool_lap
    
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
    nii = nib.load(ct_path)
    ct_mask = nii.get_fdata()
    A = nii.affine                       # 4x4
    ax = nib.aff2axcodes(A)              # e.g. ('L','P','S') or ('R','A','S')
    print(f"Image axes: {ax}  (affine maps voxel->scanner mm)")

    # # Optional LPS->RAS flip (scanner space)
    # LPS_to_RAS = np.diag([-1, -1, 1, 1])  # flips X,Y

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
        verts_xyz, faces, _, _ = measure.marching_cubes(organ_mask, level=0.5)  # Extract the mesh for the organ
        
        # verts_xyz = verts_zyx[:, [2, 1, 0]]
        # ones = np.ones((verts_xyz.shape[0], 1), dtype=np.float64)
        # V_h = np.hstack([verts_xyz, ones])                 # Nx4
        # V_mm = (A @ V_h.T).T[:, :3]                        # Nx3 in scanner mm
        # verts_xyz = A[:3, :3] @ verts_xyz.T + A[:3, -1]  # Convert to scanner mm

        # # 5) Optional: convert scanner LPS -> RAS
        # if ax == ('L', 'P', 'S'):
        #     V_mm_h = np.hstack([V_mm, np.ones((V_mm.shape[0], 1))])
        #     V_mm = (LPS_to_RAS @ V_mm_h.T).T[:, :3]
        V_mm = verts_xyz
        
        organ_mesh = o3d.geometry.TriangleMesh()
        organ_mesh.vertices = o3d.utility.Vector3dVector(V_mm)
        organ_mesh.triangles = o3d.utility.Vector3iVector(faces)
        organ_mesh.compute_vertex_normals()
        organ_mesh.paint_uniform_color(label_colors[label])  # Assign a random color to each organ
        organ_collection[organ] = organ_mesh  # Store the organ name and mesh
        cs_ct = organ_mesh.create_coordinate_frame(size=100)
        print(f"Organ: {organ}, label: {label}")

    # Visualize the organ meshes
    o3d.visualization.draw_geometries([organ_collection[organ] for organ in organ_collection]+[cs_ct])
    ## save point cloud
    # o3d.io.write_triangle_mesh(ct_path.parent / (Path(ct_path.stem).stem + ".obj"), [organ_collection[organ] for organ in organ_collection])  # Save the liver mesh as an example
    return organ_collection, organ_bool # both have length = 25

def make_fov_box(fov_x_deg, fov_y_deg, near=0.01, far=0.2):
    # Near/far planes in meters
    fov_x = math.radians(fov_x_deg)
    fov_y = math.radians(fov_y_deg)

    half_w_near = math.tan(fov_x/2) * near
    half_h_near = math.tan(fov_y/2) * near
    half_w_far  = math.tan(fov_x/2) * far
    half_h_far  = math.tan(fov_y/2) * far

    # Vertices of truncated pyramid (frustum)
    verts = np.array([
        [-half_w_near, -half_h_near, near],
        [ half_w_near, -half_h_near, near],
        [ half_w_near,  half_h_near, near],
        [-half_w_near,  half_h_near, near],
        [-half_w_far,  -half_h_far,  far],
        [ half_w_far,  -half_h_far,  far],
        [ half_w_far,   half_h_far,  far],
        [-half_w_far,   half_h_far,  far]
    ])
    
    lines = [
        [0,1],[1,2],[2,3],[3,0], # near
        [4,5],[5,6],[6,7],[7,4], # far
        [0,4],[1,5],[2,6],[3,7]  # sides
    ]
    
    frustum = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(verts),
        lines=o3d.utility.Vector2iVector(lines)
    )
    return frustum

def smooth_edges(mask2d: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    mask2d: 2D binary array (0/1 or 0/255)
    ksize : odd kernel size for uniform smoothing
    """
    m = (mask2d > 0).astype(np.float32)
    m = cv2.blur(m, (ksize, ksize))         # conv2D with uniform kernel
    return (m >= 0.5).astype(np.uint8)      # threshold at 0.5 (returns {0,1})


def mask_postprocessing(mask, erode_ksize=3,
                            dilate_ksize=3,
                            smooth_ksize=5):
    # clean up the mask

    erode_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_ksize, erode_ksize))
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
    bin_ = (mask > 0).astype(np.uint8)

    # 2) erode
    eroded = cv2.erode(bin_, erode_k, iterations=1)

    # 3) connected components 
    num, labels = cv2.connectedComponents(eroded, connectivity=8)

    if num > 1:
        # 4) histogram of component sizes (ignore 0 = background)
        areas = np.bincount(labels.ravel())[1:]  # length num-1
        keep_id = 1 + np.argmax(areas)          # label id to keep (largest)
        # 5) remove all labels except the largest region
        kept = (labels == keep_id).astype(np.uint8)
    else:
        kept = eroded  # nothing to choose, keep as-is

    # 6) dilate2D
    dilated = cv2.dilate(kept, dilate_k, iterations=1)

    # 7) fillholes (imfill)
    filled = binary_fill_holes(dilated.astype(bool)).astype(np.uint8)

    # 8) Smoothedges
    smooth = smooth_edges(filled, ksize=smooth_ksize)
    return smooth