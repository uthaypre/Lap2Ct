import os
import yaml
import cv2
import numpy as np
from easydict import EasyDict as edict
import open3d as o3d
import matplotlib
from PIL import Image, ImageFile
from typing import Optional
import os
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import time
import torch
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt
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


def get_metric_depth(org_path):
    print("start metric depth")
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
    depth_for_save = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    # cv2.imwrite(os.path.join('/mnt/d/projectsD/datasets/LAP2CT/laparoscopy/', org_path.stem + '_depth.png'), depth * 255.0 / max_depth)  # Save depth map as PNG
    filename_base = os.path.basename(org_path)
    cv2.imwrite(os.path.join('/mnt/d/projectsD/datasets/depth_val_test/DepthAnything', filename_base[:filename_base.rfind('.')] + '_DAmetric_depth.png'), depth_for_save)
    print("inside metric depth -> finito")
    return depth

def get_depthmap(img_path, outdir="/mnt/d/projectsD/datasets/depth_val_test/DepthAnything", encoder='vitl', grayscale=False):
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
    print("start relativedepth ")
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
        
        cmap = matplotlib.colormaps.get_cmap('magma') # Spectral_r
        depth_for_save = (cmap(depth_result)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        filename_base = os.path.basename(filename)
        cv2.imwrite(os.path.join(outdir, filename_base[:filename_base.rfind('.')] + '_DA_depth.png'), depth_for_save)
        print("inside relative depth-> finito")
    return depth_result

def endo_3D(lap_path, label_classes=None, label_colors=None, logger=None):
    # Load the segmentation mask
    image = cv2.imread(lap_path)
    if image is None:
        print(f"Error: Could not load image from {lap_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to grayscale if needed

    gt = cv2.imread("/mnt/d/projectsD/datasets/depth_val_test/0000000054_gt.png")
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

    # depth_path = "/mnt/d/projectsD/ml-depth-pro/data/image_21534_depth.png/image_21534_0000_.jpg"
    depth_path = "/mnt/d/projectsD/datasets/depth_val_test/output/image_21534_0000_depth.png"
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        print(f"Error: Could not load depth image from {depth_path}")
        return
        # Handle different depth image formats
    if len(depth_image.shape) == 3:
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
# Ensure both images have the same dimensions
    height, width = image_rgb.shape[:2]
    depth_height, depth_width = depth_image.shape[:2]
    
    if (height, width) != (depth_height, depth_width):
        print(f"Warning: Resizing depth image from {depth_width}x{depth_height} to {width}x{height}")
        depth_image = cv2.resize(depth_image, (width, height))
    
    # Convert to proper data types
    image_rgb = image_rgb.astype(np.uint8)
    depth_image = depth_image.astype(np.uint16)  # Open3D expects uint16 for depth
    
    # Create Open3D images with proper format
    color_o3d = o3d.geometry.Image(image_rgb)
    depth_o3d = o3d.geometry.Image(depth_image)
    
    print(f"Color image shape: {image_rgb.shape}, dtype: {image_rgb.dtype}")
    print(f"Depth image shape: {depth_image.shape}, dtype: {depth_image.dtype}")
    
    # Camera intrinsics
    fov = 70.0  # Field of view in degrees
    fx = 0.5 * width / np.tan(0.5 * np.deg2rad(fov))
    fy = 0.5 * height / np.tan(0.5 * np.deg2rad(fov))
    cx = width / 2.0
    cy = height / 2.0
    
    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color_o3d,
        depth=depth_o3d,
        depth_scale=1.0,  # Adjust based on your depth units
        depth_trunc=3000.0,  # Adjust max depth
        convert_rgb_to_intensity=False
    )
    
    # Create intrinsic parameters
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy
    )
    
    # Create point cloud
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic
    )
    # Visualize
    o3d.visualization.draw_geometries([
        point_cloud, 
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    ])    

# def depth_read(path: str, dtype=np.float16, depth_shift: float = 1000.0) -> np.ndarray:
#     """Reads an image from a png file and returns the depth image shifted to meters.
#     In Scannet by default the depth_shift is 1000.0

#     Arguments:
#         path {str} -- Path to the png file.

#     Keyword Arguments:
#         dtype  --  (default: {np.float16})
#         depth_shift {float} -- Value used to divide the values on the png.
#             (default: {100.0})

#     Returns:
#         np.ndarray -- Numpy array cointaining the depth image.
#     """

#     depth_image = np.array(Image.open(path)).astype(dtype) / depth_shift
#     return depth_image
def depth_to_color(
    depth_np: np.ndarray,
    cmap: str = "gist_rainbow",
    max_depth: Optional[float] = None,
    min_depth: Optional[float] = None,
) -> np.ndarray:
    """Converts a depth image to color using the specified color map.

    Arguments:
        depth_np {np.ndarray} -- Depth image/Or inverse depth image. [HxW]
    Keyword Arguments:
        cmap {str} -- Color map to be used. (default: {"gist_rainbow"})

    Returns:
        np.ndarray -- Color image [HxWx3]
    """
    # -- Set default arguments
    depth_np[np.isinf(depth_np)] = np.nan
    if max_depth is None:
        max_depth = np.nanmax(depth_np)
    if min_depth is None:
        min_depth = min_depth or np.nanmin(depth_np)

    cm = plt.get_cmap(cmap, lut=1000)
    depth_np_norm = (depth_np - min_depth) / (max_depth - min_depth)
    colored_depth = cm(depth_np_norm)

    return (colored_depth[:, :, :3] * 255).astype(np.uint8)


if __name__ == "__main__":
    # Load a depth image and turn into 3d Point cloud
    # gt = cv2.imread("/mnt/d/projectsD/datasets/rectified06/rectified06/depth01/0000001236.png", cv2.IMREAD_UNCHANGED)
    depth_path = "/mnt/d/projectsD/datasets/depth_val_test/0000001236_gt.png"
    depth_np = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000

    # print(np.unique(gt))
    # print(gt.shape, gt.dtype, gt.min(), gt.max())
    # depth=depth_read("/mnt/d/projectsD/datasets/rectified06/rectified06/depth01/0000001236.png", dtype=np.float32, depth_shift=1000.0)
    # vmin = gt.min()
    # vmax = np.percentile(gt, 99)
    # normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma_r')
    # colormapped_im = (mapper.to_rgba(gt)[:, :, :3] * 255).astype(np.uint8)
    # # depth = (gt - gt.min()) / (gt.max() - gt.min()) * 255.0
    # # depth = depth.astype(np.uint8)
    # # cmap = matplotlib.colormaps.get_cmap('magma') # Spectral_r
    # # colormapped_im = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im = depth_to_color(
    depth_np=depth_np,
    cmap="jet_r",
    max_depth=np.percentile(depth_np, 99),
    min_depth=depth_np.min(),
)
    im = pil.fromarray(colormapped_im)
    output_file = "/mnt/d/projectsD/datasets/depth_val_test/colored_depth__func.png"
    im.save(output_file)
    
    # # print(np.unique(depth))
    # # cmap = matplotlib.colormaps.get_cmap('magma') # Spectral_r
    # # depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    # # cv2.imwrite(os.path.join('/mnt/d/projectsD/datasets/LAP2CT/laparoscopy/', org_path.stem + '_depth.png'), depth * 255.0 / max_depth)  # Save depth map as PNG
    # # cv2.imwrite(os.path.join('/mnt/d/projectsD/datasets/depth_val_test/', 'colored_depth.png'), depth)  # Save depth map as PNG
   
    # # with open("configs/organ_segmentation.yaml", 'r') as f:
    # #     seg_config = yaml.safe_load(f)
    # # seg_config = edict(seg_config)
    # # endo_3D("/mnt/d/projectsD/datasets/depth_val_test/image_21534_0000_.png", seg_config.ct_organ_classes, seg_config.ct_organ_colors)

    # # Create depth maps
    # lap_depth_map = get_depthmap("/mnt/d/projectsD/datasets/depth_val_test/0000001236.png", encoder='vitl', grayscale=False)
    # gt_depth_map = get_metric_depth("/mnt/d/projectsD/datasets/depth_val_test/0000001236.png")

    # # # # create ground truth from scared dataset TIFF files
    # # # import tifffile as tiff

    # # # arr = tiff.imread("/mnt/d/projectsD/datasets/depth_val_test/left_depth_map.tiff")  # works fine
    # # # # print(arr.shape, arr.dtype, arr.min(), arr.max())
    # # # # print(np.unique(arr))
    # # # # Handle invalid values (NaN, inf)
    # # # arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    # # # # Normalize to 0-255 range before converting to uint8
    # # # image = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) / 255.0
    # # # if arr.max() > arr.min():  # Avoid division by zero
    # # #     depth_normalized = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
    # # # else:
    # # #     depth_normalized = np.zeros_like(arr)
    # # # print(np.unique(depth_normalized))
    # # # depth = depth_normalized.astype(np.uint8)
    # # # cmap = matplotlib.colormaps.get_cmap('magma') # Spectral_r
    # # # depth_for_save = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    # # # cv2.imwrite(os.path.join("/mnt/d/projectsD/datasets/depth_val_test", 'Left_Image_gt.png'), depth_for_save)


