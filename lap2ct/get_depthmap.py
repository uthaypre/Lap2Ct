import open3d as o3d
import numpy as np
import cv2
import torch
import sys
import os
import argparse
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
from models.DepthAnything.depth_anything.dpt import DepthAnything

# # Add the DepthAnything directory to Python path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# depth_anything_path = os.path.join(current_dir, '..', 'models', 'DepthAnything')
# sys.path.insert(0, os.path.abspath(depth_anything_path))

# # Now import DepthAnything
# try:
#     from depth_anything.dpt import DepthAnything
# except ImportError as e:
#     print(f"Import error: {e}")
#     print(f"Trying to import from: {depth_anything_path}")
#     print(f"Contents of directory: {os.listdir(depth_anything_path) if os.path.exists(depth_anything_path) else 'Directory not found'}")
#     raise
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

    margin_width = 50
    caption_height = 60
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
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
