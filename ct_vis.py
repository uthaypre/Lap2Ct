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

nii = nib.load("/mnt/d/projectsD/datasets/LAP2CT/ct/input_ct_001.nii.gz")
nii_gt = nib.load("/mnt/c/Users/uthaypre/OneDrive - ZHAW/Master/Masterarbeit/data/labelsTs_023/BDMAP_00000005.nii.gz")
nii_org = nib.load("/mnt/d/projectsD/datasets/nnUNet/nnUNet_raw/Dataset023_AbdomenAtlas1.1Mini/imagesTs/input_ct_001_0000.nii.gz")
ct_mask = nii.get_fdata()
gt = nii_gt.get_fdata()
org = nii_org.get_fdata()


# CT org unsegmented mesh
verts_xyz_org, faces_org, _, _ = measure.marching_cubes(org, level=100)
org_mesh = o3d.geometry.TriangleMesh()
org_mesh.vertices = o3d.utility.Vector3dVector(verts_xyz_org)
org_mesh.triangles = o3d.utility.Vector3iVector(faces_org)
org_mesh.compute_vertex_normals()
org_mesh = org_mesh.filter_smooth_taubin(number_of_iterations=10)
o3d.visualization.draw_geometries([org_mesh])
