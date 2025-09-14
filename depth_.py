# # # import cv2
# # # import torch

# # # from models.DepthAnythingV2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

# # # model_configs = {
# # #     'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
# # #     'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
# # #     'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
# # # }

# # # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # # encoder = 'vitl' # or 'vits', 'vitb'
# # # dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
# # # max_depth = 20 # 20 for indoor model, 80 for outdoor model

# # # model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
# # # chk = torch.load(f'models/DepthAnythingV2/metric_depth/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu')
# # # model.load_state_dict(chk, strict=True)
# # # model.to(device)
# # # if hasattr(model, 'pretrained'):     # DepthAnythingV2 keeps a backbone here
# # #     model.pretrained.to(device)

# # # model.eval()

# # # raw_img = cv2.imread('/mnt/d/projectsD/datasets/nnUNet/nnUNet_raw/Dataset333_DSAD/imagesTs/image_10624_0000.png')
# # # with torch.inference_mode():
# # #     depth = model.infer_image(raw_img)   # returns HxW depth in meters (numpy)
# # # cv2.imshow('Depth Map', depth)
# # # cv2.waitKey(0)
# # # cv2.destroyAllWindows()

import cv2
import torch

from models.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'models/DepthAnythingV2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread('/mnt/d/projectsD/datasets/nnUNet/nnUNet_raw/Dataset333_DSAD/imagesTs/image_21504_0000.png')
depth = model.infer_image(raw_img) # HxW raw depth map in numpy
cv2.imwrite('/mnt/d/projectsD/datasets/nnUNet/nnUNet_raw/Dataset333_DSAD/imagesTs/image_21504_0000_depth.png', depth)

# import open3d as o3d

# test = o3d.io.read_point_cloud("/mnt/d/projectsD/datasets/nnUNet/nnUNet_raw/Dataset333_DSAD/imagesTs/image_21504_0000.ply")
# o3d.visualization.draw_geometries([test])

