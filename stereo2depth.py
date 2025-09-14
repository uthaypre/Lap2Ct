# import numpy as np
# import matplotlib
# import cv2 as cv
# from matplotlib import pyplot as plt
# from scaredtk.calibrator import StereoCalibrator, undistort
# from pathlib import Path
# from scaredtk.calibrator import StereoCalibrator, undistort
# import scaredtk.io as sio
# import scaredtk.convertions as cvt
# import argparse
# from tqdm import tqdm
# # kf = Path('/mnt/d/projectsD/datasets/2019-selected_4-6/dataset_4/keyframe_5')
# # stereo_calib = StereoCalibrator()
# # calib = stereo_calib.load(kf/'endoscope_calibration.yaml')
# # gt_ptcloud = sio.load_scared_obj(kf/'left_point_cloud.obj')
# # gt_img3d = sio.load_img3d(kf/'left_depth_map.tiff')
# # depthmap = cvt.img3d_to_depthmap(gt_img3d)
# # left_img= cv.imread(str(kf/'Left_Image.png'))
# # outdir = Path('/mnt/d/projectsD/datasets/depth_val_test')
# # scale_factor = 128.0
# # sio.save_subpix_png(outdir/'Left_Image_GT_depthmap.png',
# #                     depthmap, scale_factor)
# # img = cv.imread(str(outdir/'Left_Image_GT_depthmap.png'))

# # depth = (img - img.min()) / (img.max() - img.min()) * 255.0

# # depth_result = (depth).astype(np.uint8) # Store the depth result

# # cmap = matplotlib.colormaps.get_cmap('magma')  # Spectral_r
# # depth_for_save = (cmap(depth_result)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
# # cv.imwrite(str(outdir/'Left_Image_GT_depthmap_colored.png'), depth_for_save)

# import numpy as np
# import imageio.v3 as iio
# import matplotlib.pyplot as plt
# from pathlib import Path
# import open3d as o3d
# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
# from pathlib import Path

# # png_path = Path("/mnt/d/projectsD/datasets/depth_val_test/Left_Image_GT_depthmap.png")
# # out_path = Path("/mnt/d/projectsD/datasets/depth_val_test/Left_Image_GT_depthmap_colored.png")
# # scale_factor = 128.0  # match what you used when saving via scaredtk

# # d16 = cv.imread(str(png_path), cv.IMREAD_UNCHANGED)   # <-- preserves 16-bit, single-channel
# # if d16 is None:
# #     raise RuntimeError("Failed to read depth PNG.")
# # print(d16.dtype, d16.shape, d16.min(), d16.max())
# # depth = d16.astype(np.float32) / scale_factor         # back to metric units
# # depth[d16 == 0] = np.nan                              # restore unknowns (toolkit stores NaNs as 0) :contentReference[oaicite:4]{index=4}

# # valid = np.isfinite(depth)
# # vmin, vmax = np.percentile(depth[valid], [0.025, 99.975])
# # norm = np.zeros_like(depth, dtype=np.float32)
# # norm[valid] = np.clip((depth[valid] - vmin) / (vmax - vmin + 1e-8), 0, 1) * 255
# # depth = norm.astype(np.uint8)
# # rgba = plt.get_cmap("magma")(depth)
# # depth_for_save = (rgba[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
# # rgba[~valid, 3] = 0.0
# # cv.imwrite(str(out_path), (rgba * 255).astype(np.uint8))  # OpenCV will write RGBA just fine
# # print("saved:", out_path)
# # print("min/max:", np.nanmin(rgba), np.nanmax(rgba))
# # print("percentiles:", np.nanpercentile(rgba, [0, 50, 100]))

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt

# # imgL = cv.imread('/mnt/d/projectsD/datasets/2019-selected_4-6/dataset_4/keyframe_5/Left_Image.png', cv.IMREAD_GRAYSCALE)
# # imgR = cv.imread('/mnt/d/projectsD/datasets/2019-selected_4-6/dataset_4/keyframe_5/Right_Image.png', cv.IMREAD_GRAYSCALE)

# # stereo = cv.StereoBM.create(numDisparities=16, blockSize=5)
# # disparity = stereo.compute(imgL,imgR)
# # depth = (disparity - disparity.min()) / (disparity.max() - disparity.min()) * 255.0
# # depth = depth.astype(np.uint8)
# # cmap = matplotlib.colormaps.get_cmap('magma') # Spectral_r
# # depth_for_save = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
# # cv.imwrite('/mnt/d/projectsD/datasets/depth_val_test/cv_depth.png', depth_for_save)



# img = cv.imread('/mnt/d/projectsD/datasets/rectified06/rectified06/depth01/0000000000.png')


import cv2 as cv

import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import open3d as o3d
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path

d16 = cv.imread("/mnt/d/projectsD/datasets/depth_val_test/dataset_4/keyframe_5/disparity.png", cv.IMREAD_UNCHANGED)
# print(d16/128)
img = cv.imread("/mnt/d/projectsD/datasets/depth_val_test/DepthAnything/Left_Image_DA_depth.png", cv.IMREAD_UNCHANGED)
# print(img)
# print(img.dtype, img.shape, img.min(), img.max(), d16.dtype, d16.shape, d16.min(), d16.max())
out_path = Path("/mnt/d/projectsD/datasets/depth_val_test/dataset_4/keyframe_5/Left_Image_GT_depthmap_colored_TEST.png")
scale_factor = 128.0  # match what you used when saving via scaredtk

# print(d16.dtype, d16.shape, d16.min(), d16.max())
depth = d16.astype(np.float32) / scale_factor         # back to metric units
depth[d16 == 0] = np.nan                              # restore unknowns (toolkit stores NaNs as 0) :contentReference[oaicite:4]{index=4}

valid = np.isfinite(depth)
vmin, vmax = np.percentile(depth[valid], [1, 99])
norm = np.zeros_like(depth, dtype=np.float32)
norm[valid] = np.clip((depth[valid] - vmin) / (vmax - vmin), 0, 1) * 255
depth = norm.astype(np.uint8)
rgba = plt.get_cmap("magma")(depth)
depth_for_save = (rgba[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
rgba[~valid, 3] = 0.0

depth_result = np.zeros_like(depth, dtype=np.float32)
depth_result[valid] = (depth[valid] - depth.min()) / (depth.max() - depth.min()) * 255.0
depth_result = depth_result.astype(np.uint8)  # Store the depth result
cmap = matplotlib.colormaps.get_cmap('magma') # Spectral_r
depth_for_save_ = (cmap(depth_result)[:, :, :3] * 255).astype(np.uint8)
# depth_for_save[~valid, 3] = 0
cv.imwrite(str(out_path), (rgba * 255).astype(np.uint8)[:,:,:3])  # OpenCV will write RGBA just fine
# cv.imwrite(str(out_path), depth_for_save_)
print(depth_for_save_)
r = (rgba * 255).astype(np.uint8)[:,:,:3]
print(r.shape)
print(img.min(), img.max(), depth_for_save_.min(), depth_for_save_.max(), r.min(), r.max())
# print("saved:", out_path)
# print("min/max:", np.nanmin(rgba), np.nanmax(rgba))
# print("percentiles:", np.nanpercentile(rgba, [0, 50, 100]))