import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
# # file = np.load("/mnt/d/projectsD/datasets/LAP2CT/laparoscopy/image_00300.npz")
# # print(file['probabilities'][0])  # List all arrays in the file
# # print(file['probabilities'].shape)  # Print the shape of the 'probabilities' array
# print(np.zeros((3, 1)))
# # import pickle
# # with open("/mnt/d/projectsD/datasets/LAP2CT/laparoscopy/image_00300.pkl", 'rb') as f:
# #     data = pickle.load(f)
# #     print(data)  # List all arrays in the file
# #     # print(data['probabilities'].shape)  # Print the shape of the 'probabilities' array
# # endo_pc = o3d.io.read_point_cloud("/mnt/d/projectsD/datasets/CTL-REG/input/01/real/liverPcds/frame_001_json.ply")
# # endo_pc.paint_uniform_color([0.5, 0.5, 0.5])  # Set color to gray
# # o3d.visualization.draw_geometries([endo_pc])

# items = ["apple", "banana", "cherry"]
# s = "_".join(items)          # "apple, banana, cherry"

# print(s)

print("Read Redwood dataset")
color_raw = o3d.io.read_image("/mnt/d/projectsD/datasets/livingroom1-color/00000.jpg")
depth_raw = o3d.io.read_image("/mnt/d/projectsD/datasets/livingroom1-depth-clean/00000.png")
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw)
print(rgbd_image)
# plt.subplot(1, 2, 1)
# plt.title('Redwood grayscale image')
# plt.imshow(rgbd_image.color)
# plt.subplot(1, 2, 2)
# plt.title('Redwood depth image')
# plt.imshow(rgbd_image.depth)
# plt.show()
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
# Flip it, otherwise the pointcloud will be upside down
# pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
# pcd.transform([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
# pcd.transform([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
# pcd.transform(np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])@np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))

flip = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
rotate = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
# pcd.transform(flip)
# pcd.transform(rotate)
# pcd.transform(np.dot(flip, rotate))
# pcd.transform(np.dot(rotate, flip))
pcd.transform(flip @ rotate)
# pcd.transform(rotate @ flip)
print(np.asarray(pcd.points[:10]))  # Print first 10 points to verify transformation
# o3d.visualization.draw_geometries([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame()])

M = [[1,2,3], [4,5,6], [7,8,9]]
N = []
for i in range(3):
    N.append(np.array([i,i+1,i+2]))
print(np.array(N))
print(np.mean(np.array(M), axis=0)[:3])
print(np.mean(np.array(N), axis=0)[:3])
