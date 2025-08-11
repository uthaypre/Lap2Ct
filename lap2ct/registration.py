import open3d as o3d
import numpy as np
import copy
import open3d as o3d
import numpy as np
import torch
from models.OAReg.utils2.normalize_pointcloud import normalize_ply
from models.OAReg.utils2.LLR import local_linear_reconstruction
from models.OAReg.utils2.loss_functions import correntropy_chamfer_distance
from models.OAReg.model.model import Siren

torch.cuda.set_device(0)
DEVICE = 'cuda'

# deformation learning
def deform_point_cloud(model, xsrc=None, xtrg=None,
                 n_samples=10000, n_steps=200, sigma2=1.0, init_lr=1.0e-4,
                 LLR_weight=1.0e2, MCC_chamfer_weight=1.0e4,
                 LLR_n_neighbors=30, eval_every_nth_step=100, point_num=None):
  """
  Deform a point cloud using a neural network model.

  Parameters
  ----------
  model : torch.nn.Module
      The neural network model to use for deformation.
  xsrc : numpy.ndarray
      The source point cloud to deform.
  xtrg : numpy.ndarray, optional
      The target point cloud to match (used in MCC distance loss).
  n_samples : int, optional
      The number of points to sample for MCC distance loss (default is 10**4).
  n_steps : int, optional
      The number of optimization steps (default is 200).
  init_lr : float, optional
      The initial learning rate for the optimizer (default is 1.0e-4).
  LRR_weight : float, optional
      The weight for LRR loss (default is 1.0e2).
  MCC_chamfer_weight : float, optional
      The weight for chamfer distance loss (default is 1.0e4).
  LLR_n_neighbors: int, optional
      The number of neighbors to use for LRR loss (default is 30).
  eval_every_nth_step : int, optional
      The number of steps between evaluations (default is 100).
  point_num: int, optional
      The minimal number of the two input point clouds 
  """

  model = model.train()
  optm = torch.optim.Adam(model.parameters(), lr=init_lr)# optimizer
  schedm = torch.optim.lr_scheduler.ReduceLROnPlateau(optm,patience=1)# lr


  MCC_chamfer_loss_total = 0
  LLR_loss_total = 0
  total_loss = 0
  n_r = 0

  # Downsampling
  n_samples=5000
  if n_samples>point_num:
      n_samples=point_num
      

  for i in range(0, n_steps):
    xbatch_src=xsrc[np.random.choice(len(xsrc), n_samples, replace=False)]
    xbatch_trg=xtrg[np.random.choice(len(xtrg), n_samples, replace=False)]
    xbatch_deformed = xbatch_src + model(xbatch_src)

    loss = 0

    # LLR loss
    LLR_loss = LLR_weight*local_linear_reconstruction(xbatch_src, xbatch_deformed, n_neighbors=LLR_n_neighbors)
    loss += LLR_loss
    LLR_loss_total += float(LLR_loss)


    # MCC
    MCC_loss=correntropy_chamfer_distance(xbatch_deformed.unsqueeze(0),xbatch_trg.unsqueeze(0),sigma2=sigma2)
    MCC_chamfer_loss = MCC_chamfer_weight*MCC_loss
    loss += MCC_chamfer_loss
    MCC_chamfer_loss_total += float(MCC_chamfer_loss)
       

    total_loss += float(loss)
    n_r += 1

    optm.zero_grad()
    loss.backward()
    optm.step()

    # Evaluate the training results
    if i % eval_every_nth_step == 0:

      LLR_loss_total /= n_r
      MCC_chamfer_loss_total /= n_r
      total_loss /= n_r

      schedm.step(float(total_loss))
      


      LLR_loss_total = 0
      MCC_chamfer_loss_total = 0
      total_loss = 0
      n_r = 0

  LLR_loss_total /= n_r
  MCC_chamfer_loss_total /= n_r
  total_loss /= n_r




def MCC_registration(xsrc=None, xtrg=None, 
                     target_normal_scale=None,target_normal_center=None,
                     n_steps=200,
                     sigma2=1.0,
                     LLR_n_neighbors=30,
                     LLR_WEIGHT=1.0e2,
                     MCC_chamfer_WEIGHT=1.0e4,
                     point_num=None):
    

#  define the deformation model
    model = Siren(in_features=3,
                    hidden_features=128,
                    hidden_layers=3,
                    out_features=3, outermost_linear=True,
                    first_omega_0=30, hidden_omega_0=30.).to(DEVICE).train()
    
    deform_point_cloud(model,
            xsrc=xsrc, xtrg=xtrg,
            init_lr=1.0e-4,
            n_steps=n_steps,
            sigma2=sigma2,
            LLR_n_neighbors=LLR_n_neighbors,
            LLR_weight=LLR_WEIGHT,
            MCC_chamfer_weight=MCC_chamfer_WEIGHT,
            point_num=point_num)
    
    
    model.eval()
    vpred = xsrc + model(xsrc).detach().clone()

    vpred_save=vpred.cpu().numpy()

    vpred_save_denormalize=target_normal_scale*vpred_save+target_normal_center


    pcd_deformed=o3d.geometry.PointCloud()
    pcd_deformed.points=o3d.utility.Vector3dVector(vpred_save_denormalize)
    return pcd_deformed



def oareg(source, target):        
    # Normalize the input point clouds
    src_normalized_ply, src_normal_center, src_normal_scale = normalize_ply(source)
    tgt_normalized_ply, tgt_normal_center, tgt_normal_scale = normalize_ply(target)

    # Get points 
    src_points = np.asarray(src_normalized_ply.points,dtype=np.float32)
    tgt_points = np.asarray(tgt_normalized_ply.points,dtype=np.float32)

    src_points = torch.from_numpy(src_points).to(DEVICE)
    tgt_points = torch.from_numpy(tgt_points).to(DEVICE)

    src_point_num=src_points.shape[0]
    tgt_point_num=tgt_points.shape[0]


    if src_point_num>tgt_point_num:
        point_num=tgt_point_num
    else:
        point_num=src_point_num

    # Iterative optimization for registration
    registered_pcd = MCC_registration(xsrc=src_points, xtrg=tgt_points,
                        target_normal_scale=tgt_normal_scale,target_normal_center=tgt_normal_center,
                        n_steps=200,
                        sigma2=1.0,
                        LLR_n_neighbors=30,
                        LLR_WEIGHT=1.0e2,
                        MCC_chamfer_WEIGHT=1.0e4,
                        point_num=point_num)

    print("**************************")
    return registered_pcd

def draw_registration_result(source, target):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # o3d.visualization.draw_geometries([source_temp, target_temp],
                                    #   zoom=0.4559,
                                    #   front=[0.6452, -0.3036, -0.7011],
                                    #   lookat=[1.9892, 2.0208, 1.8945],
                                    #   up=[-0.2779, -0.9482, 0.1556])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(source=None, target=None, voxel_size=0.05):
    draw_registration_result(source, target)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    o3d.utility.random.seed(0)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),  # Enable scaling
        4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(5000000, 0.999))
    return result

def init_transformation(source, target):
    """
    Initialize the transformation matrix for registration.
    
    Parameters:
    - source: Source point cloud (Open3D PointCloud).
    - target: Target point cloud (Open3D PointCloud).
    
    Returns:
    - transformation: Initial transformation matrix.
    """
    source_center = np.mean(np.asarray(source.points), axis=0)
    target_center = np.mean(np.asarray(target.points), axis=0)
    
    translation = target_center - source_center
    transformation = np.eye(4)
    transformation[:3, 3] = translation

    source_min = np.min(np.asarray(source.points), axis=0)
    target_min = np.min(np.asarray(target.points), axis=0)
    source_max = np.max(np.asarray(source.points), axis=0)
    target_max = np.max(np.asarray(target.points), axis=0)
    source_scale = np.linalg.norm(source_max - source_min)
    target_scale = np.linalg.norm(target_max - target_min) 
    scale = target_scale / source_scale
    transformation[:3, :3] *= scale
    source.transform(transformation)
    
    return transformation


def rigid_reg(source, target, voxel_size=1.817):
    """
    Perform rigid registration of two point clouds.
    
    Parameters:
    - source: Source point cloud (Open3D PointCloud).
    - target: Target point cloud (Open3D PointCloud).
    - voxel_size: Voxel size for downsampling and feature computation.
    
    Returns:
    - result: Registration result containing the transformation matrix.
    """
    # draw_registration_result(source, target, np.identity(4))
    init_transform = init_transformation(source, target)
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source, target,
        voxel_size)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print(result_ransac)
    source_transformed = source_down.transform(result_ransac.transformation)
    # draw_registration_result(source_down, target_down)
    return source_transformed, init_transform, result_ransac