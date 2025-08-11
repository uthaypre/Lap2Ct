import open3d as o3d
import SimpleITK as sitk
import numpy as np
from skimage import measure
import nibabel as nib
import pickle
import logging
import os
import argparse
import sys
import copy
from pathlib import Path
import yaml
from easydict import EasyDict as edict

# Import nnUNet inference function
from lap2ct.predict_nnunet import predict_nnunet, create_nnunet_args
from lap2ct.get_3dct import get_3dct
from lap2ct.get_3dlap import get_3dlap
from lap2ct.get_depthmap import get_depthmap
from lap2ct.rigid_reg import rigid_reg
from lap2ct.nonrigid_reg import oareg
from lap2ct.get_overlay import get_overlay
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
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
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

        file_handler = logging.FileHandler(unique_log_file)
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



def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Lap2Ct - Register laparoscopic images to CT scans")
    parser.add_argument('--save_log', default=True, action='store_true', help='Save log to file')
    parser.add_argument('--log_level', type=str, default='INFO', help='Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger(log_level=args.log_level, log_file=args.save_log)

    # Load configurations
    config_pth = 'configs/data.yaml'
    seg_config_pth = 'configs/organ_segmentation.yaml'
    with open(config_pth,'r') as f:
        config = yaml.safe_load(f)
    config = edict(config)
    with open(seg_config_pth, 'r') as f:
        seg_config = yaml.safe_load(f)
    seg_config = edict(seg_config)
    logger.info("Configurations loaded successfully")

    # Load input data
    Path(config.ct).mkdir(parents=True, exist_ok=True)
    Path(config.laparoscopy).mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Processing started")
        #### PREOPERATIVE ####
        ### 1. 3D CT Model ###
        logger.info("Starting preoperative processing")
        logger.info("Step 1: Processing 3D CT Model")
        
        # CT segmentation using nnUNet (currently commented out)
        # logger.info("Creating nnUNet arguments for CT segmentation")
        # ct_args = create_nnunet_args(
        #     input_path=config.nnunet_ct,
        #     output_path=config.ct,
        #     dataset_id='023',
        #     config='3d_fullres',
        #     fold='all',
        #     save_probabilities=False,
        #     model='nnUNetResEncUNetLPlans',
        #     trainer='nnUNetTrainer_8000epochs',
        #     checkpoint=config.nnunet_ct_weights)
        
        ## Running nnUNet prediction for CT data
        # logger.info("Running nnUNet prediction for CT data")
        # predict_nnunet(ct_args)
        # logger.info("CT prediction completed successfully")

        logger.info("Generating 3D CT organs from segmentation masks")
        ct_segmentation_file = next(Path(config.ct).glob("*.nii.gz"), None)
        logger.debug(f"Using CT segmentation file: {ct_segmentation_file}")
        ct_organs_mesh, ct_organs_bool = get_3dct(ct_segmentation_file, seg_config.ct_organ_classes, seg_config.ct_organ_colors)
        logger.info(f"Generated CT organ meshes for {len([m for m in ct_organs_mesh.values() if m is not None])} organs")
        
        logger.info("Creating CT overlay visualization")
        ct_pcd_overlay = get_overlay(ct_organs_mesh)
        # achtung: ct_organ_bool ist versetzt, da schwarz=0 nicht als organ mitgerechnet wurde!!!
        logger.info("Converting CT meshes to point clouds")
        ct_organs_pcd = {organ: mesh.sample_points_uniformly(number_of_points=10000) for organ, mesh in ct_organs_mesh.items() if mesh is not None}
        #### INTRAOPERATIVE ####
        logger.info("Starting intraoperative processing")
        # Loop through laparoscopy video frames
        laparoscopy_frames = list(Path(config.nnunet_laparoscopy).glob("*.png"))
        logger.info(f"Found {len(laparoscopy_frames)} laparoscopy frames to process")
        
        if not laparoscopy_frames:
            logger.warning("No laparoscopy frames found - check input directory")
            return 1
            
        camera_pose_tracker = []
        logger.debug("Initialized camera pose tracker for frame-by-frame processing")
        
        for i, frame in enumerate(laparoscopy_frames, 1):
            logger.info(f"Processing frame {i}/{len(laparoscopy_frames)}: {frame.name}")
            # Create args for Laparoscopy prediction
            logger.debug(f"{frame.name}: Creating nnUNet arguments for laparoscopy prediction")
            laparoscopy_args = create_nnunet_args(
                input_path=[[str(frame)]],
                output_path=config.laparoscopy,
                dataset_id=333,
                config='2d',
                fold='all',
                save_probabilities=True,
                model='nnUNetResEncUNetLPlans',
                trainer='nnUNetTrainer_8000epochs',
                checkpoint=config.nnunet_lap_weights)   
            #      
            ### 2. 3D Laparoscopy Reconstruction ###
            logger.info(f"{frame.name}: Step 2: 3D Laparoscopy Reconstruction")
            logger.info(f"{frame.name}: Running nnUNet prediction for Laparoscopy data")
            predict_nnunet(laparoscopy_args)
            logger.info(f"{frame.name}: Laparoscopy prediction completed successfully")

            logger.info(f"{frame.name}: Running Depth Anything for Laparoscopy frame")
            # get depth map from laparoscopy frame
            lap_depth_map = get_depthmap(frame, outdir=config.laparoscopy, encoder='vitl', grayscale=False)
            logger.info(f"{frame.name}: Depth map generated successfully")
            mask_base = frame.stem.rsplit("_", 1)[0]  # remove last underscore chunk
            mask_path = os.path.join(config.laparoscopy, (mask_base + frame.suffix))
            logger.debug(f"{frame.name}: Mask path: {mask_path}")

            logger.info(f"{frame.name}: Generating 3D point clouds from laparoscopy segmentation")
            lap_organs_pcd, lap_organs_bool = get_3dlap(mask_path, seg_config.lap_organ_classes, seg_config.lap_organ_colors)
            logger.info(f"{frame.name}: Generated laparoscopy point clouds for {len([pc for pc in lap_organs_pcd.values() if pc is not None])} organs")
            local_camera_pose = np.array([0, 0, 0, 1])
            local_camera_poses = []
            ### 3. Rigid Registration ###
            logger.info(f"{frame.name}: Step 3: Starting rigid registration")
            # Iterate through found organs in laparoscopy
            overlay = [ct_pcd_overlay]
            registered_organs_count = 0
            
            for organ_name, lap_organ in lap_organs_pcd.items():
                if lap_organ is None:
                    logger.debug(f"{frame.name}: Skipping {organ_name} - no point cloud generated")
                    continue

                logger.info(f"{frame.name}: Processing organ: {organ_name}")
                # Find corresponding organ in CT
                if organ_name in ct_organs_pcd:
                    logger.info(f"{frame.name}: Found corresponding CT organ for {organ_name}")
                    # Perform rigid registration
                    logger.info(f"{frame.name}: Rigid Registering {organ_name} to CT organ")
                    transformed_lap_organ, init_transformation, reg_results = rigid_reg(source=lap_organ, target=ct_organs_pcd[organ_name])

                    if len(reg_results.correspondence_set) < 5:
                        logger.warning(f"{frame.name}: Not enough correspondences for {organ_name} ({len(reg_results.correspondence_set)} found), skipping rigid registration")
                        continue

                    logger.debug(f"{frame.name}: Rigid Registration successful with {len(reg_results.correspondence_set)} correspondences")
                    logger.debug(f"{frame.name}: Registration fitness: {reg_results.fitness:.4f}")
                    logger.debug(f"{frame.name}: Registration RMSE: {reg_results.inlier_rmse:.4f}")
                    
                    print("this is the transformation matrix:", reg_results.transformation)
                    print("this is the init transformation matrix:", init_transformation)
                    print("this is the local camera pose:", local_camera_pose)
                    local_camera_poses.append(reg_results.transformation @ init_transformation @ local_camera_pose)

                    logger.info(f"{frame.name}: Rigid Registered {organ_name} successfully")
                    
                    ### 4. Non-Rigid Registration ###
                    logger.info(f"{frame.name}: Step 4: Starting non-rigid registration")
                    registered_lap_organ = oareg(transformed_lap_organ, ct_organs_pcd[organ_name])
                    current_color = get_color_from_name(organ_name, seg_config.lap_organ_classes, seg_config.lap_organ_colors)
                    logger.info(f"{frame.name}: Non-rigid registration completed for {organ_name}")
                    logger.debug(f"{frame.name}: Applying color to registered organ: {organ_name}")
                    registered_organs_count += 1
                    registered_lap_organ.paint_uniform_color(current_color)
                    overlay.append(registered_lap_organ)
                    overlay.append(copy.deepcopy(ct_organs_pcd[organ_name]).paint_uniform_color(np.array([0.75, 0.75, 0.75])))
                else:
                    logger.warning(f"{frame.name}: {organ_name} not found in CT organs, skipping registration")

            logger.info(f"{frame.name}: Registration completed for frame {frame.name}. Successfully registered {registered_organs_count} organs")

            # Camera pose estimation
            logger.debug(f"{frame.name}: Estimating camera pose from registered organs")
            if len(local_camera_poses) > 1:
                camera_pose = np.mean(np.array(local_camera_poses), axis=0)
                logger.info(f"{frame.name}: Camera pose estimated from {len(local_camera_poses)} organs (averaged)")
                camera_marker = get_ball_marker(camera_pose[:3])
                camera_pose_tracker.append(camera_pose[:3])
            elif len(local_camera_poses) == 1:
                logger.info(f"{frame.name}: Camera pose estimated from single organ")
                camera_marker = get_ball_marker(local_camera_poses[0][:3])
                camera_pose_tracker.append(local_camera_poses[0][:3])
            else:
                logger.warning(f"{frame.name}: No camera pose could be estimated - no successful registrations")
                continue
            
            logger.info(f"Camera pose for frame {frame.name}: {camera_pose_tracker[-1]}")

            # Visualize the registered organs
            logger.info(f"Displaying visualization for frame {frame.name}")
            o3d.visualization.draw_geometries(overlay + [camera_marker], window_name=f"Laparoscopy Frame {frame.name}")
                
        logger.info(f"All frames processed successfully. Total camera poses tracked: {len(camera_pose_tracker)}")
        logger.info("Processing completed successfully")
        return 0
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())