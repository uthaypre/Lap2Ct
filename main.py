import open3d as o3d
import numpy as np
import os
import sys
import copy
from pathlib import Path
import yaml
from easydict import EasyDict as edict

from lap2ct.nnunet import predict_nnunet, create_nnunet_args
from lap2ct.utils import get_3dlap, get_3dct, get_overlay, get_depthmap, setup_logger, get_ball_marker, get_color_from_name
from lap2ct.registration import oareg, rigid_reg

def main():
    # Load configurations
    config_pth = 'configs/data.yaml'
    seg_config_pth = 'configs/organ_segmentation.yaml'
    with open(config_pth,'r') as f:
        config = yaml.safe_load(f)
    config = edict(config)
    with open(seg_config_pth, 'r') as f:
        seg_config = yaml.safe_load(f)
    seg_config = edict(seg_config)

    # Setup logger
    logger = setup_logger(log_level=config.log_level, log_file=config.save_log)

    # Generate transparency (alpha<1)
    mat_trans = o3d.visualization.rendering.MaterialRecord()
    mat_trans.shader = "defaultLitTransparency"
    mat_trans.base_color = [0.9, 0.9, 0.9, 0.5]

    try:
        logger.info("Processing started")
        logger.info("Starting preoperative processing")
        logger.info("Step 1: Generating 3D CT Model")
        
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

        ct_segmentation_file = next(Path(config.ct).glob("*.nii.gz"), None)
        logger.debug(f"Using CT segmentation file: {ct_segmentation_file}")
        ct_organs_mesh, ct_organs_bool = get_3dct(ct_segmentation_file, seg_config.ct_organ_classes, seg_config.ct_organ_colors)
        logger.info(f"Generated CT organ meshes for {len([m for m in ct_organs_mesh.values() if m is not None])} organs")

        ct_pcd_overlay = get_overlay(ct_organs_mesh)
        ct_organs_pcd = {organ: mesh.sample_points_uniformly(number_of_points=10000) for organ, mesh in ct_organs_mesh.items() if mesh is not None}
        logger.info("Starting intraoperative processing")
        # Loop through laparoscopy video frames
        laparoscopy_frames = list(Path(config.nnunet_laparoscopy).glob("*.png"))
        logger.info(f"Found {len(laparoscopy_frames)} laparoscopy frames to process")
        
        if not laparoscopy_frames:
            logger.warning("No laparoscopy frames found - check input directory")
            return 1
            
        camera_pose_tracker = []
        
        for i, frame in enumerate(laparoscopy_frames, 1):
            logger.info(f"Processing frame {i}/{len(laparoscopy_frames)}: {frame.name}")
            # Create args for Laparoscopy prediction
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

            logger.info(f"{frame.name}: Step 2: 3D Laparoscopy Reconstruction")
            logger.info(f"{frame.name}: Running nnUNet prediction for Laparoscopy data")
            predict_nnunet(laparoscopy_args)
            logger.info(f"{frame.name}: Laparoscopy prediction completed successfully")

            logger.info(f"{frame.name}: Running Depth Anything for Laparoscopy frame")
            lap_depth_map = get_depthmap(frame, outdir=config.laparoscopy, encoder='vitl', grayscale=False)
            logger.info(f"{frame.name}: Depth map generated successfully")
            mask_base = frame.stem.rsplit("_", 1)[0]  # remove last underscore chunk
            mask_path = os.path.join(config.laparoscopy, (mask_base + frame.suffix))
            logger.debug(f"{frame.name}: Mask path: {mask_path}")

            lap_organs_pcd, lap_organs_bool = get_3dlap(mask_path, seg_config.lap_organ_classes, seg_config.lap_organ_colors)
            logger.info(f"{frame.name}: Generated laparoscopy point clouds for {len([pc for pc in lap_organs_pcd.values() if pc is not None])} organs")
            local_camera_pose = np.array([0, 0, 0, 1])
            local_camera_poses = []

            logger.info(f"{frame.name}: Step 3: Starting rigid registration")
            # Iterate through found organs in laparoscopy
            overlay = [{"name": "ct_pcd",  "geometry": ct_pcd_overlay, "material": mat_trans}]
            registered_organs_count = 0
            overlayed_organs = []
            for organ_name, lap_organ in lap_organs_pcd.items():
                if lap_organ is None:
                    logger.info(f"{frame.name}: Skipping {organ_name} - no point cloud generated")
                    continue

                logger.info(f"{frame.name}: Processing organ: {organ_name}")
                # Find corresponding organ in CT
                if organ_name not in ct_organs_pcd:
                    logger.warning(f"{frame.name}: {organ_name} not found in CT organs, skipping registration")
                    continue
                logger.info(f"{frame.name}: Found corresponding CT organ for {organ_name}")
                # Perform rigid registration
                logger.info(f"{frame.name}: Rigid Registering {organ_name} to CT organ")
                transformed_lap_organ, init_transformation, reg_results = rigid_reg(source=lap_organ, target=ct_organs_pcd[organ_name])

                if len(reg_results.correspondence_set) < 5:
                    logger.warning(f"{frame.name}: Rigid Registration failed for {organ_name} - not enough correspondences -> skipping registration")
                    continue
                else:
                    logger.debug(f"{frame.name}: Rigid Registration successful with {len(reg_results.correspondence_set)} correspondences\n"
                                 f"{frame.name}: Registration fitness: {reg_results.fitness:.4f}\n"
                                 f"{frame.name}: Registration RMSE: {reg_results.inlier_rmse:.4f}\n"
                                 f"{frame.name}: Initial transformation matrix: {init_transformation}\n"
                                 f"{frame.name}: Final transformation matrix: {reg_results.transformation}\n"
                                 f"{frame.name}: Local camera pose: {local_camera_pose}")
                    
                    local_camera_poses.append(reg_results.transformation @ init_transformation @ local_camera_pose)
                    logger.info(f"{frame.name}: Rigid Registered {organ_name} successfully")
                    
                    logger.info(f"{frame.name}: Step 4: Starting non-rigid registration")
                    registered_lap_organ = oareg(transformed_lap_organ, ct_organs_pcd[organ_name])
                    current_color = get_color_from_name(organ_name, seg_config.lap_organ_classes, seg_config.lap_organ_colors)
                    logger.info(f"{frame.name}: Non-rigid registration completed for {organ_name}")
                    logger.debug(f"{frame.name}: Applying color to registered organ: {organ_name}")
                    registered_organs_count += 1
                    registered_lap_organ.paint_uniform_color(current_color)
                    overlay.append(registered_lap_organ)
                    overlay.append(copy.deepcopy(ct_organs_pcd[organ_name]).paint_uniform_color(np.array([1, 1, 1])))
                    overlayed_organs.append(organ_name)

            logger.info(f"{frame.name}: Registration completed for frame {frame.name}. Successfully registered {registered_organs_count} organs")

            # Camera pose estimation
            logger.debug(f"{frame.name}: Estimating camera pose from registered organs")
            if len(local_camera_poses) == 0:
                logger.warning(f"{frame.name}: No camera pose could be estimated")
                continue

            if len(local_camera_poses) > 1:
                camera_pose = np.mean(np.array(local_camera_poses), axis=0)[:3]
            elif len(local_camera_poses) == 1:
                camera_pose = local_camera_poses[0][:3]
                
            logger.info(f"{frame.name}: Camera pose estimated at {camera_pose} from {len(local_camera_poses)} organs (averaged)")
            camera_marker = get_ball_marker(camera_pose)
            camera_pose_tracker.append(camera_pose)
            o3d.visualization.draw(overlay + [camera_marker], show_skybox=False)
            organ_list_named = "_".join(overlayed_organs)
            # o3d.io.write_point_cloud(f"frame_{frame.name}_{organ_list_named}.ply", sum(overlay, camera_marker))
        logger.info(f"All frames processed successfully. Total camera poses tracked: {len(camera_pose_tracker)}")
        return 0
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())