import open3d as o3d
import SimpleITK as sitk
import numpy as np
from skimage import measure
import nibabel as nib
import logging
import os
import argparse
import sys
from pathlib import Path
import yaml
from easydict import EasyDict as edict

# Import nnUNet inference function
from lap2ct.predict_nnunet import predict_nnunet, create_nnunet_args
from lap2ct.get_3dct import get_3dct
from lap2ct.get_3dlap import get_3dlap
from lap2ct.get_depthmap import get_depthmap

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





def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Lap2Ct - Register laparoscopic images to CT scans")
    # parser.add_argument('--config', type=str, help='Path to the config file.')
    # parser.add_argument('--input', '-i', type=str, help='Input file or directory path')
    # parser.add_argument('--output', '-o', type=str, default='./output', help='Output directory path')
    parser.add_argument('--save_log', default=False, action='store_true', help='Save log to file')
    parser.add_argument('--log_level', type=str, default='INFO', help='Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger(log_level=args.log_level, log_file=args.save_log)

    # Load configuration
    config_pth = 'configs/data.yaml'
    seg_config_pth = 'configs/organ_segmentation.yaml'
    with open(config_pth,'r') as f:
        config = yaml.safe_load(f)
    config = edict(config)

    with open(seg_config_pth, 'r') as f:
        seg_config = yaml.safe_load(f)
    seg_config = edict(seg_config)

    # Load input data
    # # Create output directory if it doesn't exist
    # output_path = Path(args.output)
    # output_path.mkdir(parents=True, exist_ok=True)
    Path(config.ct).mkdir(parents=True, exist_ok=True)
    Path(config.laparoscopy).mkdir(parents=True, exist_ok=True)

    try:
        # Add your main application logic here
        logger.info("Processing started")
        
        # Create args for CT prediction
        ct_args = create_nnunet_args(
            input_path=config.nnunet_ct,
            output_path=config.ct,
            dataset_id='023',
            config='3d_fullres',
            fold='all',
            save_probabilities=True,
            model='nnUNetResEncUNetLPlans',
            trainer='nnUNetTrainer_8000epochs',
            checkpoint=config.nnunet_ct_weights)
        
        logger.info("Running nnUNet prediction for CT data")
        # predict_nnunet(ct_args)
        logger.info("CT prediction completed successfully")

        ct_pcd, ct_organ_bool = get_3dct(next(Path(config.ct).glob("*.nii.gz"), None), seg_config.ct_organ_classes, seg_config.ct_organ_colors)
        # achtung: ct_organ_bool ist versetzt, da schwarz=0 nicht als organ mitgerechnet wurde!!!

        # Loop through laparoscopy video frames
        laparoscopy_frames = list(Path(config.nnunet_laparoscopy).glob("*.png"))
        for frame in laparoscopy_frames:
            logger.info(f"Processing frame: {frame.name}")
            # Create args for Laparoscopy prediction
            # laparoscopy_args = create_nnunet_args(
            #     input_path=[str(frame)],
            #     output_path=config.laparoscopy,
            #     dataset_id=333,
            #     config='2d',
            #     fold='all',
            #     save_probabilities=True,
            #     model='nnUNetResEncUNetLPlans',
            #     trainer='nnUNetTrainer_8000epochs',
            #     checkpoint=config.nnunet_lap_weights)        
        
            logger.info("Running nnUNet prediction for Laparoscopy data")
            # predict_nnunet(laparoscopy_args)
            logger.info("Laparoscopy prediction completed successfully")
            logger.info("Running Depth Anything for Laparoscopy frame")
            # get depth map from laparoscopy frame
            lap_depth_map = get_depthmap(frame, outdir=config.laparoscopy, encoder='vitl', grayscale=False)
            logger.info("Depth map generated successfully")
            mask_base = frame.stem.rsplit("_", 1)[0]  # remove last underscore chunk
            mask_path = os.path.join(config.laparoscopy, (mask_base + frame.suffix))
            print(f"Mask path: {mask_path}")
            lap_pcd = get_3dlap(mask_path, seg_config.lap_organ_classes, seg_config.lap_organ_colors)
        logger.info("Processing completed successfully")
        return 0
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())