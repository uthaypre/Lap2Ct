from main import main
import sys
import yaml
from easydict import EasyDict as edict
from lap2ct.utils import setup_logger

if __name__ == "__main__":
    # Setup logger once for the entire loop
    config_pth = 'configs/data.yaml'
    with open(config_pth,'r') as f:
        config = yaml.safe_load(f)
    config = edict(config)
    
    # Create a persistent logger
    logger = setup_logger(log_level=config.log_level, log_file=config.save_log)
    logger.info("Starting voxel size loop experiment with fast global registration")
    
    # voxel_sizes = [0.001, 0.005, 0.01,0.1,0.5,0.75,1,1.25,1.5,1.817, 2, 2.5, 3]
    voxel_sizes = [1.25, 1.5, 1.817, 2, 2.5]
    for i, voxel in enumerate(reversed(voxel_sizes), 1):
        logger.info(f"=== Experiment {i}/{len(voxel_sizes)}: Testing voxel_size={voxel} ===")
        try:
            main(voxel_size=voxel, logger=logger)
            logger.info(f"=== Experiment {i} completed successfully ===")
        except Exception as e:
            logger.error(f"=== Experiment {i} failed: {str(e)} ===", exc_info=True)
    
    logger.info("All voxel size experiments completed")