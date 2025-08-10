# import open3d as o3d
# import SimpleITK as sitk
# import numpy as np
# from skimage import measure
# import nibabel as nib
# import logging
# import os
import argparse
# import sys
from pathlib import Path
# import yaml
# from easydict import EasyDict as edict
from models.nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from models.nnUNet.nnunetv2.utilities.file_path_utilities import get_output_folder
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
# from torch import nn
# # from torch._dynamo import OptimizedModule
# from torch.nn.parallel import DistributedDataParallel
# from tqdm import tqdm
import torch

def predict_nnunet(args):


    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    # slightly passive aggressive haha
    assert args.part_id < args.num_parts, 'Do you even read the documentation? See nnUNetv2_predict -h.'

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if not hasattr(torch, "_nnunet_threads_configured"):
        torch._nnunet_threads_configured = False
    
    if torch._nnunet_threads_configured:
        device = torch.device(args.device)
    else:
        if args.device == 'cpu':
            # let's allow torch to use hella threads
            import multiprocessing
            torch.set_num_threads(multiprocessing.cpu_count())
            device = torch.device('cpu')
        elif args.device == 'cuda':
            # multithreading in torch doesn't help nnU-Net if run on GPU
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            device = torch.device('cuda')
        else:
            device = torch.device('mps')
    torch._nnunet_threads_configured = True
    predictor = nnUNetPredictor(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                verbose_preprocessing=args.verbose,
                                allow_tqdm=not args.disable_progress_bar)
    predictor.initialize_from_trained_model_folder(
        model_folder,
        args.f,
        checkpoint_name=args.chk
    )
    
    run_sequential = args.nps == 0 and args.npp == 0
    
    if run_sequential:
        
        print("Running in non-multiprocessing mode")
        predictor.predict_from_files_sequential(args.i, args.o, save_probabilities=args.save_probabilities,
                                                overwrite=not args.continue_prediction,
                                                folder_with_segs_from_prev_stage=args.prev_stage_predictions)
    
    else:
        
        predictor.predict_from_files(args.i, args.o, save_probabilities=args.save_probabilities,
                                    overwrite=not args.continue_prediction,
                                    num_processes_preprocessing=args.npp,
                                    num_processes_segmentation_export=args.nps,
                                    folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                    num_parts=args.num_parts,
                                    part_id=args.part_id)
    
def create_nnunet_args(input_path, output_path, dataset_id, config, fold='all', 
                      save_probabilities=True, model='nnUNetResEncUNetLPlans',
                      trainer='nnUNetTrainer_8000epochs', checkpoint=None):
    """
    Create an argparse.Namespace object with nnUNet prediction arguments.
    
    This mimics the command line arguments structure that nnUNet expects.
    """
    args = argparse.Namespace()

    # Required arguments
    args.i = input_path  # input folder
    args.o = output_path  # output folder
    args.d = dataset_id  # dataset ID
    args.c = config  # configuration (e.g., '3d_fullres', '2d')
    args.f = [fold] if fold != 'all' else ['all']  # folds
    
    # Model arguments
    args.p = model  # plans identifier
    args.tr = trainer  # trainer class name
    args.chk = checkpoint  # checkpoint name
    
    # Prediction arguments
    args.save_probabilities = save_probabilities
    args.step_size = 0.5
    args.disable_tta = False
    args.continue_prediction = False
    args.npp = 3  # number of processes for preprocessing
    args.nps = 3  # number of processes for segmentation export
    args.prev_stage_predictions = None
    args.num_parts = 1
    args.part_id = 0
    args.device = 'cuda'
    args.disable_progress_bar = False
    args.verbose = True
    
    return args