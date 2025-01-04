"""
Common utils
"""

import warnings
warnings.filterwarnings("ignore")
import os
import datetime
import argparse
import logging
import sys

logger = None

def create_output_dir(args):
    output_dir = args.env + datetime.datetime.now().strftime('-%Y-%m-%d_%H-%M-%S')
    output_fullpath = os.path.join(os.path.expanduser(args.output_basedir), output_dir)
    os.makedirs(output_fullpath, exist_ok=True)
    return output_fullpath

def init_logger(args):
    global logger
    tmp_path = create_output_dir(args)
    log_file = os.path.join(tmp_path, "fight_log.txt")
    
    # Set up logging to both console and file
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler(sys.stdout)
                        ])
    logger = logging.getLogger()
    return logger

def com_print(text):
    if logger is None:
        print(text)  # Fallback to print if logger is not initialized
    else:
        logger.info(text)

def get_model_file_name(args):
    return f"{args.env}-{args.alg}-{args.nn}-{args.num_timesteps}"

# Additional utility functions can be added here as needed