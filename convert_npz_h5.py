import os
import numpy as np
import argparse
import tensorflow as tf
from model.vgg16 import Vgg16 
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WeightConverter")

def find_npz_files(root_dir):
    """ Recursively find all .npz files. """
    npz_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".npz"):
                npz_files.append(os.path.join(dirpath, file))
    return npz_files

def convert_npz_to_h5(npz_file, save_dir, root_dir):
    """ Load weights from .npz, apply to VGG16 model, and save as .weights.h5. """
    
    model_dir = os.path.dirname(npz_file)
    relative_path = os.path.relpath(model_dir, root_dir)
    save_path = os.path.join(save_dir, relative_path)
    os.makedirs(save_path, exist_ok=True)

    h5_file = os.path.join(save_path, os.path.basename(npz_file).replace(".npz", ".weights.h5"))

    logger.info(f"Processing: {npz_file}")

    model = Vgg16(input_shape=(112, 112, 3), n_classes=2,
                  first_trained_layer=11,
                  logger=logger, type={'model': 'normal'})
    model.compile()

    data = np.load(npz_file)
    weights = [data[f"arr_{i}"] for i in range(len(data.files))]
    model.model.set_weights(weights)

    model.model.save_weights(h5_file)

    logger.info(f"Saved converted weights to {h5_file}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/path/to/_weights",
        help="Directory containing the weight files"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="/path/to/_weightsnpz",
        help="Directory where converted npz files will be saved"
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU device index to use"
    )

    parser.add_argument(
        "--gpu_memory_limit",
        type=int,
        default=4 * 1024,
        help="GPU memory limit in MB"
    )

    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices("GPU")

    if gpus:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[args.gpu_id],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=args.gpu_memory_limit)]
        )

    os.makedirs(args.save_dir, exist_ok=True)
    all_npz_files = find_npz_files(args.root_dir)

    for npz_file in all_npz_files:
        convert_npz_to_h5(npz_file, args.save_dir, args.root_dir)

    logger.info("Weight Conversion Completed.")