import os
import numpy as np
import argparse
import tensorflow as tf
from model.vgg16 import Vgg16
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WeightExtractor")


def find_weight_files(root_dir):
    """ Recursively find all .weights.h5 files. """
    weight_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".weights.h5"):
                weight_files.append(os.path.join(dirpath, file))
    return weight_files

def save_weights_as_npz(weight_file, save_dir, root_dir):
    """ Extract weights and save as .npz using VGG16 model definition. """
    
    model_dir = os.path.dirname(weight_file)
    relative_path = os.path.relpath(model_dir, root_dir)
    save_path = os.path.join(save_dir, relative_path)
    os.makedirs(save_path, exist_ok=True)

    npz_file = os.path.join(save_path, os.path.basename(weight_file).replace(".weights.h5", ".npz"))

    logger.info(f"Processing: {weight_file}")

    model = Vgg16(input_shape=(112, 112, 3), n_classes=2,
                    first_trained_layer=11,
                    logger=logger, type={'model': 'normal'})

    model.compile()
    model.load_weights(weight_file)
    weights = model.model.get_weights()
    np.savez(npz_file, *weights)

    logger.info(f"Saved NPZ Weights to {npz_file}")

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
    all_weight_files = find_weight_files(args.root_dir)
    for weight_file in all_weight_files:
        save_weights_as_npz(weight_file, args.save_dir, args.root_dir)

    logger.info("Weight Conversion Completed")