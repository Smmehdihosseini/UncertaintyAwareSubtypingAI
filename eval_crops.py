import os
import json
import random
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import gc
from utils.log import Logger
import tensorflow as tf
from model.vgg16 import Vgg16
from dataset.dataset import CropEvaluationDataset
from model.weights_loader import find_weights

try:
    with open("_info/config.json", 'r') as file:
        conf = json.load(file)
except:
    raise FileNotFoundError("Couldn't Load Config File. Check if the `config.json` file exists under `_info` directory.")

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Crop-level Evaluation of Training/Validation Data")

    parser.add_argument("--info_dir", type=str, default=conf['info_dir'], help="Information Files Directory")
    parser.add_argument("--weights_dir", type=str, default=conf['weights_dir'], help='Weights Directory')
    parser.add_argument("--results_dir", type=str, default=conf['results_dir'], help="Results Directory")
    parser.add_argument("--dfs_dir", type=str, default=conf['dfs_dir'], help="DataFrames Directory")
    parser.add_argument("--cache_dir", type=str, default=conf['cache_dir'], help='Temporary Image Cache')
    parser.add_argument("--log_dir", type=str, default=conf['log_dir'], help="Path to Logs Directory")
    parser.add_argument("--verbose", type=bool, default=True, help="Verbose Print Details")
    parser.add_argument("--first_trained_layer", type=int, default=11, help='VGG16 First Trained Layer')
    parser.add_argument("--model_crop_size", type=int, default=112, help="Model Resize Crop Size")
    parser.add_argument("--wsi_level", type=int, default=0, help="Whole Slide Image Level of Magnification During Cropping Stage")
    parser.add_argument("--overlap", type=int, default=1, help="Patches Overlap During Cropping Stage; 1: No Overlap, 2: 50%, etc.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch Size For Inference Multiprocessing")
    parser.add_argument("--eval_type", type=str, default='Train', help="Evaluate Training/Validation Set")
    parser.add_argument("--weights_id", type=str, default='Fold1', help='Weights Identifier')
    parser.add_argument("--cross_val", type=str, default="Fold1", help='Cross Validation Evaluation')
    parser.add_argument("--eval_stages", type=str, default="Root, Node, Leaf1, Leaf2", help='Stages to be evaluated, it must be comma-separated')
    parser.add_argument("--runtime_id", type=str, default="train_Fold1", help="Runtime id")
    parser.add_argument("--gpu_memory", type=str, default=conf['gpu_memory'], help="GPU Memory Usage for Runtime For Each GPU in GB, comma-separated")
    parser.add_argument("--random_seed", type=int, default=42, help="Random Seed for Reproducibility")

    args = parser.parse_args()

    random.seed(args.random_seed)
    logger = Logger(log_dir=args.log_dir,
                    log_name=f'eval_crops_{args.runtime_id}')

    gpus = tf.config.list_physical_devices('GPU')
    gpu_allocate = args.gpu_memory.split(',')
    gpu_allocate = [float(fraction) for fraction in gpu_allocate]

    if len(gpus) != len(gpu_allocate):
        raise ValueError("Allocations doesn't match the GPUs count")

    logger.critical(f">>> Cuda Visible GPUs: {gpus} ...")

    for i, gpu in enumerate(gpus):
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [
                tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=int(gpu_allocate[i] * 1024)
                )
            ]
        )
    
    if args.verbose:
        logger.critical(">>> Runtime Arguments:")
        logger.critical("--"*20)

        for argument, val in vars(args).items():
            logger.critical(f"--------> {argument} = {val} ({type(val)})")
        logger.critical("--"*20)
        logger.critical(" ")
        logger.critical("++"*10+" Script Started "+"++"*10)

    eval_stages = [stage.strip() for stage in args.eval_stages.split(',')] 

    save_dir = os.path.join(args.results_dir, "crop_eval", args.runtime_id)
    os.makedirs(save_dir, exist_ok=True)

    logger.info(">>> Loading Models Parameters ...")
    with open(os.path.join(args.info_dir, "model_params.json"), 'r') as file:
        model_params = json.load(file)[args.runtime_id]
        
    weights = find_weights(weights_dir=args.weights_dir,
                            weights_id=args.weights_id,
                            model_params=model_params)

    for stage in eval_stages:

        if args.verbose:
            logger.info(f">>> Evaluating the Crops on '{stage}' Model ...")

        if args.eval_type=='Train':
            eval_df_prefix = ''
            if args.cross_val!='None':
                eval_df_prefix = f"{args.cross_val}"   
            eval_df = pd.read_csv(os.path.join(args.dfs_dir, args.weights_id, f"{eval_df_prefix}_{stage}.csv"))

        elif args.eval_type=='Validation':
            eval_df_name = 'val_df'
            if args.cross_val!='None':
                eval_df_name = f"{args.cross_val}_{eval_df_name}"

            eval_df = pd.read_csv(os.path.join(args.dfs_dir, args.weights_id, f"{eval_df_name}.csv"))

        crop_eval_dataset = CropEvaluationDataset(dataframe=eval_df,
                                            cache_dir=args.cache_dir,
                                            level=args.wsi_level,
                                            crop_size=(args.model_crop_size, args.model_crop_size),
                                            batch_size=args.batch_size
                                            )

        total_crops = eval_df.shape[0]
        total_batches = (total_crops + args.batch_size - 1) // args.batch_size

        if args.verbose:
            logger.info(f">>> Creating '{stage} 'Dataset (Crops: {total_crops}, Batches: {total_batches}) ...")
        
        crop_eval_dataset.process_and_cache_all_images()
        tf_dataset = crop_eval_dataset.get_dataset()

        eval_df['preds'] = None

        vgg16 = Vgg16(input_shape=(args.model_crop_size, args.model_crop_size, 3),
                        logger=logger,
                        type=model_params[stage],
                        n_classes=2,
                        first_trained_layer=args.first_trained_layer)
        
        if args.verbose:
            logger.info(f">>> Loading '{weights[stage]}', Default '{model_params[stage]['default_weight']}' ...")

        weights_path = os.path.join(
                        args.weights_dir,
                        args.weights_id,
                        stage,
                        weights[stage]
                    )
        vgg16.load_weights(weights_path=weights_path)

        for batch_indices, batch_images in tqdm(tf_dataset, desc=">>> Processing Batches", 
                                                total=total_batches,
                                                unit="Batch"):
            
            predictions, _, _ = vgg16.predict(np.array(batch_images))
            eval_updates = []

            for idx, pred in zip(batch_indices.numpy(), predictions):
                eval_updates.append((idx, pred.tolist()))

            del predictions, batch_images
            tf.keras.backend.clear_session()

            updates_df = pd.DataFrame(eval_updates, columns=['index', 'preds'])
            updates_df.set_index('index', inplace=True)
            eval_df.update(updates_df)
            del eval_updates, updates_df
            gc.collect()

        eval_df.to_csv(
                os.path.join(save_dir, f"{stage}.csv"),
                index=False
            )

    if args.verbose:
        logger.critical(" ")
        logger.critical("++"*10+" Script Ended "+"++"*10)