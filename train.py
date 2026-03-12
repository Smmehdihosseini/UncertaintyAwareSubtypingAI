import os
import random
import json
import pandas as pd
import argparse
from utils.log import Logger
import tensorflow as tf
from model.vgg16 import Vgg16, SaveCallback, BestModelCallback, LastEpochCallback, MetricsLoggerCallback
from dataset.dataset import CropDatasetCached

try:
    with open("_info/config.json", 'r') as file:
        conf = json.load(file)
except:
    raise FileNotFoundError("Couldn't Load Config File. Check if the `config.json` file exists under `_info` directory.")

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Training Stages of Binary Classification of ExpertDT")

    parser.add_argument("--dfs_dir", type=str, default=conf['dfs_dir'], help='DataFrame Files Directory')
    parser.add_argument("--weights_dir", type=str, default=conf['weights_dir'], help='Weights Directory')
    parser.add_argument("--cache_dir", type=str, default=conf['cache_dir'], help='Temporary Image Cache Folder')
    parser.add_argument("--log_dir", type=str, default=conf['log_dir'], help="Logs Directory")
    parser.add_argument("--verbose", type=bool, default=True, help="Log Messages Verbose")
    parser.add_argument("--level", type=int, default=0, help='Whole Slide Image Magnification During Cropping')
    parser.add_argument("--model_crop_size", type=int, default=112, help='Model Resize Crop Size')
    parser.add_argument("--batch_size", type=int, default=128, help='Dataset Batch Size')
    parser.add_argument("--stage", type=str, default="Root", help='Training Stage of ExpertDT')
    parser.add_argument("--model_type", type=str, default='normal', help='Model Type: normal, mc, etc.')
    parser.add_argument("--weights_id", type=str, default='Fold1', help='Weights Identifier')
    parser.add_argument("--first_trained_layer", type=int, default=11, help='VGG16 First Trained Layer')
    parser.add_argument("--learning_rate", type=float, default=1e-5, help='Training Learning Rate')
    parser.add_argument("--epochs", type=int, default=150, help='Number of Training Epochs')
    parser.add_argument("--loss", type=str, default='categorical_crossentropy', help='Training Loss Function')
    parser.add_argument("--monitor_metric", type=str, default='loss', help='Training Monitor Metric')
    parser.add_argument("--early_patience", type=int, default=20, help='Early Stopping Callback Patience')
    parser.add_argument("--save_epochs", type=int, default=10, help='Number of Epochs for Saving Weights Callback')
    parser.add_argument("--cross_val", type=str, default="Fold1", help='Cross Validation Training; Set to None for Normal Mode or Fold Number for Cross Validation Mode')
    parser.add_argument("--gpu_memory", type=str, default=conf['gpu_memory'], help="GPU Memory Usage for Runtime For Each GPU in GB, comma-separated")
    parser.add_argument("--random_seed", type=int, default=42, help="Random Seed for Reproducibility")

    args = parser.parse_args()

    random.seed(args.random_seed)

    logger = Logger(log_dir=args.log_dir,
                    log_name=f'train_{args.weights_id}')

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
        logger.info(f">>> Loading Data for {args.stage} ...")

    if args.cross_val == "None":
        data_dir = os.path.join(args.dfs_dir, f"{args.stage}.csv")
    else:
        data_dir = os.path.join(
            args.dfs_dir,
            args.cross_val,
            f"{args.cross_val}_{args.stage}.csv"
        )

    stage_data = pd.read_csv(data_dir)

    if args.verbose:
        logger.info(f">>> Labels Distribution of '{args.stage}' Stage:")

    if args.verbose:
        logger.info("--"*20)
        for val, count in stage_data['type'].value_counts().items():
            logger.info(f"--------> {val} : {count}")
        logger.info(f"--------> Total : {stage_data.shape[0]}")
        logger.info("--"*20)

    if args.verbose:
        logger.info(">>> Preparing Training DataLoader ...")

    dataset = CropDatasetCached(dataframe=stage_data,
                          cache_dir=args.cache_dir,
                          level=args.level,
                          crop_size=(args.model_crop_size, args.model_crop_size),
                          batch_size=args.batch_size,
                          n_classes=2)
    
    dataset.process_and_cache_all_images()
    tf_dataset = dataset.get_dataset()

    if args.verbose:
        logger.info(">>> Preparing Model ...")

    vgg16 = Vgg16(input_shape=(args.model_crop_size, args.model_crop_size, 3),
                  type={'model':args.model_type},
                  n_classes=2,
                  logger=logger,
                  first_trained_layer=args.first_trained_layer)
    
    if args.verbose:
        logger.info(">>> Compiling Model ...")

    vgg16.compile(learning_rate=args.learning_rate,
                        loss=args.loss,
                        metrics=['accuracy'])

    if args.verbose:
        logger.info(">>> Loading Callbacks ...")

    early_callback = tf.keras.callbacks.EarlyStopping(monitor=args.monitor_metric,
                                                      patience=args.early_patience,
                                                      verbose=1 if args.verbose else 0)
    
    stage_weights_dir = os.path.join(args.weights_dir, args.weights_id, args.stage)
    save_callback = SaveCallback(
        logger=logger,
        save_path=stage_weights_dir,
        save_epoch=args.save_epochs,
        save_weights_only=True
    )

    best_callback = BestModelCallback(
        logger=logger,
        save_path=stage_weights_dir,
        monitor=args.monitor_metric,
        save_weights_only=True
    )

    metrics_logger_callback = MetricsLoggerCallback(logger=logger)

    last_epoch_callback = LastEpochCallback(
        logger=logger,
        save_path=stage_weights_dir,
        save_weights_only=True
    )

    if args.verbose:
        logger.info(">>> Starting Training Model ...")

    vgg16.fit(tf_dataset,
                epochs=args.epochs,
                callbacks=[early_callback,
                           save_callback,
                           best_callback,
                           metrics_logger_callback,
                           last_epoch_callback])
    
    if args.verbose:
        logger.critical(" ")
        logger.critical("++"*10+" Script Ended "+"++"*10)