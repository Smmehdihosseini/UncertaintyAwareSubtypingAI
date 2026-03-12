import os
import random
import pandas as pd
import json
import argparse
import glob
import time
import numpy as np
import yaml
from dataset.data_split import patient_kfold_split
from wsi_manager.crop import CropIndexer
from dataset.balancer import Balancer
from utils.log import Logger

try:
    with open("_info/config.json", 'r') as file:
        conf = json.load(file)
except:
    raise FileNotFoundError("Couldn't Load Config File. Check if the `config.json` file exists under `_info` directory.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Prepare Data for Binary Classification of ExpertDT in Cross-validation Mode")

    parser.add_argument("--wsis_dir", type=str, default=conf['wsis_dir'], help="WSIs Folder")
    parser.add_argument("--info_dir", type=str, default=conf['info_dir'], help="Information Files Directory")
    parser.add_argument("--dfs_dir", type=str, default=conf['dfs_dir'], help="DataFrame Files Directory")
    parser.add_argument("--log_dir", type=str, default=conf['log_dir'], help="Log Files Directory")
    parser.add_argument("--verbose", type=bool, default=True, help="Log Messages Verbose")
    parser.add_argument("--load_id_list", type=bool, default=False, help="Load ID Split List, False Results to Create and Save New List File")
    parser.add_argument("--load_data_df", type=bool, default=False, help="Load Training DataFrame, False Results to Create and Save New DataFrame")
    parser.add_argument("--crop_size", type=int, default=1000, help="Crops Size")
    parser.add_argument("--wsi_level", type=int, default=0, help="Whole Slide Image Magnification Level During Cropping")
    parser.add_argument("--overlap", type=int, default=1, help="Crops Overlap During Cropping; 1: No Overlap, 2: 50%, etc.")
    parser.add_argument("--wsi_formats", nargs=3, type=str, default=['scn', 'svs', 'tif'], help="Possible Formats for Whole Slide Image")
    parser.add_argument("--balance_method", type=str, default="undersample", help="Balance Method for Each Stage of Tree")
    parser.add_argument("--multiprocessing", type=bool, default=True, help="Multiprocessing for CropIndexer")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of Threads for Multiprocessing")
    parser.add_argument("--cross_val_fold", type=int, default=4, help="Number of Cross Validation Folds")
    parser.add_argument("--random_seed", type=int, default=42, help="Random Seed for Reproducibility")

    args = parser.parse_args()

    random.seed(args.random_seed)

    logger = Logger(log_dir=args.log_dir,
                    log_name='prepare_data_cv')
    
    if args.verbose:
        logger.critical(">>> Runtime Arguments:")
        logger.critical("--"*20)

        for argument, val in vars(args).items():
            logger.critical(f"--------> {argument} = {val} ({type(val)})")
        logger.critical("--"*20)
        logger.critical(" ")
        logger.critical("++"*10+" Script Started "+"++"*10)

    if args.verbose:
        logger.info(">>> Loading Parameters YAML ...")
    
    with open(os.path.join(args.info_dir, "prepare_data.yaml"), "r") as file:
        params = yaml.safe_load(file)

    try:
        for param in params.keys():
            if params.get(param, []) is None:
                params[param] = []
            logger.info(f"{param.upper()}: {params.get(param, [])}")
    except Exception as e:
        logger.error(
            f"Problem with Prepare Data YAML '{os.path.join(args.info_dir, 'prepare_data.yaml')}' folder."
        )
        raise e

    if args.verbose:
        logger.info(">>> Loading Patient IDs ...")

    try:
        ids_df = pd.read_csv(os.path.join(args.info_dir, "ids.csv"))
    except FileNotFoundError as e:
        logger.error(f"No IDs File in '{args.info_dir}' folder.")
        raise e

    ids_df = ids_df.dropna()
    ids_df.roi_exist = ids_df.roi_exist.astype(bool)
    valid_centers_batches = []
    for center, batches in params['centers'].items():
        for batch in batches:
            valid_centers_batches.append((center, batch))

    valid_df = pd.DataFrame(valid_centers_batches, columns=['center', 'batch'])
    ids_cohort = ids_df.merge(valid_df, on=['center', 'batch'], how='inner')
    ids_cohort = ids_cohort[~ids_cohort["id"].isin(params.get('exclude', []))]
    include_rows = ids_df[ids_df['id'].isin(params.get('include', []))]

    ids_tot = pd.concat([ids_cohort, include_rows])
    ids_tot = ids_tot.reset_index(drop=True)
    subtype_counts = ids_tot['subtype'].value_counts().to_list()

    if args.verbose:
        logger.info(f">>> Found {ids_tot.shape[0]} Patients {subtype_counts} ...")
        logger.info(f">>> Loading/Creating Train Val ID Split File ...")

    split_ids = patient_kfold_split(ids_df=ids_tot,
                                   n_folds=args.cross_val_fold,
                                   random_seed=args.random_seed,
                                   load=args.load_id_list,
                                   split_ids_dir=os.path.join(args.info_dir, "split_ids.json"),
                                   logger=logger)

    if not args.load_data_df:

        crop_list = []
        n_crops = 0

        for idx, patient in ids_tot.iterrows():

            t_start = time.time()
            subtype = patient.subtype

            if args.verbose:
                logger.info(f">>> Get '{patient.id}' Image Patches from XML Annotations...")

            st_wsis_dir = os.path.join(args.wsis_dir, patient.subtype)
            patient_wsis_dir = glob.glob(os.path.join(st_wsis_dir, f"*{patient.id}*"))

            if patient_wsis_dir:
                for slide_dir in patient_wsis_dir:
                    slidename = os.path.basename(slide_dir)[:-4]
                    xml_dir = os.path.join(st_wsis_dir, f"{patient.subtype}_xml", f"{slidename}.xml")

                    slide_section = CropIndexer(type='XML',
                                                crop_size=args.crop_size,
                                                overlap=args.overlap,
                                                multiprocessing=args.multiprocessing,
                                                num_threads=args.num_threads)

                    crops, backgrounds = slide_section.crop(slide_dir=slide_dir, xml_dir=xml_dir)
                    n_crops += len(crops)

                    if args.verbose:
                        logger.info(f"--------> +{len(crops)} Crops | {backgrounds} Background | Total {n_crops}")

                    for crop in crops:
                        crop_comp = {}
                        crop_comp['subtype'] = patient.subtype
                        crop_comp['annot_type'] = 'XML'
                        crop_comp['id'] = patient.id
                        crop_comp['path'] = slide_dir
                        crop_comp['is_tumor'] = True if crop['label'] == 'tumor' else False
                        crop_comp['type'] = patient.subtype if crop['label'] == 'tumor' else crop['label'].strip()
                        crop_comp['top'] = crop['top']
                        crop_comp['left'] = crop['left']
                        crop_comp['size'] = crop['size']

                        crop_list.append(crop_comp)

            else:
                logger.warning(f">>> No Slides Found For '{patient.id}'")

            t_end = time.time()

            if args.verbose:
                logger.info(f"+++ Finished Cropping '{patient.id}' Slides in {round(t_end - t_start, 2)}s!")

            if (idx+1)%10 == 0:
                if args.verbose:
                    logger.warning(">>> Saving DataFrame Checkpoint !")

                data_df = pd.DataFrame(crop_list)
                data_df.to_csv(os.path.join(args.dfs_dir, "crops_df.csv"), index=False)

        if args.verbose:
            logger.info("\n>>> Saving The Final Crops DataFrame ...")

        data_df = pd.DataFrame(crop_list)
        os.makedirs(f"{args.dfs_dir}", exist_ok=True)
        data_df.to_csv(os.path.join(args.dfs_dir, "crops_df.csv"), index=False)

    else:
        try:
            data_df = pd.read_csv(os.path.join(args.dfs_dir, "crops_df.csv"))
        except FileNotFoundError as e:
            logger.error(f"No Training Crops .csv File '{args.dfs_dir}'")
            raise e

    if args.verbose:
        logger.info(">>> Loading Tree Pair Classes ...")

    try:
        with open(os.path.join(args.info_dir, "tree_pair_dict.json"), "r") as file:
            tree_pair_dict = json.load(file)
    except FileNotFoundError as e:
        logger.error(f"No Tree Pair JSON File in '{args.info_dir}'")
        raise e

    if args.verbose:
        logger.info(">>> Saving Training DataFrames For Each Fold ...")

    for fold, fold_ids in split_ids.items():

        if args.verbose:
            logger.info(f">>> [{fold}]: Saving Training DataFrame ...")

        fold_train_df = data_df[data_df['id'].isin([id for subtype in fold_ids['Train'].values() for id in subtype])]
        fold_test_df = data_df[data_df['id'].isin([id for subtype in fold_ids['Val'].values() for id in subtype])]

        fold_train_dir = os.path.join(args.dfs_dir, fold)
        fold_test_dir = os.path.join(args.dfs_dir, fold)

        os.makedirs(fold_train_dir, exist_ok=True)
        os.makedirs(fold_test_dir, exist_ok=True)

        fold_train_df.to_csv(
            os.path.join(fold_train_dir, f"{fold}_train_df.csv"),
            index=False
        )

        fold_test_df.to_csv(
            os.path.join(fold_test_dir, f"{fold}_val_df.csv"),
            index=False
        )

        if args.verbose:
            logger.info(f">>> [{fold}]: Handling Dataset Imbalance Labels ...")

        data_balancer = Balancer(method=args.balance_method,
                                 random_state=args.random_seed,
                                 root_normal_only=True)
        
        balanced_dfs = data_balancer.apply(fold_train_df, tree_pair_dict)

        if args.verbose:
            logger.info(f">>> [{fold}]: Saving Balanced DataFrames to '{args.dfs_dir}' ...")

        balanced_dfs['Root']['label'] = np.where(balanced_dfs['Root']['is_tumor'] == False, 0, 1)
        balanced_dfs['Node']['label'] = np.where(balanced_dfs['Node']['type'].isin(tree_pair_dict['Node']['class_0']), 0, 1)
        balanced_dfs['Leaf1']['label'] = np.where(balanced_dfs['Leaf1']['type'].isin(tree_pair_dict['Leaf1']['class_0']), 0, 1)
        balanced_dfs['Leaf2']['label'] = np.where(balanced_dfs['Leaf2']['type'].isin(tree_pair_dict['Leaf2']['class_0']), 0, 1)

        for stage in tree_pair_dict.keys():

            balanced_dfs[stage] = balanced_dfs[stage].sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
            balanced_dfs[stage].to_csv(
                            os.path.join(fold_train_dir, f"{fold}_{stage}.csv"),
                            index=False
                        )
            
            if args.verbose:
                logger.info(f">>> [{fold}]: Value Counts for '{stage}':")

            if args.verbose:
                logger.info("--"*20)
                for val, count in balanced_dfs[stage]['type'].value_counts().items():
                    logger.info(f"--------> {val} : {count}")
                logger.info(f"--------> Total : {balanced_dfs[stage].shape[0]}")
                logger.info("--"*20)

    if args.verbose:
        logger.critical(" ")
        logger.critical("++"*10+" Script Ended "+"++"*10)