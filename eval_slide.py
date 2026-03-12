import os
import random
import json
import random
import glob
import pandas as pd
import argparse
from distutils.util import strtobool
from utils.log import Logger
import tensorflow as tf
from run.expertdt import ExpertDT
from run.mcexpertdt import MCExpertDT

try:
    with open("_info/config.json", 'r') as file:
        conf = json.load(file)
except:
    raise FileNotFoundError("Couldn't Load Config File. Check if the `config.json` file exists under `_info` directory.")

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Evaluate Patients' Slides Using ExpertDT")

    parser.add_argument("--wsis_dir", type=str, default=conf['wsis_dir'], help="WSIs Folder")
    parser.add_argument("--info_dir", type=str, default=conf['info_dir'], help="Information Files Directory")
    parser.add_argument("--weights_dir", type=str, default=conf['weights_dir'], help='Weights Directory')
    parser.add_argument("--results_dir", type=str, default=conf['results_dir'], help="Results Directory")
    parser.add_argument("--figures_dir", type=str, default=conf['figures_dir'], help="Figures Directory")
    parser.add_argument("--preds_dir", type=str, default=conf['preds_dir'], help="Predictions Files Save Directory")
    parser.add_argument("--log_dir", type=str, default=conf['log_dir'], help="Path to Logs Directory")
    parser.add_argument("--verbose", type=bool, default=True, help="Verbose Print Details")
    parser.add_argument("--crop_size", type=int, default=1000, help="Crops Size")
    parser.add_argument("--model_crop_size", type=int, default=112, help="Model Resize Crop Size")
    parser.add_argument("--wsi_level", type=int, default=0, help="Whole Slide Image Magnification Level During Cropping")
    parser.add_argument("--overlap", type=int, default=1, help="Crops Overlap During Cropping Stage; 1: No Overlap, 2: 50%, etc.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch Size For Inference Multiprocessing")
    parser.add_argument("--weights_id", type=str, default='Fold1', help='Weights Identifier')
    parser.add_argument("--model_mode", type=str, default='MCExpertDT', help="ExpertDT model or MCExpertDT")
    parser.add_argument("--wsi_formats", nargs=3, type=str, default=['scn', 'svs', 'tif'], help="Possible Formats for Whole Slide Images")
    parser.add_argument("--stain_transfer", type=lambda x: bool(strtobool(x)), default=False, help="Use Stain Transfer (true or false)")
    parser.add_argument("--save_plots", type=bool, default=True, help="Save Figures")
    parser.add_argument("--save_preds", type=bool, default=True, help="Save Predictions")
    parser.add_argument("--eval_cases", type=str, default='Train', help="Type of Analyse; Train, Test or Cases")
    parser.add_argument("--cases_id", type=str, default='Lyon', help="If eval_cases set to 'Cases', Cases ID Must Be Set")
    parser.add_argument("--stat_metrics", nargs=4, type=str, default=["mean", "std", "entropy", "conf_95"], help="Analyze Statistics Metrics")
    parser.add_argument("--save_metrics", type=bool, default=False, help="Save Statistic Metrics")
    parser.add_argument("--cross_val", type=str, default="Fold1", help='Cross Validation Evaluation')
    parser.add_argument("--runtime_id", type=str, default="train_Fold1", help="Runtime id")
    parser.add_argument("--gpu_memory", type=str, default=conf['gpu_memory'], help="GPU Memory Usage for Runtime For Each GPU in GB, comma-separated")
    parser.add_argument("--random_seed", type=int, default=42, help="Random Seed for Reproducibility")

    args = parser.parse_args()

    random.seed(args.random_seed)

    logger = Logger(log_dir=args.log_dir,
                    log_name=f'eval_slide_{args.runtime_id}')

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

    try:
        ids_df = pd.read_csv(os.path.join(args.info_dir, "ids.csv"))
    except FileNotFoundError as e:
        logger.error(f"No IDs File in '{args.info_dir}' folder.")
        raise e

    logger.info(">>> Loading Split IDs ...")
    with open(os.path.join(args.info_dir, "split_ids.json"), "r") as file:
        split_ids = json.load(file)

    logger.info(">>> Loading Models Parameters ...")
    with open(os.path.join(args.info_dir, "model_params.json"), "r") as file:
        model_params = json.load(file)[args.runtime_id]

    save_dir = os.path.join(args.results_dir, f"{args.eval_cases.lower()}_eval")

    os.makedirs(os.path.join(save_dir, "preds"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "stats"), exist_ok=True)

    results_path = os.path.join(save_dir, "preds", f"{args.runtime_id}.csv")
    stats_path = os.path.join(save_dir, "stats", f"{args.runtime_id}.csv")

    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        logger.info(">>> Prediction Results Exists, Loading it ...")
    else:
        results_df = pd.DataFrame()
        logger.info(">>> Prediction Results Not Found, Creating New Prediction Results ...")

    if os.path.exists(stats_path):
        stats_df = pd.read_csv(stats_path)
        logger.info(">>> Stats Results Exists, Loading it ...")
    else:
        stats_df = pd.DataFrame()
        logger.info(">>> Stats Results Not Found, Creating New Stats Results ...")

    if args.cross_val!='None':
        split_ids = split_ids[args.cross_val]
    if args.eval_cases=='Train':
        eval_cohort = split_ids['Train']
    elif args.eval_cases=='Test':
        eval_cohort = split_ids['Test']
    else:
        if args.eval_cases=='Cases':
            try:
                with open(os.path.join(args.info_dir, "cases_ids.json"), "r") as file:
                    eval_cohort = json.load(file)
            except:
                logger.error("Cases IDs File Couln't Be Found")
                raise FileNotFoundError("Cases IDs File Couln't Be Found")
            if args.cases_id not in eval_cohort.keys():
                logger.error(f"Couldn't Find the Cases ID in Cases File, Are You Sure About {args.cases_id}")
                raise ModuleNotFoundError(f"Couldn't Find the Cases ID in Cases File, Are You Sure About {args.cases_id}")                
            
            eval_cohort = eval_cohort[args.cases_id]
            if not any(len(lst)>0 for lst in eval_cohort.values()):
                logger.error("Cases Couldn't be Found in Split IDs List")
                raise ModuleNotFoundError("Cases Couldn't be Found in Split IDs List")
        else:
            logger.error("Eval Cases is Not Valid, Try 'Train', 'Test' or 'Cases'")
            raise ModuleNotFoundError("Eval Case in Not Valid, Try 'Train', 'Test' or 'Cases'")

    logger.info(f">>> Evaluating '{args.eval_cases}' Patients:")
         
    for subtype in eval_cohort.keys():

        if args.verbose:
            logger.info(f" ")
            logger.info(f"--"*30)
            logger.info(f">>> [{args.eval_cases}] Going Through '{subtype}' Subtype:")
            logger.info(f"--"*30)
            logger.info(f" ")

        for id in eval_cohort[subtype]:
            
            if args.verbose:
                logger.info(f"++"*30)
                logger.info(f">>> Patient ID: {id}")
                logger.info(f"++"*30)

            results_list = []
            stats_list = []
            
            st_wsis_dir = os.path.join(args.wsis_dir, subtype)
            patient_wsis_dir = glob.glob(os.path.join(st_wsis_dir, f"*{id}*"))

            if patient_wsis_dir:
                for slide_dir in patient_wsis_dir:

                    filename = os.path.basename(slide_dir)
                    slidename = filename[:-4]
                    slideformat = filename[-3:]

                    result_dict = {}
                    stats_dict = {}
                    
                    if not results_df.empty and 'slide' in results_df.columns and slidename in results_df['slide'].values:
                        logger.info(f"--->>> Results Already Exists for Slide {slidename} ---<<<")
                    else:
                        if slideformat in args.wsi_formats:

                            logger.info(f"----------->>> Analyzing Slide {slidename} -----------<<<")

                            result_dict['id'] = id
                            result_dict['slide'] = slidename
                            result_dict['format'] = slideformat

                            if args.model_mode=="ExpertDT":

                                expertdt = ExpertDT(logger=logger,
                                                    crop_size=args.crop_size,
                                                    model_crop_size=args.model_crop_size,
                                                    level=args.wsi_level,
                                                    overlap=args.overlap,
                                                    batch_size=args.batch_size,
                                                    weights_dir=args.weights_dir,
                                                    weights_id = args.weights_id,
                                                    tree_pair_dir=os.path.join(args.info_dir, "tree_pair_dict.json"),
                                                    save_plots=args.save_plots,
                                                    save_preds=args.save_preds,
                                                    model_params=model_params)
                                
                            elif args.model_mode=="MCExpertDT":

                                expertdt = MCExpertDT(logger=logger,
                                                    crop_size=args.crop_size,
                                                    model_crop_size=args.model_crop_size,
                                                    level=args.wsi_level,
                                                    overlap=args.overlap,
                                                    batch_size=args.batch_size,
                                                    weights_dir=args.weights_dir,
                                                    weights_id = args.weights_id,
                                                    tree_pair_dir=os.path.join(args.info_dir, "tree_pair_dict.json"),
                                                    save_plots=args.save_plots,
                                                    save_preds=args.save_preds,
                                                    model_params=model_params,
                                                    stain_transfer=args.stain_transfer,
                                                    stain_info={
                                                            'stain_id':'hes2he_multi',
                                                            'stain_dir':'/path/to/stainer/cyclegan',
                                                            'crop_size':256
                                                    })
                            
                            expertdt.predict(slide_dir=slide_dir)
                            expertdt.save_figs(save_dir=os.path.join(args.figures_dir, args.eval_cases, args.runtime_id),
                                                id=id,
                                                slidename=slidename,
                                                prob_plot='hist')
                            
                            if expertdt.save_preds:
                                save_pred_dir = os.path.join(
                                                    args.preds_dir,
                                                    args.eval_cases,
                                                    args.runtime_id,
                                                    str(id),
                                                    slidename
                                                )
                                
                                expertdt.save_tree(save_dir=save_pred_dir)

                            logger.info(f"--------> Final Results:")
                            logger.info(f"--------> Predicted Subtype: {expertdt.max_subtype}")
                            logger.info(f"--------> Subtype Count:")
                            for st_res, st_count in expertdt.subtype_counts.items():
                                logger.info(f"----------------> {st_res}: {st_count}")

                            result_dict['node_pruned'] = expertdt.node_cert
                            result_dict['node_count_1'] = expertdt.node_count_1
                            result_dict['node_count_2'] = expertdt.node_count_2
                            result_dict['ccRCC'] = expertdt.subtype_counts['ccRCC']
                            result_dict['pRCC'] = expertdt.subtype_counts['pRCC']
                            result_dict['CHROMO'] = expertdt.subtype_counts['CHROMO']
                            result_dict['ONCOCYTOMA'] = expertdt.subtype_counts['ONCOCYTOMA']
                            result_dict['Non-Tumor'] = expertdt.subtype_counts['Non-Tumor']

                            if args.model_mode=="MCExpertDT":
                                result_dict['Uncertain (NT)'] = expertdt.subtype_counts['Uncertain (NT)']
                                result_dict['Uncertain (T)'] = expertdt.subtype_counts['Uncertain (T)']

                            result_dict['subtype'] = subtype
                            result_dict['result'] = expertdt.max_subtype
                            result_dict['is_correct'] = expertdt.max_subtype==subtype

                            results_list.append(result_dict)

                            if args.save_metrics:

                                stats_dict['id'] = id
                                stats_dict['slide'] = slidename
                                stats_dict['node_pruned'] = expertdt.node_cert
                                stats_dict['subtype'] = subtype
                                stats_dict['result'] = expertdt.max_subtype
                                stats_dict['is_correct'] = expertdt.max_subtype==subtype

                                stats_res = expertdt.get_metrics(metrics=args.stat_metrics)
                                stats_dict.update(stats_res)
                                stats_list.append(stats_dict)                

            results_df = pd.concat([results_df, pd.DataFrame(results_list)], ignore_index=True)
            results_df.to_csv(results_path, index=False)
            
            if args.save_metrics:
                stats_df = pd.concat([stats_df, pd.DataFrame(stats_list)], ignore_index=True)
                stats_df.to_csv(stats_path, index=False)

    if args.verbose:
        logger.critical(" ")
        logger.critical("++"*10+" Script Ended "+"++"*10)