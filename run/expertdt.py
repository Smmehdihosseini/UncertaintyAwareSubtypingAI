import os
from pathlib import Path
import sys
import math
import numpy as np
import json
from tqdm import tqdm
from scipy import ndimage
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from concurrent.futures import ThreadPoolExecutor, as_completed
import tensorflow as tf
from utils.stats import get_stats

if sys.platform.startswith("win"):
    openslide_path = os.getenv("OPENSLIDE_PATH")
    if openslide_path is None:
        raise EnvironmentError("Environment variable OPENSLIDE_PATH is not set")
    if not hasattr(os, "add_dll_directory"):
        raise RuntimeError("os.add_dll_directory is unavailable. Python >= 3.8 is required on Windows.")

    os.add_dll_directory(openslide_path)

import openslide

from wsi_manager import tissue
from model.vgg16 import Vgg16
from model.weights_loader import find_weights

class ExpertDT:
    
    def __init__(self,
                 logger,
                 crop_size=1000,
                 model_crop_size=112,
                 level=0,
                 overlap=1,
                 batch_size=128,
                 weights_dir='_weights',
                 weights_id=None,
                 tree_pair_dir='_info/tree_pair_dict.json',
                 first_trained_layer = 11,
                 save_plots=True,
                 save_preds=False,
                 model_params=None):

        self.logger = logger
        self.crop_size = crop_size
        self.model_crop_size = model_crop_size
        self.level = level
        self.overlap = overlap
        self.batch_size = batch_size
        self.weights_dir = weights_dir
        self.weights_id = weights_id
        self.first_trained_layer = first_trained_layer
        self.tree_pair_dir = tree_pair_dir
        self.save_plots = save_plots
        self.save_preds = save_preds
        self.model_params = model_params

        self.stages_status = {
                                "Root": "Not Defined",
                                "Node": "Not Defined",
                                "Leaf1": "Not Defined",
                                "Leaf2": "Not Defined",
                            }

        self.load_tree_pair_dict()

        self.final_weights = find_weights(weights_dir=self.weights_dir,
                                          weights_id=self.weights_id,
                                          model_params=self.model_params)
        
        tf.keras.backend.clear_session()

    def load_tree_pair_dict(self):

        with open(f"{self.tree_pair_dir}", 'r') as file:
            self.tree_pair_dict = json.load(file)   

    def process_patch(self, args):

        x, y, step, bounds_y, bounds_x, size = args
        if x * step + self.crop_size > self.bounds_width or y * step + self.crop_size > self.bounds_height:
            return None
        
        top = step * y + bounds_y
        left = step * x + bounds_x
        region = self.slide.read_region([left, top], self.level, [size, size])
        region = region.convert('RGB')
        region = region.resize(size=(self.model_crop_size, self.model_crop_size))
        is_tissue = tissue.detect(region=region, method='gradient')
        
        if is_tissue:
            label = 1
        else:
            label = 0

        patch = {'x': x, 'y': y, 'top': top, 'left': left, 'size': size, 'label': label}

        return patch

    def init_patches(self, slide_dir):

        self.logger.info("--------> Start Slide Analysis ...")

        self.slide = openslide.OpenSlide(slide_dir)
        downsample = self.slide.level_downsamples[self.level]
        self.bounds_width, self.bounds_height = self.slide.dimensions if 'openslide.bounds-width' not in self.slide.properties else (int(self.slide.properties['openslide.bounds-width']), int(self.slide.properties['openslide.bounds-height']))
        self.bounds_x, self.bounds_y = (0, 0) if 'openslide.bounds-x' not in self.slide.properties else (int(self.slide.properties['openslide.bounds-x']), int(self.slide.properties['openslide.bounds-y']))

        _step_ = int(self.crop_size / self.overlap)
        self._size_ = math.floor(self.crop_size / downsample)
        _y_steps_ = int(math.ceil((self.bounds_height - self.crop_size) / _step_))
        _x_steps_ = int(math.ceil((self.bounds_width - self.crop_size) / _step_))

        patch_args = [(x, y, _step_, self.bounds_y, self.bounds_x, self._size_) for y in range(_y_steps_) for x in range(_x_steps_)]

        self.patches_dict = []
        self.slide_array = np.zeros((_y_steps_, _x_steps_))

        self.logger.info("--------> [Step 1/4] Cropping WSI into patches ...")

        with tqdm(total=len(patch_args)) as pbar:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self.process_patch, arg) for arg in patch_args]
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        self.patches_dict.append(result)
                        self.slide_array[result['y'], result['x']] = result['label']
                    pbar.update(1)

        self.patches_dict = list(self.patches_dict)    
    
    def check_mc(self, stage):
        return self.model_params[stage]['model']=='mc'
    
    def _refine(self, input):
        
        central = input[input.size // 2]
        values, counts = np.unique(input, return_counts=True)
        max_freq = counts.max()
        modes = values[counts == max_freq]

        if central==0 or central in modes:
            return central
        
        modes = modes[(modes != 0)]
        if len(modes)==0:
            return central
        elif len(modes) > 0:
            return modes[0]
        
    def refine_output(self, input_crops, size=(3, 3)):

        refined_crops = ndimage.generic_filter(input_crops, self._refine, size=size)

        return refined_crops

    def check_stage_requirement(self, stage):

        if stage == "Node":
            if self.stages_status["Root"] != "Not Defined":
                return True
            else:
                return False
        
        elif stage == "Leaf1" or stage == "Leaf2" :
            if self.stages_status["Node"] != "Not Defined":
                return True
            else:
                return False
            
    def predict_root(self):
        
        vgg16 = Vgg16(input_shape=(self.model_crop_size, self.model_crop_size, 3),
                      type=self.model_params['Root'],
                      n_classes=2,
                      logger=self.logger,
                      first_trained_layer=self.first_trained_layer)
        
        weights_path = Path(self.weights_dir) / self.weights_id / "Root" / self.final_weights["Root"]
        vgg16.load_weights(weights_path=str(weights_path))

        self.root_array = self.slide_array.copy()
        self.root_prob_array = np.zeros_like(self.root_array, dtype=float)

        if self.check_mc('Root'):
            pred_len = self.model_params['Root']['iter']
        else:
            pred_len = 1

        self.root_preds_array = np.zeros((self.root_array.shape[0], self.root_array.shape[1], pred_len, 2))

        image_batch = []
        patch_indices = []

        def process_batch(batch, indices):
            if batch:
                predictions, pred_classes, pred_probs = vgg16.predict(np.array(batch))
                for pred, pred_class, pred_prob, index in zip(predictions, pred_classes, pred_probs, indices):
                    y, x = self.patches_dict[index]['y'], self.patches_dict[index]['x']
                    self.root_array[y, x] = pred_class+1
                    self.root_preds_array[y, x, :, :] = pred
                    self.root_prob_array[y, x] = pred_prob

        self.logger.info("--------> [Step 2/4] >>> Root Stage Analysis ...")

        for patch_index in tqdm(range(len(self.patches_dict))):

            root_condition = self.patches_dict[patch_index]['label'] != 0

            if root_condition:
                region = self.slide.read_region([self.patches_dict[patch_index]['left'],
                                                 self.patches_dict[patch_index]['top']],
                                                self.level,
                                                [self.patches_dict[patch_index]['size'],
                                                 self.patches_dict[patch_index]['size']])
                region = region.convert('RGB').resize(size=(self.model_crop_size, self.model_crop_size))
                image_array = np.array(region) / 255.0

                image_batch.append(image_array)
                patch_indices.append(patch_index)

                if len(image_batch) == self.batch_size:
                    process_batch(image_batch, patch_indices)
                    image_batch = []
                    patch_indices = []

        process_batch(image_batch, patch_indices)

        self.root_refined_array = self.refine_output(self.root_array)
        self.stages_status['Root'] = "Done"

        
    def predict_node(self):
        
        if not self.check_stage_requirement(stage="Node"):
            self.logger.warning("--------> No Root Analysis Found ...")
            return

        vgg16 = Vgg16(input_shape=(self.model_crop_size, self.model_crop_size, 3),
                      type=self.model_params['Node'],
                      n_classes=2,
                      logger=self.logger,
                      first_trained_layer=self.first_trained_layer)
        
        weights_path = Path(self.weights_dir) / self.weights_id / "Node" / self.final_weights["Node"]
        vgg16.load_weights(weights_path=str(weights_path))

        self.node_array = self.root_refined_array.copy()
        self.node_prob_array = np.zeros_like(self.node_array, dtype=float)

        if self.check_mc('Node'):
            pred_len = int(self.model_params['Node']['iter'])
        else:
            pred_len = 1

        self.node_preds_array = np.zeros((self.node_array.shape[0], self.node_array.shape[1], pred_len, 2))

        self.logger.info("--------> [Step 3/4] >>> Node Stage Analysis ...")
        
        image_batch = []
        patch_info_batch = []

        def process_batch(batch, info_batch):
            if not batch:
                return
            
            predictions, pred_classes, pred_probs = vgg16.predict(np.array(batch))
            
            for pred, pred_class, pred_prob, info in zip(predictions, pred_classes, pred_probs, info_batch):
                y, x, patch_index = info
                if pred_class == 0:
                    node_label = 3
                else:
                    node_label = 2

                self.node_array[y, x] = node_label
                self.node_preds_array[y, x, :, :] = pred
                self.node_prob_array[y, x] = pred_prob

        for patch_index in tqdm(range(len(self.patches_dict))):
            root_crop_label = self.root_refined_array[self.patches_dict[patch_index]['y'], self.patches_dict[patch_index]['x']]

            if root_crop_label == 2:
                region = self.slide.read_region([self.patches_dict[patch_index]['left'],
                                                 self.patches_dict[patch_index]['top']],
                                                self.level,
                                                [self.patches_dict[patch_index]['size'],
                                                 self.patches_dict[patch_index]['size']])
                region = region.convert('RGB').resize(size=(self.model_crop_size, self.model_crop_size))
                image_array = np.array(region) / 255.0

                image_batch.append(image_array)
                patch_info_batch.append((self.patches_dict[patch_index]['y'], self.patches_dict[patch_index]['x'], patch_index))

                if len(image_batch) == self.batch_size:
                    process_batch(image_batch, patch_info_batch)
                    image_batch = []
                    patch_info_batch = []

        process_batch(image_batch, patch_info_batch)

        self.node_refined_array = self.refine_output(self.node_array)

        unique_values, counts = np.unique(self.node_refined_array, return_counts=True)
        value_counts = dict(zip(unique_values, counts))

        self.node_count_1, self.node_count_2 = value_counts.get(2, 0), value_counts.get(3, 0)
        
        self.node_cert = 0

        if self.node_count_1+self.node_count_2 != 0:
            self.node_cert = round(abs(self.node_count_1 - self.node_count_2) / (self.node_count_1 + self.node_count_2) * 100, 2)

            if self.node_cert >= self.model_params['Node']['pruning_threshold']:
                node_majority = 1 if self.node_count_1 > self.node_count_2 else 2
                self.logger.info(f"--------> Node Confidence: : {self.node_cert}%, Node to Leaf {node_majority}")
                self.stages_status['Node'] = f'Leaf{node_majority}' 
            else:
                self.stages_status['Node'] = 'Pruned'
                self.logger.info("--------> Node is Pruned! Connecting Root Directly to Leafs ...")
        else:
            self.stages_status['Node'] = 'Pruned'
            self.logger.info("--------> Node is Pruned! Connecting Root Directly to Leafs ...")

    def predict_leaf1(self):

        vgg16 = Vgg16(input_shape=(self.model_crop_size, self.model_crop_size, 3),
                      type=self.model_params['Leaf1'],
                      n_classes=2,
                      logger=self.logger,
                      first_trained_layer=self.first_trained_layer)
        
        weights_path = Path(self.weights_dir) / self.weights_id / "Leaf1" / self.final_weights["Leaf1"]
        vgg16.load_weights(weights_path=str(weights_path))

        self.leaf1_array = self.node_refined_array.copy()
        self.leaf1_array[self.leaf1_array == 3] = 1
        self.leaf1_prob_array = np.zeros_like(self.leaf1_array, dtype=float)

        if self.check_mc('Leaf1'):
            pred_len = int(self.model_params['Leaf1']['model_params']['iter'])
        else:
            pred_len = 1
        
        self.leaf1_preds_array = np.zeros((self.leaf1_array.shape[0], self.leaf1_array.shape[1], pred_len, 2))

        self.logger.info("--------> [Step 4/4] >>> Leaf Stage Analysis: Leaf 1 ...")
        
        image_batch = []
        patch_info_batch = []

        def process_batch(batch, info_batch):
            if not batch:
                return
            
            predictions, pred_classes, pred_probs = vgg16.predict(np.array(batch))
            
            for pred, pred_class, pred_prob, info in zip(predictions, pred_classes, pred_probs, info_batch):
                y, x, patch_index = info
                if pred_class == 0:
                    leaf1_label = 4
                else:
                    leaf1_label = 5

                self.leaf1_array[y, x] = leaf1_label
                self.leaf1_preds_array[y, x, :, :] = pred
                self.leaf1_prob_array[y, x] = pred_prob

        for patch_index in tqdm(range(len(self.patches_dict))):

            node_crop_label = self.node_refined_array[self.patches_dict[patch_index]['y'], self.patches_dict[patch_index]['x']]

            if node_crop_label == 2:
                region = self.slide.read_region([self.patches_dict[patch_index]['left'],
                                                 self.patches_dict[patch_index]['top']],
                                                self.level,
                                                [self.patches_dict[patch_index]['size'],
                                                 self.patches_dict[patch_index]['size']])
                
                region = region.convert('RGB').resize(size=(self.model_crop_size, self.model_crop_size))
                image_array = np.array(region) / 255.0

                image_batch.append(image_array)
                patch_info_batch.append((self.patches_dict[patch_index]['y'], self.patches_dict[patch_index]['x'], patch_index))

                if len(image_batch) == self.batch_size:
                    process_batch(image_batch, patch_info_batch)
                    image_batch = []
                    patch_info_batch = []

        process_batch(image_batch, patch_info_batch)

        self.leaf1_refined_array = self.refine_output(self.leaf1_array)

        unique_leaf1, count_leaf1 = np.unique(self.leaf1_refined_array, return_counts=True)
        self.value_counts_leaf1 = dict(zip(unique_leaf1, count_leaf1))
    
    def predict_leaf2(self):

        vgg16 = Vgg16(input_shape=(self.model_crop_size, self.model_crop_size, 3),
                      type=self.model_params['Leaf2'],
                      n_classes=2,
                      logger=self.logger,
                      first_trained_layer=self.first_trained_layer)
        
        weights_path = Path(self.weights_dir) / self.weights_id / "Leaf2" / self.final_weights["Leaf2"]
        vgg16.load_weights(weights_path=str(weights_path))

        self.leaf2_array = self.node_refined_array.copy()
        self.leaf2_array[self.leaf2_array == 2] = 1
        self.leaf2_prob_array = np.zeros_like(self.leaf2_array, dtype=float)

        if self.check_mc('Leaf2'):
            pred_len = int(self.model_params['Leaf2']['model_params']['iter'])
        else:
            pred_len = 1

        self.leaf2_preds_array = np.zeros((self.leaf2_array.shape[0], self.leaf2_array.shape[1], pred_len, 2))

        self.logger.info("--------> [Step 4/4] >>> Leaf Stage Analysis: Leaf 2 ...")
        
        image_batch = []
        patch_info_batch = []

        def process_batch(batch, info_batch):
            if not batch:
                return
            
            predictions, pred_classes, pred_probs = vgg16.predict(np.array(batch))
            
            for pred, pred_class, pred_prob, info in zip(predictions, pred_classes, pred_probs, info_batch):
                y, x, patch_index = info
                if pred_class == 0:
                    leaf2_label = 2
                else:
                    leaf2_label = 3

                self.leaf2_array[y, x] = leaf2_label
                self.leaf2_preds_array[y, x, :, :] = pred
                self.leaf2_prob_array[y, x] = pred_prob

        for patch_index in tqdm(range(len(self.patches_dict))):

            node_crop_label = self.node_refined_array[self.patches_dict[patch_index]['y'], self.patches_dict[patch_index]['x']]

            if node_crop_label == 3:
                region = self.slide.read_region([self.patches_dict[patch_index]['left'],
                                                 self.patches_dict[patch_index]['top']],
                                                self.level,
                                                [self.patches_dict[patch_index]['size'],
                                                 self.patches_dict[patch_index]['size']])
                
                region = region.convert('RGB').resize(size=(self.model_crop_size, self.model_crop_size))
                image_array = np.array(region) / 255.0

                image_batch.append(image_array)
                patch_info_batch.append((self.patches_dict[patch_index]['y'], self.patches_dict[patch_index]['x'], patch_index))

                if len(image_batch) == self.batch_size:
                    process_batch(image_batch, patch_info_batch)
                    image_batch = []
                    patch_info_batch = []

        process_batch(image_batch, patch_info_batch)

        self.leaf2_refined_array = self.refine_output(self.leaf2_array)

        unique_leaf2, count_leaf2 = np.unique(self.leaf2_refined_array, return_counts=True)
        self.value_counts_leaf2 = dict(zip(unique_leaf2, count_leaf2))

    def predict_leaf(self):

        if not self.check_stage_requirement(stage="Node"):
            self.logger.warning("--------> No Root Analysis Found ...")
            return
        
        if self.stages_status['Node'] == 'Pruned':

            self.predict_leaf1()
            self.predict_leaf2()

            value_counts_leaf = {**self.value_counts_leaf1, **self.value_counts_leaf2}

            ccrcc_count, prcc_count = value_counts_leaf.get(2, 0), value_counts_leaf.get(3, 0)
            chromo_count, onco_count = value_counts_leaf.get(4, 0), value_counts_leaf.get(5, 0)

            self.subtype_counts = {
                                'ccRCC': ccrcc_count,
                                'pRCC': prcc_count,
                                'CHROMO': chromo_count,
                                'ONCOCYTOMA': onco_count
                            }
        
            self.subtype_counts['Non-Tumor'] = int(value_counts_leaf.get(1, 0)/2)
            self.max_subtype = max(self.subtype_counts, key=self.subtype_counts.get)

        elif self.stages_status['Node'] == 'Leaf1':

            self.predict_leaf1()

            chromo_count, onco_count = self.value_counts_leaf1.get(4, 0), self.value_counts_leaf1.get(5, 0)

            self.subtype_counts = {
                                'ccRCC': 0,
                                'pRCC': 0,
                                'CHROMO': chromo_count,
                                'ONCOCYTOMA': onco_count
                            }
            
            self.subtype_counts['Non-Tumor'] = int(self.value_counts_leaf1.get(1, 0))
            self.max_subtype = max(self.subtype_counts, key=self.subtype_counts.get)

        elif self.stages_status['Node'] == 'Leaf2':

            self.predict_leaf2()

            ccrcc_count, prcc_count = self.value_counts_leaf2.get(2, 0), self.value_counts_leaf2.get(3, 0)

            self.subtype_counts = {
                                'ccRCC': ccrcc_count,
                                'pRCC': prcc_count,
                                'CHROMO': 0,
                                'ONCOCYTOMA': 0
                            }
            
            self.subtype_counts['Non-Tumor'] = int(self.value_counts_leaf2.get(1, 0))
            self.max_subtype = max(self.subtype_counts, key=self.subtype_counts.get)

    def predict(self, slide_dir):

        self.init_patches(slide_dir)
        self.predict_root()
        self.predict_node()
        self.predict_leaf()

    def save_tree(self, save_dir):

        tree_info = {}

        tree_info['bounds_x'] = self.bounds_x
        tree_info['bounds_y'] = self.bounds_y
        tree_info['size'] = self._size_
        tree_info['node_status'] = self.stages_status['Node']
        tree_info['pruning_threshold'] = self.model_params['Node']['pruning_threshold']
        tree_info['node_cert'] = self.node_cert

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(os.path.join(save_dir, "tree_info.json"), "w") as file:
            json.dump(tree_info, file, indent=4)

        np.savez_compressed(
            os.path.join(save_dir, "Root.npz"),
            array=self.root_array,
            preds=self.root_preds_array
        )

        np.savez_compressed(
            os.path.join(save_dir, "Node.npz"),
            array=self.node_array,
            preds=self.node_preds_array
        )

        if self.stages_status["Node"] == "Pruned":
            np.savez_compressed(
                os.path.join(save_dir, "Leaf1.npz"),
                array=self.leaf1_array,
                preds=self.leaf1_preds_array
            )
            np.savez_compressed(
                os.path.join(save_dir, "Leaf2.npz"),
                array=self.leaf2_array,
                preds=self.leaf2_preds_array
            )

        elif self.stages_status["Node"] == "Leaf1":
            np.savez_compressed(
                os.path.join(save_dir, "Leaf1.npz"),
                array=self.leaf1_array,
                preds=self.leaf1_preds_array
            )

        elif self.stages_status["Node"] == "Leaf2":
            np.savez_compressed(
                os.path.join(save_dir, "Leaf2.npz"),
                array=self.leaf2_array,
                preds=self.leaf2_preds_array
            )

    def get_metrics(self, metrics):

        metrics_dict = {}

        stages_metrics = ['Root', 'Node']

        if self.stages_status['Node']=='Pruned':
            stages_metrics.append('Leaf1')
            stages_metrics.append('Leaf2')
        else:
            stages_metrics.append(self.stages_status['Node'])

        for stage in stages_metrics:

            if stage=="Root":
                arrays = [self.root_refined_array, self.root_prob_array]
                remove_values = [0.0]

            elif stage=="Node":
                arrays = [self.node_refined_array, self.node_prob_array]
                remove_values = [0.0, 1.0]

            elif stage=="Leaf1":
                arrays = [self.leaf1_refined_array, self.leaf1_prob_array]
                remove_values = [0.0, 1.0]

            elif stage=="Leaf2":
                arrays = [self.leaf2_refined_array, self.leaf2_prob_array]
                remove_values = [0.0, 1.0]

            stage_kwargs = PlotTree().PlotStage(stage=stage)
            unique_labels = np.unique(arrays[0])
            probs_by_class = {}

            for label in unique_labels:
                
                indices = np.where(arrays[0] == label)
                probs_for_class = arrays[1][indices]
                probs_by_class[label] = list(filter(lambda x: x != 0.0, probs_for_class.tolist()))

            for value in remove_values:
                removed_labels = probs_by_class.pop(value, 'Key not found')

            del removed_labels

            probs_by_class = {stage_kwargs['keymap_prob'].get(k, k): v for k, v in probs_by_class.items()}

            for class_name in probs_by_class.keys():
                for metric in metrics:
                    metrics_dict[f'{stage}_{class_name}_{metric}_prob'] = get_stats(type=metric, input=probs_by_class[class_name])

        return metrics_dict

    def save_figs(self, save_dir, id, slidename, prob_plot='hist'):

        if self.save_plots:

            self.logger.info("--------> Saving Figures ...")

            slide_dir = os.path.join(save_dir, id, slidename)
            if not os.path.exists(slide_dir):
                os.makedirs(slide_dir)

            stages_plot = ['Root', 'Node']

            if self.stages_status['Node']=='Pruned':
                stages_plot.append('Leaf1')
                stages_plot.append('Leaf2')
            else:
                stages_plot.append(self.stages_status['Node'])

            for stage in stages_plot:
                self.save_plot_stage(
                    stage,
                    name_dir=os.path.join(slide_dir, f"Heatmap_{stage}_{slidename}.png"),
                    dpi=300,
                    alpha=1
                )

                self.save_plot_stage_probs(
                    stage,
                    plot_type=prob_plot,
                    name_dir=os.path.join(slide_dir, f"Density_{stage}_{slidename}.png"),
                    dpi=300,
                    alpha=1
                )
                
    def save_plot_stage(self, stage, name_dir, dpi=300, alpha=1):
        
        if stage=="Root":
            arrays = [self.root_array, self.root_refined_array]
            alpha_channel = self.root_prob_array.copy()
            alpha_channel[alpha_channel == 0] = 1
            alpha_channel[self.root_refined_array == 0] = 1

        elif stage=="Node":
            arrays = [self.node_array, self.node_refined_array]
            alpha_channel = self.node_prob_array.copy()
            alpha_channel[alpha_channel == 0] = 1
            alpha_channel[self.node_refined_array == 0] = 1

        elif stage=="Leaf1":
            arrays = [self.leaf1_array, self.leaf1_refined_array]
            alpha_channel = self.leaf1_prob_array.copy()
            alpha_channel[alpha_channel == 0] = 1
            alpha_channel[self.leaf1_refined_array == 0] = 1

        elif stage=="Leaf2":
            arrays = [self.leaf2_array, self.leaf2_refined_array]
            alpha_channel = self.leaf2_prob_array.copy()
            alpha_channel[alpha_channel == 0] = 1
            alpha_channel[self.leaf2_refined_array == 0] = 1

        stage_kwargs = PlotTree().PlotStage(stage=stage, alpha=alpha)

        sorted_keys = sorted(stage_kwargs['label_dict'].keys(), key=int)

        labels = [stage_kwargs['label_dict'][key] for key in sorted_keys]
        colors = [stage_kwargs['stage_colors'][int(key)] for key in sorted_keys]
        handles = [Patch(color=color, label=label) for color, label in zip(colors, labels)]

        cmap = ListedColormap(colors)
        bounds = [float(key) - 0.5 for key in sorted_keys] + [float(sorted_keys[-1]) + 0.5]
        norm = BoundaryNorm(bounds, cmap.N)

        fig, axes = plt.subplots(1, len(arrays)+1, figsize=(20, 5))

        rgba_image = cmap(norm(arrays[0]))
        rgba_image[..., 3] = alpha_channel
        im = axes[0].imshow(rgba_image, cmap=cmap, norm=norm)
        axes[0].set_title(stage_kwargs['titles'][0])

        rgba_image = cmap(norm(arrays[1]))
        rgba_image[..., 3] = alpha_channel
        im = axes[1].imshow(rgba_image, cmap=cmap, norm=norm)
        axes[1].set_title(stage_kwargs['titles'][1])

        plt.legend(handles=handles, bbox_to_anchor=(1.65, 1), loc='best')
        plt.tight_layout()
        plt.savefig(name_dir, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()

    def save_plot_stage_probs(self, stage, name_dir, plot_type='hist', dpi=300, alpha=1):

        if stage=="Root":
            arrays = [self.root_refined_array, self.root_prob_array]
            remove_values = [0.0]

        elif stage=="Node":
            arrays = [self.node_refined_array, self.node_prob_array]
            remove_values = [0.0, 1.0]

        elif stage=="Leaf1":
            arrays = [self.leaf1_refined_array, self.leaf1_prob_array]
            remove_values = [0.0, 1.0]

        elif stage=="Leaf2":
            arrays = [self.leaf2_refined_array, self.leaf2_prob_array]
            remove_values = [0.0, 1.0]

        stage_kwargs = PlotTree().PlotStage(stage=stage, alpha=alpha)
        unique_labels = np.unique(arrays[0])
        probs_by_class = {}

        for label in unique_labels:
            indices = np.where(arrays[0] == label)
            probs_for_class = arrays[1][indices]
            probs_by_class[label] = list(filter(lambda x: x != 0.0, probs_for_class.tolist()))

        for value in remove_values:
            removed_labels = probs_by_class.pop(value, 'Key not found')

        del removed_labels

        probs_by_class = {stage_kwargs['keymap_prob'].get(k, k): v for k, v in probs_by_class.items()}

        plt.figure(figsize=(10, 6))
        
        if plot_type=='hist':
            bins = np.arange(0, 1.02, 0.02)
            for class_name, probs in probs_by_class.items():
                plt.hist(probs, bins=bins, alpha=0.5, label=class_name)
        elif plot_type=='kde':
            for class_name, probs in probs_by_class.items():
                sns.kdeplot(probs, label=class_name, multiple="stack")

        plt.title('Probability Distribution by Class')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.xlim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(name_dir, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()

    def plot_stage_on_slide(self, stage, alpha=1):

        if stage=="Root":
            arrays = [self.root_array, self.root_refined_array]
            alpha_channel = self.root_prob_array.copy()
            alpha_channel[alpha_channel == 0] = 1
            alpha_channel[self.root_refined_array == 0] = 1

        elif stage=="Node":
            arrays = [self.node_array, self.node_refined_array]
            alpha_channel = self.node_prob_array.copy()
            alpha_channel[alpha_channel == 0] = 1
            alpha_channel[self.node_refined_array == 0] = 1

        elif stage=="Leaf1":
            arrays = [self.leaf1_array, self.leaf1_refined_array]
            alpha_channel = self.leaf1_prob_array.copy()
            alpha_channel[alpha_channel == 0] = 1
            alpha_channel[self.leaf1_refined_array == 0] = 1

        elif stage=="Leaf2":
            arrays = [self.leaf2_array, self.leaf2_refined_array]
            alpha_channel = self.leaf2_prob_array.copy()
            alpha_channel[alpha_channel == 0] = 1
            alpha_channel[self.leaf2_refined_array == 0] = 1

        stage_kwargs = PlotTree().PlotStage(stage=stage, alpha=alpha)

        sorted_keys = sorted(stage_kwargs['label_dict'].keys(), key=int)

        labels = [stage_kwargs['label_dict'][key] for key in sorted_keys]
        colors = [stage_kwargs['stage_colors'][int(key)] for key in sorted_keys]

        cmap = ListedColormap(colors)
        bounds = [float(key) - 0.5 for key in sorted_keys] + [float(sorted_keys[-1]) + 0.5]
        norm = BoundaryNorm(bounds, cmap.N)

        fig, axes = plt.subplots(1, len(arrays), figsize=(20, 5))

        rgba_image = cmap(norm(arrays[0]))
        rgba_image[..., 3] = alpha_channel
        im = axes[0].imshow(rgba_image, cmap=cmap, norm=norm)
        axes[0].set_title(stage_kwargs['titles'][0])

        rgba_image = cmap(norm(arrays[1]))
        rgba_image[..., 3] = alpha_channel
        im = axes[1].imshow(rgba_image, cmap=cmap, norm=norm)
        axes[1].set_title(stage_kwargs['titles'][1])

        handles = [Patch(color=color, label=label) for color, label in zip(colors, labels)]
        plt.legend(handles=handles, bbox_to_anchor=(1.25, 1), loc='upper left')

        plt.tight_layout()
        plt.show()

class PlotTree:

    def __init__(self):
        
        self.plot_dict = {}

    def PlotStage(self, stage, alpha=1):

        self.colors_rgba = {
                                "Black": (0.0, 0.0, 0.0, alpha),
                                "White": (1.0, 1.0, 1.0, alpha),
                                "Red": (1.0, 0.0, 0.0, alpha),
                                "Orange": (1.0, 0.5, 0.0, alpha),
                                "Yellow": (1.0, 1.0, 0.0, alpha),
                                "Amber": (1.0, 0.75, 0.0, alpha),
                                "Lime Green": (0.5, 1.0, 0.0, alpha),
                                "Green": (0.0, 1.0, 0.0, alpha),
                                "Spring Green": (0.0, 1.0, 0.5, alpha),
                                "Cyan": (0.0, 1.0, 1.0, alpha), 
                                "Blue": (0.0, 0.0, 1.0, alpha),
                                "Purple": (0.5, 0.0, 1.0, alpha),
                                "Magenta": (1.0, 0.0, 1.0, alpha),
                                "Pink": (1.0, 0.75, 0.8, alpha),
                                "Dark Red": (0.6, 0.0, 0.0, alpha),
                                "Brown":(0.6, 0.3, 0.1, alpha),
                                "Olive": (0.5, 0.5, 0.0, alpha),
                                "Navy Blue": (0.0, 0.0, 0.5, alpha),
                                "Teal":(0.0, 0.5, 0.5, alpha),
                                "Violet": (0.5, 0.0, 0.5, alpha) 
                            }

        self.plot_dict = {
            
            "Root": {
                "titles": ['Root', 'Refined Root'],
                "stage_colors": [self.colors_rgba['White'], self.colors_rgba['Pink'], self.colors_rgba['Dark Red'], "", "", "", self.colors_rgba['Amber']],
                "label_dict": {"0": "Background", "1": "Non-Tumor", "2": "Tumor", "6": "Uncertain"},
                "keymap_prob":{1.0:"Non-Tumor", 2.0:"Tumor", 6.0:"Uncertain"}
            },

            "Node": {
                "titles": ['Node', 'Refined Node'],
                "stage_colors": [self.colors_rgba['White'], self.colors_rgba['Pink'], self.colors_rgba['Blue'], self.colors_rgba['Orange'], "", "", self.colors_rgba['Amber']],
                "label_dict": {"0": "Background", "1": "Non-Tumor", "2": "CHROMO/ONCO", "3": "ccRCC/pRCC", "6": "Uncertain"},
                "keymap_prob":{2.0:"CHROMO/ONCOCYTOMA", 3.0:"ccRCC/pRCC", 6.0:"Uncertain"}

            },

            "Leaf1": {
                "titles": ['Leaf1', 'Refined Leaf1'],
                "stage_colors": [self.colors_rgba['White'], self.colors_rgba['Pink'], "", "", self.colors_rgba['Purple'], self.colors_rgba['Magenta'], self.colors_rgba['Amber']],
                "label_dict": {"0": "Background", "1": "Non-Tumor", "4": "CHROMO", "5": "ONCOCYTOMA", "6": "Uncertain"},
                "keymap_prob":{4.0:"CHROMO", 5.0:"ONCOCYTOMA", 6.0:"Uncertain"}
            },

            "Leaf2": {
                "titles": ['Leaf2', 'Refined Leaf2'],
                "stage_colors": [self.colors_rgba['White'], self.colors_rgba['Pink'], self.colors_rgba['Lime Green'], self.colors_rgba['Cyan'], "", "", self.colors_rgba['Amber']],
                "label_dict": {"0": "Background", "1": "Non-Tumor", "2": "ccRCC", "3": "pRCC", "6": "Uncertain"},
                "keymap_prob":{2.0:"ccRCC", 3.0:"pRCC", 6.0:"Uncertain"}
            }
        }

        return self.plot_dict[stage]