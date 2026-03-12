import os
import sys
import numpy as np
import math
import random
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import multiprocessing as mp
import inspect
from scipy.stats import mode
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.metrics import roc_curve
from tqdm import tqdm

from wsi_manager import tissue

if sys.platform.startswith("win"):
    openslide_path = os.getenv("OPENSLIDE_PATH")
    if openslide_path is None:
        raise EnvironmentError("Environment variable OPENSLIDE_PATH is not set")
    if not hasattr(os, "add_dll_directory"):
        raise RuntimeError("os.add_dll_directory is unavailable. Python >= 3.8 is required on Windows.")

    os.add_dll_directory(openslide_path)

import openslide

class CropList:
    """
    The class CropList implements a type which behaves like a python list, but reads data from disk only when required.
    This allows the usage of large WSI without being constrained by the amount of available memory.
    """
    def __init__(self, indexes, size=None):
        self.indexes = indexes
        self.size = size

    def __add__(self, b):
        return CropList(self.indexes + b.indexes)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        section = self.indexes[idx]
        region = openslide.OpenSlide(
            section['filepath_slide']).read_region([section['left'],
                                                    section['top']],
                                                   section['level'],
                                                   [section['size'],
                                                    section['size']])
        region = region.convert('RGB')
        region = region.resize(size=(self.size, self.size))
        return np.array(region)

    def shuffle(self):
        random.shuffle(self.indexes)
    
class CropIndexer:
    
    def __init__(self, type, crop_size, overlap=1, multiprocessing=True, num_threads=4):

        self.crop_default = crop_size
        self.level = 0
        self.overlap = int(1/overlap)
        self.type = type
        self.multiprocessing = multiprocessing
        self.num_threads = num_threads
    
    @staticmethod
    def parse_xml_mask(xml_path):

        tree = ET.parse(xml_path)
        root = tree.getroot()
        masks_points = []

        for mask in root.find('Annotations'):
            points = []
            label = mask.attrib['PartOfGroup']
            annotation_type = mask.attrib.get('Type', 'Polygon')

            if annotation_type == 'Polygon':
                for coordinate in mask.find('Coordinates'):
                    points.append((float(coordinate.attrib['X']),
                                float(coordinate.attrib['Y'])))
            elif annotation_type == 'Spline':
                spline_points = []
                for coordinate in mask.find('Coordinates'):
                    spline_points.append((float(coordinate.attrib['X']),
                                        float(coordinate.attrib['Y'])))
                line = LineString(spline_points)
                num_interp_points = max(100, len(spline_points) * 2)
                interp_points = np.linspace(0, line.length, num_interp_points)
                points = [line.interpolate(distance).coords[0] for distance in interp_points]

            masks_points.append({'label': label, 'points': points})

        return masks_points

    @staticmethod
    def calculate_intersection_area(patch_polygon, masks_points):

        intersection_areas = defaultdict(float)
        for mask in masks_points:
            mask_polygon = Polygon(mask['points'])
            if not mask_polygon.is_valid:
                mask_polygon = mask_polygon.buffer(0)
            intersection_area = patch_polygon.intersection(mask_polygon).area
            intersection_areas[mask['label']] += intersection_area
        return intersection_areas

    @staticmethod
    def decide_label(intersection_areas):

        if not intersection_areas:
            return []
        max_label = max(intersection_areas, key=intersection_areas.get)
        return max_label if intersection_areas[max_label] > 0 else []

    def patch_label(self, patch, masks_points):

        patch_polygon = Polygon([(patch['left'], patch['top']),
                                 (patch['left']+patch['size'], patch['top']),
                                 (patch['left']+patch['size'], patch['top']+patch['size']),
                                 (patch['left'], patch['top']+patch['size'])]).buffer(0)

        labels = []
        for mask in masks_points:
            if patch_polygon.intersects(Polygon(mask['points'])):
                labels.append(mask['label'])

        if len(labels) > 1:
            intersection_areas = self.calculate_intersection_area(patch_polygon, masks_points)
            labels = self.decide_label(intersection_areas)

        return labels

    def process_chunk(self, chunk, slide_dir, crop_default, step, size,
                      bounds_x, bounds_y, bounds_width, bounds_height,
                      type, masks_points, level):
        
        patches = []
        backgrounds = 0
        slide = openslide.OpenSlide(slide_dir)

        for x, y in chunk:
            if x * step + crop_default > bounds_width or y * step + crop_default > bounds_height:
                continue

            top = bounds_y + step * y
            left = bounds_x + step * x
            patch = {'top': top, 'left': left, 'size': size, 'augmented': False}

            if type == 'XML':
                patch['label'] = self.patch_label(patch, masks_points)
                if len(patch['label']) > 0:
                    patch['label'] = patch['label'][0]
                    patches.append(patch)
            elif type == 'SLIDE':
                region = slide.read_region([left, top], level, [size, size])
                region = region.convert('RGB').resize(size=(crop_default, crop_default))
                if tissue.detect(region, method='gradient'):
                    patches.append(patch)
                else:
                    backgrounds += 1

        return patches, backgrounds

    def crop_parallel(self):

        self.backgrounds = 0
        self.masks_points = None

        if self.type == 'XML':
            self.masks_points = self.parse_xml_mask(xml_path=self.xml_dir)

        slide = openslide.OpenSlide(self.slide_dir)
        downsample = slide.level_downsamples[self.level]

        self.bounds_width = int(slide.properties.get('openslide.bounds-width', slide.dimensions[0]))
        self.bounds_height = int(slide.properties.get('openslide.bounds-height', slide.dimensions[1]))
        self.bounds_x = int(slide.properties.get('openslide.bounds-x', 0))
        self.bounds_y = int(slide.properties.get('openslide.bounds-y', 0))

        self.step = int(self.crop_default / self.overlap)
        self.size = math.floor(self.crop_default / downsample)
        self.patches = []

        chunk_coords = [(x, y) for y in range(self.bounds_height // self.step) for x in range(self.bounds_width // self.step)]
        chunks = [chunk_coords[i::self.num_threads] for i in range(self.num_threads)]

        pool_args = [(chunk, self.slide_dir, self.crop_default, self.step, self.size,
                      self.bounds_x, self.bounds_y, self.bounds_width, self.bounds_height, self.type,
                      self.masks_points, self.level) for chunk in chunks]

        with mp.Pool(self.num_threads) as pool:
            results = pool.starmap(self.process_chunk, pool_args)

        patches, backgrounds = zip(*results)
        self.backgrounds = sum(backgrounds)
        self.patches = sum(patches, [])

        return list(self.patches), self.backgrounds

    def crop_normal(self):

        self.backgrounds = 0
        self.masks_points = None

        if self.type == 'XML':
            self.masks_points = self.parse_xml_mask(xml_path=self.xml_dir)

        slide = openslide.OpenSlide(self.slide_dir)
        downsample = slide.level_downsamples[self.level]

        self.bounds_width = int(slide.properties.get('openslide.bounds-width', slide.dimensions[0]))
        self.bounds_height = int(slide.properties.get('openslide.bounds-height', slide.dimensions[1]))
        self.bounds_x = int(slide.properties.get('openslide.bounds-x', 0))
        self.bounds_y = int(slide.properties.get('openslide.bounds-y', 0))

        self.step = int(self.crop_default / self.overlap)
        self.size = math.floor(self.crop_default / downsample)
        self.sections = []

        for y in range(int(math.floor(self.bounds_height / self.step))):
            for x in range(int(math.floor(self.bounds_width / self.step))):
                if x * self.step + self.crop_default > self.bounds_width or y * self.step + self.crop_default > self.bounds_height:
                    continue

                top = self.bounds_y + self.step * y
                left = self.bounds_x + self.step * x
                patch = {'top': top, 'left': left, 'size': self.size, 'augmented': False}

                if self.type == 'XML':
                    patch['label'] = self.patch_label(patch=patch)
                    if len(patch['label']) > 0:
                        patch['label'] = patch['label'][0]
                        self.sections.append(patch)
                elif self.type == 'SLIDE':
                    if not self.check_background(patch, slide, self.crop_default, self.level):
                        self.sections.append(patch)
                    else:
                        self.backgrounds += 1

        return list(self.sections), self.backgrounds

    def crop(self, slide_dir, xml_dir=None):

        self.slide_dir = slide_dir
        self.xml_dir = xml_dir

        if self.multiprocessing:
            return self.crop_parallel()
        else:
            return self.crop_normal()

    def plot_patches(self, figsize=(10, 10), linewidth=2, edgecolor='black', facecolor='red', alpha=0.3):
        
        thumbnail = openslide.OpenSlide(self.slide_dir).get_thumbnail((self.bounds_width, self.bounds_height))
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(thumbnail)

        for section in self.sections:
            rect = patches.Rectangle(
                (section['left'] - self.bounds_x, section['top'] - self.bounds_y),
                section['size'], section['size'], linewidth=linewidth, 
                edgecolor=edgecolor, facecolor=facecolor, alpha=alpha)
            
            ax.add_patch(rect)

        plt.show()


class CropAnalysis:
    def __init__(self, stage):

        self.stage = stage
        self.metric_names = {
            "pred_ent":"Predictive Entropy",
            "renyi_ent":"Renyi Entropy",
            "vr":"Variation Ratio",
            "mi":"Mutual Information",
            "tv":"Total Variance",
            "margin_conf":"Margin Confidence"
        }
        self.feature_map = {
            'pred_ent': 'apply_pred_ent',
            'renyi_ent': 'apply_renyi_ent',
            'vr': 'apply_vr',
            'mi': 'apply_mi',
            'tv': 'apply_tv',
            'margin_conf': 'apply_margin_conf'
        }

    def get_label(self, stage, row, tree_pair_dict):
        """
        Determine the label (class) based on the given stage, row, and a dictionary of tree pairs.

        Parameters:
        - stage: The current stage of classification (a key in tree_pair_dict).
        - row: A row of data, typically from a DataFrame.
        - tree_pair_dict: Dictionary containing classification information with keys 'class_0' and 'class_1'.

        Returns:
        - 0 or 1 based on the type in the row and the corresponding class in the tree_pair_dict.
        
        Raises:
        - ValueError: If the row's type does not match any known classes in the given stage.
        """
        class_0 = tree_pair_dict[stage]['class_0']
        class_1 = tree_pair_dict[stage]['class_1']
        if row['type'] in class_0:
            return 0
        elif row['type'] in class_1:
            return 1
        else:
            raise ValueError(f"Unknown Type '{row['type']}' for Stage '{stage}'")

    def get_freq_class(self, preds):
        """
        Get the most frequent predicted class from a set of predictions.

        Returns:
        - The most frequent class among the predictions (mode of the predicted classes).
        """
        preds = np.array(preds)
        predicted_classes = np.argmax(preds, axis=1)
        return mode(predicted_classes)[0]

    def __variation_ratio__(self, preds):
        """
        Calculate the variation ratio, which measures the uncertainty in predictions.

        Returns:
        - The variation ratio, a value between 0 and 1 where 0 indicates no uncertainty and 1 indicates maximum uncertainty.
        """
        preds = np.array(preds)
        f_c_star = np.max(np.bincount(np.argmax(preds, axis=1)))
        T = preds.shape[0]
        return 1 - f_c_star / T

    def __shannon_entropy__(self, preds):
        """
        Calculate the Shannon entropy of a set of predictions.

        Returns:
        - The Shannon entropy, a measure of uncertainty in the predictions.
        """
        preds = np.array(preds)
        preds_clipped = np.clip(preds, 1e-10, 1)
        entropy = -np.sum(preds_clipped * np.log2(preds_clipped))
        max_entropy = np.log2(preds.shape[1])
        return entropy / max_entropy

    def __predictive_entropy__(self, preds):
        """
        Calculate the predictive entropy, which measures the expected uncertainty in predictions.

        Returns:
        - The predictive entropy, a measure of uncertainty across different predictions.
        """
        preds = np.array(preds)
        p_star = np.mean(preds, axis=0)
        entropy = -np.sum(p_star * np.log2(p_star + 1e-10))
        max_entropy = np.log2(preds.shape[1])
        return entropy / max_entropy

    def __renyi_entropy__(self, preds, alpha=0.1, bins=50):
        """
        Calculate the Rényi entropy, a generalized measure of entropy.

        Parameters:
        - alpha: The order of the Rényi entropy. Must be greater than 0 and not equal to 1.
        - bins: The number of bins for the probability distribution (default is 50).

        Returns:
        - The Rényi entropy, normalized by the log2 of the number of bins.
        
        Raises:
        - ValueError: If alpha is not greater than 0 or is equal to 1.
        """
        preds = np.array(preds)
        norm_factor = np.log2(bins)
        preds_clipped = np.clip(preds, 1e-10, 1)
        if alpha <= 0 or alpha == 1:
            raise ValueError("Alpha Should Be > 0 and Not Equal to 1")
        if alpha == float('inf'):
            return -np.log(np.max(preds_clipped))/norm_factor
        else:
            renyi = 1.0 / (1.0 - alpha) * np.log(np.sum(preds_clipped ** alpha))
            return renyi/norm_factor

    def __mutual_information__(self, preds):
        """
        Calculate the mutual information, a measure of the reduction in uncertainty.

        Returns:
        - The mutual information, representing the amount of information shared between the predicted probabilities and the true distribution.
        """
        preds = np.array(preds)
        p_star = np.mean(preds, axis=0)
        entropy = -np.sum(p_star * np.log2(p_star + 1e-10))
        expected_entropy = -np.mean(np.sum(preds * np.log2(preds + 1e-10), axis=1))
        max_entropy = np.log2(preds.shape[1])
        return (entropy - expected_entropy) / max_entropy

    def __total_variance__(self, preds):
        """
        Calculate the total variance of the predictions, a measure of spread or dispersion.

        Returns:
        - The total variance, representing how much the predictions vary from their mean.
        """
        preds = np.array(preds)
        p_star = np.mean(preds, axis=0)
        return np.mean(np.sum((preds - p_star) ** 2, axis=1))

    def __margin_of_confidence__(self, preds):
        """
        Calculate the margin of confidence, the difference between the top two predicted probabilities.

        Returns:
        - The average margin of confidence, indicating how much more confident the model is in the most likely class compared to the second most likely class.
        """
        preds = np.array(preds)
        sorted_preds = np.sort(preds, axis=1)
        return np.mean(sorted_preds[:, -1] - sorted_preds[:, -2])

    @staticmethod
    def __bhattacharyya_distance__(dist1, dist2):
        """
        Calculate the Bhattacharyya distance between two discrete probability distributions.

        Parameters:
        - dist1: numpy array representing the first probability distribution.
        - dist2: numpy array representing the second probability distribution.

        Returns:
        - dist: Bhattacharyya distance between the distributions, a measure of the similarity between the two distributions.
        """

        dist1 /= np.sum(dist1)
        dist2 /= np.sum(dist2)
        
        BC = np.sum(np.sqrt(dist1 * dist2))
        BC = min(BC, 1)
        dist = -np.log(BC)
        
        return dist
    
    def apply_pred_ent(self, input):
        print(f"--------> Computing '{self.metric_names['pred_ent']}' ...")
        input['pred_ent'] = input['pred_probs'].apply(self.__predictive_entropy__)
        return input

    def apply_renyi_ent(self, input, **kwargs):
        print(f"--------> Computing '{self.metric_names['renyi_ent']}' ...")
        alpha = kwargs.get('alpha', 0.1)
        bins = kwargs.get('bins', 50)
        input['renyi_ent'] = input['pred_probs'].apply(lambda x: self.__renyi_entropy__(x,
                                                                                       alpha=alpha,
                                                                                       bins=bins))
        return input
    
    def apply_vr(self, input):
        print(f"--------> Computing '{self.metric_names['vr']}' ...")
        input['vr'] = input['pred_probs'].apply(self.__variation_ratio__)
        return input    
    
    def apply_mi(self, input):
        print(f"--------> Computing '{self.metric_names['mi']}' ...")
        input['mi'] = input['pred_probs'].apply(self.__mutual_information__)
        return input

    def apply_tv(self, input):
        print(f"--------> Computing '{self.metric_names['tv']}' ...")
        input['tv'] = input['pred_probs'].apply(self.__total_variance__)
        return input

    def apply_margin_conf(self, input):
        print(f"--------> Computing '{self.metric_names['margin_conf']}' ...")
        input['margin_conf'] = input['pred_probs'].apply(self.__margin_of_confidence__)
        return input

    def apply_metrics(self, input, features=[], **kwargs):
        """
        Apply the specified feature calculation to the input DataFrame.

        Parameters:
        - input: The input DataFrame containing prediction probabilities.
        - feature: A list of strings indicating which feature to calculate (e.g., 'pred_ent', 'renyi_ent', 'vr', etc.).
        - kwargs: Additional arguments to pass to the specific feature function.

        Returns:
        - The modified input DataFrame with the calculated feature added.
        """

        print(">>> Get Uncertainty Metrics ...")
        if features:
            for feature in features:
                if feature in self.feature_map:
                    method_name = self.feature_map[feature]
                    method = getattr(self, method_name)
                    method_params = inspect.signature(method).parameters
                    if 'kwargs' in method_params:
                        input = method(input, **kwargs)
                    else:
                        input = method(input)
                else:
                    raise ValueError(f"Unknown Feature '{feature}' Specified")
            return input 
        else:
            for feature in self.feature_map:
                method_name = self.feature_map[feature]
                method = getattr(self, method_name)
                method_params = inspect.signature(method).parameters
                if 'kwargs' in method_params:
                    input = method(input, **kwargs)
                else:
                    input = method(input)
            return input

    def find_pred_labels(self, input):
        print(">>> Get Prediction Labels ...")
        input['pred'] = input['pred_probs'].apply(self.get_freq_class)
        return input
    
    def get_correct(self, input):
        print(">>> Get Correct/Incorrect Predictions ...")
        input['correct'] = input['label'] == input['pred']
        return input

    def plot_metrics(self, input, savefig=False, save_dir=None, **kwargs):
    
        figsize = kwargs.get('figsize', (10, 20))
        fig_alpha = kwargs.get('fig_alpha', 0.5)
        bins = kwargs.get('bins', 50)
        density = kwargs.get('density', True)

        print(">>> Plot Metrics ...")
        _, axes = plt.subplots(nrows=len(self.feature_map), figsize=figsize)

        for ax, metric in zip(axes, list(self.feature_map.keys())):

            correct = input[input['correct']]
            incorrect = input[~input['correct']]

            min_range = min(correct[metric].min(), incorrect[metric].min())
            max_range = max(correct[metric].max(), incorrect[metric].max())
            range_hist = (min_range, max_range)

            ax.hist(correct[metric], alpha=fig_alpha, bins=bins, density=density, label='Correct')
            ax.hist(incorrect[metric], alpha=fig_alpha, bins=bins, density=density, label='Incorrect')

            correct_hist = np.histogram(correct[metric], bins=bins, range=range_hist, density=True)[0]
            incorrect_hist = np.histogram(incorrect[metric], bins=bins, range=range_hist, density=True)[0]

            correct_hist /= correct_hist.sum()
            incorrect_hist /= incorrect_hist.sum()

            b_distance = self.__bhattacharyya_distance__(correct_hist, incorrect_hist)
            ax.set_title(f'{self.stage} Analysis - {self.metric_names[metric]}, Bhattacharyya Distance = {b_distance:.5f}')
            ax.set_xlabel(self.metric_names[metric])
            ax.set_ylabel('Density')
            ax.legend()

        plt.tight_layout()

        if savefig:
            if save_dir:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, f'{self.stage}_metrics_plot.png')
                plt.savefig(save_path, bbox_inches='tight')
                print(f">>> Figure Saved to: '{save_path}'")
            else:
                raise ValueError("No Save Directory Found, Try to Specify the Saving Directory in 'save_dir'")

        plt.show()

    def plot_dist_vs_renyi_alpha(self, input, alpha_range, savefig=False, save_dir=None, **kwargs):
        """
        Plot the Bhattacharyya distance between the distributions of Rényi entropy for correct and incorrect predictions
        across a range of alpha values, and highlight the minimum and maximum distances.

        Parameters:
        - input_df: DataFrame containing prediction correctness information.
        - preds: DataFrame containing the prediction probabilities.
        - alpha_range: A range of alpha values to calculate the Rényi entropy for.
        - bins: Number of bins for the histograms (default is 50).
        - figsize: Tuple indicating the size of the figure (default is (10, 5)).
        - marker: The marker style for the plot (default is 'o').

        Returns:
        - A plot displaying the Bhattacharyya distance for different values of alpha.
        """

        bins = kwargs.get('bins', 50)
        figsize = kwargs.get('figsize', (10, 6))
        fig_alpha = kwargs.get('fig_alpha', 0.5)
        marker = kwargs.get('marker', 'o')
        distances = []


        for alpha in tqdm(alpha_range, total=len(alpha_range), desc="Calculating Distances", unit='Alpha'):

            input['renyi_ent'] = input['pred_probs'].apply(lambda x: self.__renyi_entropy__(x,
                                                                                alpha=alpha,
                                                                                bins=bins))
            
            correct = input[input['correct']]
            incorrect = input[~input['correct']]
            
            min_range = min(correct['renyi_ent'].min(), incorrect['renyi_ent'].min())
            max_range = max(correct['renyi_ent'].max(), incorrect['renyi_ent'].max())
            range_hist = (min_range, max_range)
            correct_hist = np.histogram(correct['renyi_ent'], bins=bins, range=range_hist, density=True)[0]
            incorrect_hist = np.histogram(incorrect['renyi_ent'], bins=bins, range=range_hist, density=True)[0]

            correct_hist /= correct_hist.sum()
            incorrect_hist /= incorrect_hist.sum()

            distance = self.__bhattacharyya_distance__(correct_hist, incorrect_hist)
            distances.append(distance)

        min_idx = np.argmin(distances)
        max_idx = np.argmax(distances)
        self.min_alpha = alpha_range[min_idx]
        self.max_alpha = alpha_range[max_idx]
        min_distance = distances[min_idx]
        max_distance = distances[max_idx]

        print(f"--------> Minimum Distance: {min_distance:.4f} at Alpha: {self.min_alpha:.4f}")
        print(f"--------> Maximum Distance: {max_distance:.4f} at Alpha: {self.max_alpha:.4f}")
        print(">>> Plot Renyi Entropy Distances for Different Alpha Values ...")

        plt.figure(figsize=figsize)
        plt.plot(alpha_range, distances, marker=marker, linewidth=2, linestyle='--', color='black', label='Distance')

        plt.scatter(self.min_alpha, min_distance, color='red', s=50, zorder=5,
                    label=f'Min Distance {min_distance:.4f} (Alpha={self.min_alpha:.4f})')
        plt.scatter(self.max_alpha, max_distance, color='green', s=50, zorder=5,
                    label=f'Max Distance {max_distance:.4f} (Alpha={self.max_alpha:.4f})')

        plt.title(f'{self.stage} Analysis - Rényi Entropy Distance for Different Alpha Values')
        plt.xlabel('Alpha')
        plt.ylabel('Distance')
        plt.xticks(alpha_range, rotation=90)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if savefig:
            if save_dir:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, f'{self.stage}_renyi_distances_plot.png')
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                print(f">>> Figure Saved to: '{save_path}'")
            else:
                raise ValueError("No Save Directory Found, Try to Specify the Saving Directory in 'save_dir'")
            
        plt.show()

        plt.figure(figsize=figsize)
        input['renyi_ent'] = input['pred_probs'].apply(lambda x: self.__renyi_entropy__(x,
                                                                                        alpha=self.max_alpha,
                                                                                        bins=bins))
        correct = input[input['correct']]
        incorrect = input[~input['correct']]

        min_range = min(correct['renyi_ent'].min(), incorrect['renyi_ent'].min())
        max_range = max(correct['renyi_ent'].max(), incorrect['renyi_ent'].max())
        range_hist = (min_range, max_range)

        plt.hist(correct['renyi_ent'], alpha=fig_alpha, bins=bins, density='renyi_ent', label='Correct')
        plt.hist(incorrect['renyi_ent'], alpha=fig_alpha, bins=bins, density='renyi_ent', label='Incorrect')

        correct_hist = np.histogram(correct['renyi_ent'], bins=bins, range=range_hist, density=True)[0]
        incorrect_hist = np.histogram(incorrect['renyi_ent'], bins=bins, range=range_hist, density=True)[0]

        correct_hist /= correct_hist.sum()
        incorrect_hist /= incorrect_hist.sum()

        b_distance = self.__bhattacharyya_distance__(correct_hist, incorrect_hist)
        plt.title(f"{self.stage} Analysis - {self.metric_names['renyi_ent']}, Bhattacharyya Distance = {b_distance:.5f}")
        plt.xlabel(self.metric_names['renyi_ent'])
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()

        if savefig:
            if save_dir:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, f'{self.stage}_renyi_max_alpha_plot.png')
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                print(f">>> Figure Saved to: '{save_path}'")
            else:
                raise ValueError("No Save Directory Found, Try to Specify the Saving Directory in 'save_dir'")
            
        plt.show()

    def plot_params_vs_renyi_ents(self, input, ent_range, savefig=False, save_dir=None, **kwargs):

        alpha = kwargs.get('alpha', self.max_alpha)
        bins = kwargs.get('bins', 50)
        figsize = kwargs.get('figsize', (10, 20))

        count_params = {'Nu':[], 'Ncc': [], 'Nic': [], 'Niu': [], 'Ncu': []}
        ratio_params = {'Rcc': [], 'Riu': [], 'UA': []}

        for ent in tqdm(ent_range, total=len(ent_range), desc="Calculating Params", unit='Entropy'):

            input['renyi_ent'] = input['pred_probs'].apply(lambda x: self.__renyi_entropy__(x,
                                                                                alpha=alpha,
                                                                                bins=bins))

            input['certain'] = input['renyi_ent'] <= ent
            N = len(input)
            Nu = input[~input['certain']].shape[0]
            Ncc = input[input['correct'] & input['certain']].shape[0]
            Nic = input[~input['correct'] & input['certain']].shape[0]
            Niu = input[~input['correct'] & ~input['certain']].shape[0]
            Ncu = input[input['correct'] & ~input['certain']].shape[0]

            Rcc = Ncc / (Ncc + Nic) if (Ncc + Nic) > 0 else 0
            Riu = Niu / (Niu + Nic) if (Niu + Nic) > 0 else 0
            UA = (Ncc + Niu) / N if N > 0 else 0

            count_params['Nu'].append(Nu)
            count_params['Ncc'].append(Ncc)
            count_params['Nic'].append(Nic)
            count_params['Niu'].append(Niu)
            count_params['Ncu'].append(Ncu)

            ratio_params['Rcc'].append(Rcc)
            ratio_params['Riu'].append(Riu)
            ratio_params['UA'].append(UA)

        print(">>> Plot Parameters for Different Renyi Entropy Values ...")
        _, axs = plt.subplots(2, 1, figsize=figsize)

        for metric, values in ratio_params.items():
            axs[0].plot(ent_range, values, label=metric)

        axs[0].set_title('Ratio Parameters Evaluation vs. Entropy Thresholds')
        axs[0].set_xlabel('Entropy')
        axs[0].set_ylabel('Value')
        axs[0].set_xlim(ent_range[0], ent_range[-1])
        axs[0].set_xticks(np.linspace(ent_range[0], ent_range[-1], len(ent_range)))
        axs[0].set_xticklabels(np.round(np.linspace(ent_range[0], ent_range[-1], len(ent_range)), 4), rotation=90)
        axs[0].set_ylim(0, 1.0)
        axs[0].set_yticks(np.arange(0, 1.1, 0.1))
        axs[0].legend()
        axs[0].grid(True)

        for metric, values in count_params.items():
            axs[1].plot(ent_range, values, label=metric)

        axs[1].set_title('Count Parameters Evaluation vs. Entropy Thresholds')
        axs[1].set_xlabel('Entropy')
        axs[1].set_ylabel('Value')
        axs[1].set_xlim(ent_range[0], ent_range[-1])
        axs[1].set_xticks(np.linspace(ent_range[0], ent_range[-1], len(ent_range)))
        axs[1].set_xticklabels(np.round(np.linspace(ent_range[0], ent_range[-1], len(ent_range)), 4), rotation=90)
        axs[1].set_ylim(0, N)
        axs[1].set_yticks(np.arange(0, N+1, 10000))
        axs[1].legend()
        axs[1].grid(True)
        plt.tight_layout()

        if savefig:
            if save_dir:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, f'{self.stage}_renyi_params_vs_ents.png')
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                print(f">>> Figure Saved to: '{save_path}'")
            else:
                raise ValueError("No Save Directory Found, Try to Specify the Saving Directory in 'save_dir'")
            
        plt.show()

    def apply_uncertainty(self, input, threshold, metric, **kwargs):

        method_name = self.feature_map[metric]
        method = getattr(self, method_name)
        method_params = inspect.signature(method).parameters
        if 'kwargs' in method_params:
            input = method(input, **kwargs)
        else:
            input = method(input)

        figsize = kwargs.get('figsize', (10, 6))
        input['certain'] = input[metric] <= threshold

        N = len(input)
        Nu = input[~input['certain']].shape[0]
        Ni = input[~input['correct']].shape[0]
        Ncc = input[input['correct'] & input['certain']].shape[0]
        Nic = input[~input['correct'] & input['certain']].shape[0]
        Niu = input[~input['correct'] & ~input['certain']].shape[0]
        Ncu = input[input['correct'] & ~input['certain']].shape[0]

        Rcc = Ncc / (Ncc + Nic) if (Ncc + Nic) > 0 else 0
        Riu = Niu / (Niu + Nic) if (Niu + Nic) > 0 else 0
        UA = (Ncc + Niu) / N if N > 0 else 0

        certain_input = input[input['correct']]

        precision_all = precision_score(input['label'], input['pred'])
        precision_certain = precision_score(certain_input['label'], certain_input['pred'])
        recall_all = recall_score(input['label'], input['pred'])
        recall_certain = recall_score(certain_input['label'], certain_input['pred'])
        auc_all = roc_auc_score(input['label'], input['pred'])
        auc_certain = roc_auc_score(certain_input['label'], certain_input['pred'])

        print(f">>> Results for '{self.metric_names[metric]}' with '{threshold}' Threshold:")
        print(f">>> No. Crops: {N}")
        print(f">>> No. Uncertain Crops: {Nu}")
        print(f">>> No. Incorrect Crops: {Ni}")
        print(f">>> No. Correct & Certain Crops: {Ncc}")
        print(f">>> No. Incorrect & Certain Crops: {Nic}")
        print(f">>> No. Incorrect & Uncertain Crops: {Niu}")
        print(f">>> No. Correct & Uncertain Crops: {Ncu}")

        print(f">>> Correct & Certain Ratio: {Rcc:.6f}")
        print(f">>> Incorrect & Uncertain Ratio: {Riu:.6f}")
        print(f">>> Uncertainty Accuracy: {UA:.6f}")

        print(f">>> Precision: (Normal = {precision_all:.6f}, Uncertainty Aware = {precision_certain:.6f})")
        print(f">>> Recall: (Normal = {recall_all:.6f}, Uncertainty Aware = {recall_certain:.6f})")
        print(f">>> AUC: (Normal = {auc_all:.6f}, Uncertainty Aware = {auc_certain:.6f})")

        fpr_all, tpr_all, _ = roc_curve(input['label'], input['pred'])
        auc_all = roc_auc_score(input['label'], input['pred'])

        fpr_certain, tpr_certain, _ = roc_curve(certain_input['label'], certain_input['pred'])
        auc_certain = roc_auc_score(certain_input['label'], certain_input['pred'])

        plt.figure(figsize=figsize)
        plt.plot(fpr_all, tpr_all, label=f'All Data (AUC = {auc_all:.6f})')
        plt.plot(fpr_certain, tpr_certain, label=f'Certain Data (AUC = {auc_certain:.6f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.50)')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()