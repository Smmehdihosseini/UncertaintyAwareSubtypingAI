import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters
import matplotlib.patches as patches
from tqdm import tqdm
import concurrent.futures

if sys.platform.startswith("win"):
    openslide_path = os.getenv("OPENSLIDE_PATH")
    if openslide_path is None:
        raise EnvironmentError("Environment variable OPENSLIDE_PATH is not set")
    if not hasattr(os, "add_dll_directory"):
        raise RuntimeError("os.add_dll_directory is unavailable. Python >= 3.8 is required on Windows.")

    os.add_dll_directory(openslide_path)

import openslide


def detect(region, method='mean_region', **kwargs):
    
    if method=='gradient':

        gradient_thr = kwargs.get('gradient_thr', 0.015)
        dark_thr = kwargs.get('dark_thr', 0.1)
        dark_pixel_thr = kwargs.get('dark_pixel_thr', 70)
        var_thr = kwargs.get('var_thr', 0.0005)

        region_gray = np.array(region.convert("L"))
        sobel_x = filters.sobel_h(region_gray)
        sobel_y = filters.sobel_v(region_gray)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        mean_gradient = np.mean(gradient_magnitude)
        gradient_variance = np.var(gradient_magnitude)
        
        region_np = np.array(region)    
        dark_pixels = np.sum(np.all(region_np < dark_pixel_thr, axis=-1))
        total_pixels = region_np.shape[0] * region_np.shape[1]
        dark_ratio = dark_pixels / total_pixels

        back_art = (mean_gradient < gradient_thr or
                dark_ratio > dark_thr or 
                gradient_variance < var_thr)
            
    elif method=='mean_region':

        mean_thr = kwargs.get('mean_thr', 210)

        back_art = np.mean(region) > mean_thr

    return not back_art

class SlideTissueProcessor:

    def __init__(self, slide_path, x, y, region_width, region_height, crop_size=1000, tissue_method='gradient'):

        self.slide_path = slide_path
        self.x = x
        self.y = y
        self.region_width = region_width
        self.region_height = region_height
        self.crop_size = crop_size
        self.slide = openslide.OpenSlide(slide_path)
        self.tissue_method = tissue_method
        self.tissue_count = 0
        self.non_tissue_count = 0
        self.tissue_regions = []

    def __process_patch__(self, args):

        crop_x, crop_y, = args
        region = self.slide.read_region((crop_x, crop_y), 0, (self.crop_size, self.crop_size))
        region_rgb = region.convert("RGB")
        region_resized = region_rgb.resize((112, 112))

        if detect(region_resized, self.tissue_method):
            return (crop_x, crop_y)
        return None

    def process(self):

        coordinates = [(self.x + i, self.y + j) for i in range(0, self.region_width, self.crop_size) for j in range(0, self.region_height, self.crop_size)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
            results = list(tqdm(executor.map(self.__process_patch__, coordinates), unit='Patch',
                                            total=len(coordinates), desc=">>> Processing the Slide Region"))
        
        for result in results:
            if result is not None:
                self.tissue_regions.append(result)
                self.tissue_count += 1

        if self.tissue_regions:
            print(f">>> Found {self.tissue_count}/{len(results)} Tissue Patches")
        else:
            print(">>> No Tissue Regions Found")

    def add_patch(self, ax, coord):

        crop_x, crop_y = coord
        rect = patches.Rectangle(
            (crop_x - self.x, crop_y - self.y),
            self.crop_size,
            self.crop_size,
            linewidth=self.edgewidth,
            edgecolor=self.edgecolor,
            facecolor=self.tissuecolor,
            alpha=self.tissuealpha)
        
        ax.add_patch(rect)

    def plot_tissue_regions(self,
                            tissuecolor='red',
                            tissuealpha=0.1,
                            edgecolor='black',
                            edgewidth=3):
                
        self.tissuecolor = tissuecolor
        self.tissuealpha = tissuealpha
        self.edgecolor = edgecolor
        self.edgewidth = edgewidth

        thumbnail_width = self.region_width
        thumbnail_height = self.region_height

        print(">>> Add Slide Thumbnail ...")
        thumbnail = self.slide.read_region((self.x, self.y), 0, (thumbnail_width, thumbnail_height)).convert("RGB")
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(thumbnail)

        with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
            futures = [executor.submit(self.add_patch, ax, coord) for coord in self.tissue_regions]
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures), unit='Patch', desc=">>> Add Patches"):
                future.result()

        plt.show()
        plt.close(fig)