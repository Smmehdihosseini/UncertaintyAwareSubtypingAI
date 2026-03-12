import os
from pathlib import Path
import platform
import json
import numpy as np
import xml.etree.ElementTree as ET

OS_MODE = platform.system()

try:
    with open("_info/config.json", 'r') as file:
        conf = json.load(file)
except:
    raise FileNotFoundError("Couldn't Load Config File. Check if the `config.json` file exists under `_info` directory.")

class WholeSlideAnnotator:

    def __init__(self, pred_dir, res_dir, res_name):

        self.stages = ['Root', 'Node', 'Leaf1', 'Leaf2']
        self.pred_dir = pred_dir
        self.res_dir = res_dir
        self.res_name = res_name
        
        self.colors_hex = {
                "Black": "#000000",
                "White": "#FFFFFF",
                "Red": "#FF0000",
                "Dark Red": "#990000",
                "Orange": "#FF8000",
                "Yellow": "#FFFF00",
                "Amber": "#FFBF00",
                "Lime Green": "#80FF00",
                "Green": "#00FF00",
                "Spring Green": "#00FF80",
                "Cyan": "#00FFFF",
                "Blue": "#0000FF",
                "Purple": "#8000FF",
                "Magenta": "#FF00FF",
                "Pink": "#FFBFCF",
                "Dark Pink": "#66534D",
                "Brown": "#994D19",
                "Olive": "#808000",
                "Navy Blue": "#000080",
                "Teal": "#008080",
                "Violet": "#800080"
            }
        
        self.plot_dict = {
            
            "Root": {
                "stage_colors": {"2": self.colors_hex['Dark Red'], "6": self.colors_hex['Dark Pink'], "7": self.colors_hex['Red']},
                "label_dict": {"2": "Tumor", "6": "Uncertain (NT)", "7": "Uncertain (T)"},
                "keymap_prob":{2.0:"Tumor", 6.0:"Uncertain (NT)", 7.0:"Uncertain (T)"}
            },

            "Node": {
                "stage_colors": {"2": self.colors_hex['Blue'], "3": self.colors_hex['Orange'], "6": self.colors_hex['Dark Pink'], "7": self.colors_hex['Red']},
                "label_dict": {"2": "CHROMO/ONCO", "3": "ccRCC/pRCC", "6": "Uncertain (NT)", "7": "Uncertain (T)"},
                "keymap_prob":{3.0:"ccRCC/pRCC", 6.0:"Uncertain (NT)", 7.0:"Uncertain (T)"}

            },

            "Leaf1": {
                "stage_colors": {"4": self.colors_hex['Purple'], "5": self.colors_hex['Navy Blue'], "6": self.colors_hex['Dark Pink'], "7": self.colors_hex['Red']},                
                "label_dict": {"4": "CHROMO", "5": "ONCOCYTOMA", "6": "Uncertain (NT)", "7": "Uncertain (T)"},
                "keymap_prob":{4.0:"CHROMO", 5.0:"ONCOCYTOMA", 6.0:"Uncertain (NT)", 7.0:"Uncertain (T)"}
            },

            "Leaf2": {
                "stage_colors": {"2": self.colors_hex['Lime Green'], "3": self.colors_hex['Cyan'], "6": self.colors_hex['Dark Pink'], "7": self.colors_hex['Red']},                                
                "label_dict": {"2": "ccRCC", "3": "pRCC", "6": "Uncertain (NT)", "7": "Uncertain (T)"},
                "keymap_prob":{2.0:"ccRCC", 3.0:"pRCC", 6.0:"Uncertain (NT)", 7.0:"Uncertain (T)"}
            }
        }
        
    def get_hex_color(self, color):

        return self.colors_hex[color]

    def square_annotations(self, x, y, size, name, color, bounds_x, bounds_y):

        annotation = ET.Element('Annotation', Name=name, Type="Polygon",
                                PartOfGroup=name.split(' - ')[0], Color=color)
        
        coordinates = ET.SubElement(annotation, 'Coordinates')

        top = y * size + bounds_y
        top_next = (y + 1) * size + bounds_y

        left = x * size + bounds_x
        left_next = (x + 1) * size + bounds_x

        sq_coords = [(top, left), (top, left_next),
                    (top_next, left_next), (top_next, left)]

        for i, (coord_y, coord_x) in enumerate(sq_coords):
            ET.SubElement(coordinates, 'Coordinate', Order=str(i),
                        X=str(coord_x), Y=str(coord_y))

        return annotation
    
    def load_data(self, id, slide):

        data_dict = {}
        slide_dir = os.path.join(self.pred_dir, self.res_name, str(id), slide)
        files = os.listdir(slide_dir)

        for stage in self.stages:
            stage_file = f"{stage}.npz"
            if stage_file in files:
                data_dict[stage] = np.load(
                    os.path.join(slide_dir, stage_file)
                )["array"]

        with open(os.path.join(slide_dir, "tree_info.json"), "r") as file:
            tree_info = json.load(file)

        return data_dict, tree_info
    
    def save_as_xml(self, annotations, groups, file_path):

        asap_annotations = ET.Element('ASAP_Annotations')
        annotations_element = ET.SubElement(asap_annotations, 'Annotations')
        for annotation in annotations:
            annotations_element.append(annotation)
        
        group_elements = ET.SubElement(asap_annotations, 'AnnotationGroups')
        for group_name, group_color in groups.items():
            ET.SubElement(group_elements, 'Group', Name=group_name,
                          PartOfGroup="None", Color=group_color)
        
        tree = ET.ElementTree(asap_annotations)
        tree.write(file_path, encoding='utf-8', xml_declaration=True)

    def generate(self):

        for id in os.listdir(os.path.join(self.pred_dir, self.res_name)):
            for slide in os.listdir(os.path.join(self.pred_dir, self.res_name, str(id))):

                data_dict, tree_info = self.load_data(id, slide)

                slide_bounds_x = tree_info['bounds_x']
                slide_bounds_y = tree_info['bounds_y']
                slide_size = tree_info['size']

                for stage, data in data_dict.items():

                    base_save_dir = os.path.join(
                                    self.res_dir,
                                    "annotations",
                                    self.res_name,
                                    str(id),
                                    slide,
                                    stage
                                )
                    os.makedirs(base_save_dir, exist_ok=True)

                    annotations = []
                    colors = self.plot_dict[stage]['stage_colors']
                    labels = self.plot_dict[stage]['stage_colors']
                    total_groups = {labels[label]: colors[label] for label in labels.keys()}

                    for label in labels.keys():
                        y_indices, x_indices = np.where(data == label)
                        for i, (y, x) in enumerate(zip(y_indices, x_indices)):
                            name = f'{labels[label]} - {i}'
                            color = colors[label]
                            annotation = self.square_annotations(
                                                    x,
                                                    y,
                                                    slide_size,
                                                    name,
                                                    color,
                                                    slide_bounds_x,
                                                    slide_bounds_y
                                                )
                            

                            annotations.append(annotation)

                        label_save_dir = os.path.join(base_save_dir, f"{label}.xml")
                        self.save_as_xml([annotation],
                                            {labels[label]: colors[label]},
                                            label_save_dir)

                    stage_save_dir = os.path.join(base_save_dir, f"{stage}.xml")
                    self.save_as_xml(annotations, total_groups, stage_save_dir)