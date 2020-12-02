"""
__author__: satheesh.k
Created: Monday, 30th November 2020 11:59:38 am
"""

import json
import os
from pathlib import Path
import random
import math
import traceback
import copy
import collections
import textwrap
from .utils import _plot_boxes,get_dataloader
from PIL import Image,ImageDraw,ImageFont,ImageOps
from typing import Dict, List, Union, Optional,Tuple
from .config import *


class LabelVisualizer():

    def __init__(self,
                imgs_path: Union[str,Path],
                annotations_path: Union[str,Path],
                annotations_format: Optional[str] = 'coco',
                img_size: Optional[int] = 512,
                class_map: Optional[Dict] = dict(),
                manifest_task_idx: Optional[int] = 1) :

        #Check Images directory path for available images.
        self.imgs_path = imgs_path if isinstance(imgs_path,Path) else Path(imgs_path)
        assert self._check_num_imgs(),f"No images found in {str(self.imgs_path)}, Please check."

        #Check Annotations format and directory path.
        self.annotations_path = annotations_path if isinstance(annotations_path,Path) else Path(annotations_path)
        self.annotations_format = annotations_format.lower()
        assert self.annotations_format in ANNOTATION_FORMATS,f"{annotations_format} format not supported. Currently supporting {ANNOTATION_FORMATS}."
        assert self._check_annontations_dir(),f"No valid {annotations_format} annotations found in {self.annotations_path.name}.Please check"
        self.img_size = img_size
        self.class_map = class_map
        self.previous_batch = []
        
        resize = (self.img_size,self.img_size)
        if annotations_format == "manifest":
            self.dataloader = get_dataloader(self.imgs_path,
                                             self.annotations_path,
                                             self.annotations_format,
                                             resize=resize,
                                             task_idx=manifest_task_idx)
        else:
            self.dataloader = get_dataloader(self.imgs_path,
                                             self.annotations_path,
                                             self.annotations_format,
                                             resize=resize)
            
        if annotations_format not in ["yolo","yolov5"]:
            self.class_map = self.dataloader.get_class_map()

        self.class_color_map = dict()
        avail_colors = copy.deepcopy(COLORS)
        for cat_id,_ in self.class_map.items():
            if len(avail_colors):
                color = random.choice(avail_colors)
            else:
                color = 'green'
            self.class_color_map[cat_id] = color
            if color in avail_colors:
                avail_colors.remove(color)
        
    def  _check_num_imgs(self) -> int:
        file_counts = collections.Counter(p.suffix for p in self.imgs_path.iterdir())
        return sum([file_counts.get("."+ext,0) for ext in IMAGE_EXT])

    def _check_annontations_dir(self) -> bool:

        if self.annotations_format == "coco":
            return self.annotations_path.suffix == ".json"
        
        elif self.annotations_format == "manifest":
            return self.annotations_path.suffix == ".manifest"

        elif self.annotations_format == "pascal":
            return len([xml_file for xml_file in self.annotations_path.glob('*.xml')])

        elif self.annotations_format in ["yolo","yolov5"]:
            return len([txt_file for txt_file in self.annotations_path.glob('*.txt')])
        
        return False

    def show_batch(self,
                   num_imgs: Optional[int] = 9,
                   previous: Optional[bool] = False,
                   save: Optional[bool]=False):

        if previous and len(self.previous_batch):
            batch = self.previous_batch
        else:    
            batch = self.dataloader.get_batch(num_imgs)
            self.previous_batch = batch
            
        drawn_imgs = []

        for img_ann_info in batch:
            img_name = img_ann_info["img"]["image_name"]
            img = img_ann_info["img"]["image"]
            anns = img_ann_info["anns"]
            try:
                drawn_img = _plot_boxes(img,anns,class_map=self.class_map,class_color_map=self.class_color_map)
            except:
                print(f"Could not plot bounding boxes for {img_name}")
                traceback.print_exc()
                continue
            drawn_img = ImageOps.expand(drawn_img, border=IMAGE_BORDER, fill=(255,255,255))

            lines = textwrap.wrap(img_name, width=32)
            y_text = IMAGE_BORDER//2 if len(lines) <= 1 else 0
            dimg = ImageDraw.Draw(drawn_img)
            font = dimg.getfont()
            w = drawn_img.size[0]
            for line in lines:
                width, height = font.getsize(line)
                dimg.multiline_text(((w-width)//2, y_text), line, font=font, fill=(0,0,0))
                y_text += height

            drawn_imgs.append(drawn_img)

        if num_imgs != len(drawn_imgs):
            num_imgs = len(drawn_imgs)
            print(f"Visualizing {num_imgs} images.")
        
        if num_imgs == 1:
            if save:
                drawn_imgs[0].save(self.annotations_format+"_vis.jpg")
            return drawn_imgs[0]
            
        cols = 2 if num_imgs <= 6 else 3
        rows = math.ceil(num_imgs/cols)
        width = cols*(self.img_size+2*IMAGE_BORDER)
        height = rows*(self.img_size+2*IMAGE_BORDER)
        canvas = Image.new('RGB',(width,height),color=(255,255,255))
        idx = 0
        for y in range(0,height,self.img_size+2*IMAGE_BORDER+1):
            for x in range(0,width,self.img_size+2*IMAGE_BORDER+1):
                if idx < num_imgs:
                    canvas.paste(drawn_imgs[idx],(x,y))
                    idx += 1
        if save:
            canvas.save(self.annotations_format+"_vis.jpg")
        return canvas