"""
__author__: satheesh.k
Created: Wednesday, 2nd December 2020 10:52:48 am
"""

import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional,Dict,Tuple
import bounding_box.bounding_box as bb
from .dataloaders import (COCODataLoader,
                         PascalDataLoader,
                         YoloDataLoader,
                         ManifestDataLoader)


def _plot_boxes(img: Image,
                bboxes: np.ndarray,
                class_map: Optional[Dict] = dict(),
                class_color_map: Optional[Dict] = dict()): 
        draw_img = np.array(img)
        for box in bboxes:
            bbox = list(map(lambda x: max(0,int(x)),box[:-1]))
            category = class_map.get(int(box[-1]),str(int(box[-1])))
            color = class_color_map.get(int(box[-1]),'green')
            bb.add(draw_img,*bbox,category,color=color)
        return Image.fromarray(draw_img)
    
def get_dataloader(imgs_path: Path,
                   annotations_path: Path,
                   annotations_format: str,
                   resize: Optional[Tuple] = (512,512),
                   task_idx: Optional[int] = 1):
    
    if annotations_format == "coco":
        return COCODataLoader(imgs_path, annotations_path,resize=resize)
    elif annotations_format == "pascal":
        return PascalDataLoader(imgs_path, annotations_path,resize=resize)
    elif annotations_format == "yolo":
        return YoloDataLoader(imgs_path, annotations_path,resize=resize)
    elif annotations_format == "manifest":
        return ManifestDataLoader(imgs_path, annotations_path,resize=resize,task_idx=task_idx)
    else:
        raise NotImplementedError