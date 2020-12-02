"""
__author__: satheesh.k
Created: Wednesday, 2nd December 2020 4:29:37 pm 
"""
from pathlib import Path
import numpy as np
from random import shuffle
from joblib import Parallel, delayed
from typing import Optional,Tuple
from PIL import Image
import json
import xml.etree.ElementTree as ET
from .config import IMAGE_EXT
from pycocotools.coco import COCO

class COCODataLoader():
    
    def __init__(self,
                imgs_path: Path,
                annontations_path: Path,
                resize: Optional[Tuple] = (512,512)):

        self.imgs_path = imgs_path
        self.annotations_path = annontations_path
        self.resize = resize
        
        self.class_map = dict()
        self.coco=COCO(self.annotations_path)
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.class_map = {cat['id']:cat['name'] for cat in cats}
        self.imgIds = self.coco.getImgIds()
        
    def __len__(self):
        return len(self.imgIds)

    def get_class_map(self):
        return self.class_map
    
    def __getitem__(self, index):

        coco_img = self.coco.loadImgs(index)[0]
        img_name = coco_img['file_name']
        img_path = self.imgs_path.joinpath(img_name)
        img_annids = self.coco.getAnnIds(imgIds=coco_img['id'])
        img_anns = self.coco.loadAnns(img_annids)
        img_bboxes = []
        for ann in img_anns:
            img_bboxes.append(ann['bbox']+[ann['category_id']])
        img,anns = Resizer(self.resize)({"image_path":img_path,"anns":img_bboxes})
        item  = {"img":{"image_name":img_path.name,"image":img},"anns": anns}
        return item

    def get_batch(self,num_imgs: Optional[int] = 1):

        actual_num_images = min(self.__len__(),num_imgs)
        if actual_num_images < num_imgs:
            print(f"Found only {actual_num_images} in the dataset.")
            
        shuffle(self.imgIds)
        img_indices = self.imgIds[:actual_num_images]
        
        backend = 'threading'
        r = Parallel(n_jobs=-1,backend=backend)(delayed(self.__getitem__)(idx) for idx in img_indices)
        return r
        


class PascalDataLoader():
    
    def __init__(self,
                imgs_path: Path,
                annontations_path: Path,
                resize: Optional[Tuple] = (512,512)):

        self.imgs_path = imgs_path
        self.annotations_path = annontations_path
        self.resize = resize
        
        self.class_map = dict()
        self.annotations = [xml_file for xml_file in self.annotations_path.glob('*.xml')]
        
    def __len__(self):
        return len(self.annotations)

    def get_class_map(self):
        return self.class_map
    
    def __getitem__(self, index):
        
        xml_path = str(self.annotations[index])
        tree = ET.parse(xml_path)
        root = tree.getroot()
        img_name = root.find('filename').text.split('/')[-1]
        img_path = self.imgs_path.joinpath(img_name)
        img_bboxes = []
        for annot in root.iter("object"):
            category = annot.find("name").text
            if category in self.class_map.values():
                category_id = [k for k,v in self.class_map.items() if v == category][0]
            else:
                category_id = len(self.class_map)
                self.class_map[category_id] = category

            for box in annot.findall("bndbox"):
                x = int(box.find("xmin").text)
                y = int(box.find("ymin").text)
                w = int(box.find("xmax").text)-x
                h = int(box.find("ymax").text)-y
                img_bboxes.append([x,y,w,h,category_id])
                
        img,anns = Resizer(self.resize)({"image_path":img_path,"anns":img_bboxes})
        item  = {"img":{"image_name":img_path.name,"image":img},"anns": anns}
        return item

    def get_batch(self,num_imgs: Optional[int] = 1):

        actual_num_images = min(self.__len__(),num_imgs)
        if actual_num_images < num_imgs:
            print(f"Found only {actual_num_images} in the dataset.")
        
        annontations_idx = list(range(self.__len__()))
        shuffle(annontations_idx)
        ann_indices = annontations_idx[:actual_num_images]
        
        backend = 'threading'
        r = Parallel(n_jobs=-1,backend=backend)(delayed(self.__getitem__)(idx) for idx in ann_indices)
        return r
    
class YoloDataLoader():
    
    def __init__(self,
                imgs_path: Path,
                annontations_path: Path,
                resize: Optional[Tuple] = (512,512)):

        self.imgs_path = imgs_path
        self.annotations_path = annontations_path
        self.resize = resize
        
        self.class_map = dict()
        self.annotations = [txt_file for txt_file in self.annotations_path.glob('*.txt')]
        
    def __len__(self):
        return len(self.annotations)

    def get_class_map(self):
        return self.class_map
    
    def __getitem__(self, index):
        
        txt_path = self.annotations[index]
        img_name = txt_path.stem
        img_path = ""
        for ext in IMAGE_EXT:
            path = self.imgs_path / img_name
            path = path.with_suffix("."+ext)
            if path.is_file():
                img_path = path
                break
        img_bboxes = []
        img = Image.open(img_path)
        img_width,img_height = img.size
        with open(txt_path,'r') as ip:
            lines = ip.readlines()
            for line in lines:
                line = line.strip('\n').split(' ')
                category = int(line[0])
                bbox = []
                for i in line[1:]:
                    bbox.append(float(i))
                width = bbox[2]*img_width
                height = bbox[3]*img_height
                bbox[0] = int((bbox[0]*img_width)-(width/2.0))
                bbox[1] = int((bbox[1]*img_height)-(height/2.0))
                bbox[2] = width
                bbox[3] = height
                bbox.append(category)
                img_bboxes.append(bbox)
                
        img,anns = Resizer(self.resize)({"image_path":img_path,"anns":img_bboxes})
        item  = {"img":{"image_name":img_path.name,"image":img},"anns": anns}
        return item

    def get_batch(self,num_imgs: Optional[int] = 1):

        actual_num_images = min(self.__len__(),num_imgs)
        if actual_num_images < num_imgs:
            print(f"Found only {actual_num_images} in the dataset.")
        
        annontations_idx = list(range(self.__len__()))
        shuffle(annontations_idx)
        ann_indices = annontations_idx[:actual_num_images]
        
        backend = 'threading'
        r = Parallel(n_jobs=-1,backend=backend)(delayed(self.__getitem__)(idx) for idx in ann_indices)
        return r
    
class ManifestDataLoader():
    
    def __init__(self,
                imgs_path: Path,
                annontations_path: Path,
                resize: Optional[Tuple] = (512,512),
                task_idx: Optional[int] = 1):

        self.imgs_path = imgs_path
        self.annotations_path = annontations_path
        self.resize = resize
        
        self.class_map = dict()
        manifest_anns = []
        task_name = ""
        class_map = dict()
        with open(self.annotations_path,'r') as of:
            lines = of.readlines()
            for line in lines:
                json_line = json.loads(line[:-1])
                task_name = list(json_line.keys())[task_idx]
                class_map = json_line[task_name+"-metadata"]["class-map"]
                for obj_id,obj_class in class_map.items():
                    self.class_map[int(obj_id)] = obj_class
                manifest_anns.append(json_line)
        
        self.task_name = task_name
        self.annotations = manifest_anns
        
    def __len__(self):
        return len(self.annotations)

    def get_class_map(self):
        return self.class_map
    
    def __getitem__(self, index):
        
        manifest_ann = self.annotations[index]
        img_name = manifest_ann['source-ref'].split('/')[-1]
        img_path = self.imgs_path.joinpath(img_name)
        img_bboxes = []
        for ann in manifest_ann[self.task_name]["annotations"]:
            left = ann["left"]
            top = ann["top"]
            width = ann["width"]
            height = ann["height"]
            category_id = ann["class_id"]
            img_bboxes.append([left,top,width,height,category_id])        
        img,anns = Resizer(self.resize)({"image_path":img_path,"anns":img_bboxes})
        item  = {"img":{"image_name":img_path.name,"image":img},"anns": anns}
        return item

    def get_batch(self,num_imgs: Optional[int] = 1):

        actual_num_images = min(self.__len__(),num_imgs)
        if actual_num_images < num_imgs:
            print(f"Found only {actual_num_images} in the dataset.")
        
        annontations_idx = list(range(self.__len__()))
        shuffle(annontations_idx)
        ann_indices = annontations_idx[:actual_num_images]
        
        backend = 'threading'
        r = Parallel(n_jobs=-1,backend=backend)(delayed(self.__getitem__)(idx) for idx in ann_indices)
        return r
    
class Resizer(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, expected_size: Optional[Tuple] = (512,512)):
        assert isinstance(expected_size, (int, tuple))
        self.expected_size = expected_size

    def __call__(self, sample):
        
        img_path, anns = sample['image_path'], sample['anns']
        img = self._get_resized_img(img_path)
        bboxes = self._regress_boxes(anns)
        return img,bboxes

    def _set_letterbox_dims(self):

        iw, ih = self.orig_dim
        ew, eh = self.expected_size
        
        
        scale = min(eh / ih, ew / iw)    
        nh = int(ih * scale)
        nw = int(iw * scale)
        self.new_dim = (nw,nh)

        offset_x, offset_y = (ew - nw) // 2, (eh - nh) // 2
        self.offset = (offset_x, offset_y)

        upsample_x,upsample_y = iw / nw, ih / nh
        self.upsample = (upsample_x,upsample_y)
        

    def _get_resized_img(self, img_path: str):

        img = Image.open(img_path)
        self.orig_dim = img.size
        self._set_letterbox_dims()
        img = img.resize(self.new_dim)
        new_img = Image.new('RGB',self.expected_size,color=(255,255,255))
        new_img.paste(img, self.offset)
        return new_img

    def _regress_boxes(self, bboxes: np.ndarray):
        
        if not len(bboxes):
            return []

        if not hasattr(bboxes, "ndim"):
            bboxes = np.array(bboxes)
            
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]
        
        bboxes[:, 0] = bboxes[:, 0]/self.upsample[0]
        bboxes[:, 1] = bboxes[:, 1]/self.upsample[1]
        bboxes[:, 2] = bboxes[:, 2]/self.upsample[0]
        bboxes[:, 3] = bboxes[:, 3]/self.upsample[1]
        
        bboxes[:, 0] += self.offset[0]
        bboxes[:, 1] += self.offset[1]
        bboxes[:, 2] += self.offset[0]
        bboxes[:, 3] += self.offset[1]
        
        return bboxes