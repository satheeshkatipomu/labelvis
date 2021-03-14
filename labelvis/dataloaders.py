"""
__author__: satheesh.k
Created: Wednesday, 2nd December 2020 4:29:37 pm 
"""
from pathlib import Path
import pandas as pd
import numpy as np
from random import shuffle
from joblib import Parallel, delayed
from typing import Optional, Tuple
from PIL import Image
import json
import xml.etree.ElementTree as ET
from .config import IMAGE_EXT
from pycocotools.coco import COCO


class BaseDataLoader:
    def __init__(self):
        pass

    def __getitem__(self, index):
        img_df = self.filtered_df[self.filtered_df["image_path"] == index]
        img_bboxes = []
        scores = []
        for i, row in img_df.iterrows():
            if row["annotation"] is not None and row["category"] is not None:
                img_bboxes.append(row["annotation"] + [row["category"]])
                if "score" in row.keys():
                    scores.append(row["score"])

        img, anns = Resizer(self.resize)({"image_path": index, "anns": img_bboxes})
        item = {"img": {"image_name": index.name, "image": img}, "anns": anns}
        if len(scores) > 0:
            item["scores"] = scores
        return item

    def get_batch(self, num_imgs: Optional[int] = 1, **kwargs):

        self.filtered_df = self.apply_filters(**kwargs)
        unique_images = list(self.filtered_df.image_path.unique())
        actual_num_images = min(len(unique_images), num_imgs)
        if actual_num_images < num_imgs:
            print(f"Found only {actual_num_images} in the dataset.")

        shuffle(unique_images)
        img_indices = unique_images[:actual_num_images]

        backend = "threading"
        r = Parallel(n_jobs=-1, backend=backend)(
            delayed(self.__getitem__)(idx) for idx in img_indices
        )
        return r

    def apply_filters(self, **kwargs):
        df = self.df.copy()
        if (
            "show_only_images_with_no_labels" in kwargs
            and kwargs["show_only_images_with_no_labels"] == True
        ):
            df = self.df[self.df["annotation"].isna() & self.df["category"].isna()]
            return df

        if (
            "show_only_images_with_labels" in kwargs
            and kwargs["show_only_images_with_labels"] == True
        ):
            df = self.df.dropna(axis=0)

        if "filter_categories" in kwargs and kwargs["filter_categories"] is not None:
            filter_labels = kwargs["filter_categories"]
            ds_classes = list(self.df.category.unique())
            labels = []
            if len(filter_labels) > 0:
                labels = (
                    [filter_labels] if isinstance(filter_labels, str) else filter_labels
                )
                for label in labels:
                    if label not in ds_classes:
                        print(
                            f"{label} category is not present in the dataset. Please check"
                        )
            if len(labels) > 0:
                df = self.df[self.df["category"].isin(labels)]
            else:
                df = self.df

        return df


class COCODataLoader(BaseDataLoader):
    def __init__(
        self,
        imgs_path: Path,
        annontations_path: Path,
        resize: Optional[Tuple] = (512, 512),
    ):
        super().__init__()
        self.imgs_path = imgs_path
        self.annotations_path = annontations_path
        self.resize = resize

        self.class_map = dict()
        self.coco = COCO(self.annotations_path)
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.class_map = {cat["id"]: cat["name"] for cat in cats}
        self.imgIds = self.coco.getImgIds()
        self._to_df()

    def __len__(self):
        return len(self.imgIds)

    def get_class_map(self):
        return self.class_map

    def _to_df(self):
        image_paths = []
        annotations = []
        categoires = []
        for img_id in self.imgIds:
            coco_img = self.coco.loadImgs(img_id)[0]
            img_name = coco_img["file_name"]
            img_path = self.imgs_path.joinpath(img_name)
            img_annids = self.coco.getAnnIds(imgIds=coco_img["id"])
            img_anns = self.coco.loadAnns(img_annids)
            for ann in img_anns:
                annotations.append(ann["bbox"])
                categoires.append(ann["category_id"])
                image_paths.append(img_path)
            if len(img_anns) < 1:
                annotations.append(None)
                categoires.append(None)
                image_paths.append(img_path)
        self.df = pd.DataFrame(
            {
                "image_path": image_paths,
                "annotation": annotations,
                "category": categoires,
            }
        )


class PascalDataLoader(BaseDataLoader):
    def __init__(
        self,
        imgs_path: Path,
        annontations_path: Path,
        resize: Optional[Tuple] = (512, 512),
    ):
        super().__init__()
        self.imgs_path = imgs_path
        self.annotations_path = annontations_path
        self.resize = resize

        self.class_map = dict()
        self.annotations = [
            xml_file for xml_file in self.annotations_path.glob("*.xml")
        ]
        self.images = [
            p for p in Path(self.imgs_path).glob("**/*") if p.suffix in IMAGE_EXT
        ]
        self._to_df()

    def __len__(self):
        return len(self.annotations)

    def get_class_map(self):
        return self.class_map

    def _to_df(self):
        image_paths = []
        annotations = []
        categoires = []
        for xml_path in self.annotations:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            img_name = root.find("filename").text.split("/")[-1]
            img_path = self.imgs_path.joinpath(img_name)
            for annot in root.iter("object"):
                category = annot.find("name").text
                if category in self.class_map.values():
                    category_id = [
                        k for k, v in self.class_map.items() if v == category
                    ][0]
                else:
                    category_id = len(self.class_map)
                    self.class_map[category_id] = category
                for box in annot.findall("bndbox"):
                    x = int(box.find("xmin").text)
                    y = int(box.find("ymin").text)
                    w = int(box.find("xmax").text) - x
                    h = int(box.find("ymax").text) - y
                    annotations.append([x, y, w, h])
                    image_paths.append(img_path)
                    categoires.append(category_id)

        for img_path in self.images:
            if img_path not in image_paths:
                annotations.append(None)
                image_paths.append(img_path)
                categoires.append(None)

        self.df = pd.DataFrame(
            {
                "image_path": image_paths,
                "annotation": annotations,
                "category": categoires,
            }
        )


class YoloDataLoader(BaseDataLoader):
    def __init__(
        self,
        imgs_path: Path,
        annontations_path: Path,
        resize: Optional[Tuple] = (512, 512),
    ):
        super().__init__()
        self.imgs_path = imgs_path
        self.annotations_path = annontations_path
        self.resize = resize

        self.class_map = dict()
        self.annotations = [
            txt_file for txt_file in self.annotations_path.glob("*.txt")
        ]
        self.images = [
            p for p in Path(self.imgs_path).glob("**/*") if p.suffix in IMAGE_EXT
        ]
        self._to_df()

    def __len__(self):
        return len(self.annotations)

    def get_class_map(self):
        return self.class_map

    def _to_df(self):
        image_paths = []
        annotations = []
        categoires = []
        for txt_path in self.annotations:
            img_name = txt_path.stem
            img_path = ""
            for ext in IMAGE_EXT:
                path = self.imgs_path / img_name
                path = path.with_suffix("." + ext)
                if path.is_file():
                    img_path = path
                    break
            img = Image.open(img_path)
            img_width, img_height = img.size
            with open(txt_path, "r") as ip:
                lines = ip.readlines()
                for line in lines:
                    line = line.strip("\n").split(" ")
                    category = int(line[0])
                    bbox = []
                    for i in line[1:]:
                        bbox.append(float(i))
                    width = bbox[2] * img_width
                    height = bbox[3] * img_height
                    bbox[0] = int((bbox[0] * img_width) - (width / 2.0))
                    bbox[1] = int((bbox[1] * img_height) - (height / 2.0))
                    bbox[2] = width
                    bbox[3] = height
                    image_paths.append(img_path)
                    annotations.append(bbox)
                    categoires.append(category)
                if len(lines) < 1:
                    annotations.append(None)
                    image_paths.append(img_path)
                    categoires.append(None)

        for img_path in self.images:
            if img_path not in image_paths:
                annotations.append(None)
                image_paths.append(img_path)
                categoires.append(None)

        self.df = pd.DataFrame(
            {
                "image_path": image_paths,
                "annotation": annotations,
                "category": categoires,
            }
        )


class ManifestDataLoader(BaseDataLoader):
    def __init__(
        self,
        imgs_path: Path,
        annontations_path: Path,
        resize: Optional[Tuple] = (512, 512),
        task_idx: Optional[int] = 1,
    ):
        super().__init__()
        self.imgs_path = imgs_path
        self.annotations_path = annontations_path
        self.resize = resize

        self.class_map = dict()
        manifest_anns = []
        task_name = ""
        class_map = dict()
        with open(self.annotations_path, "r") as of:
            lines = of.readlines()
            for line in lines:
                try:
                    json_line = json.loads(line[:-1])
                    task_name = list(json_line.keys())[task_idx]
                    class_map = json_line[task_name + "-metadata"]["class-map"]
                    for obj_id, obj_class in class_map.items():
                        self.class_map[int(obj_id)] = obj_class
                    manifest_anns.append(json_line)
                except:
                    pass

        self.task_name = task_name
        self.annotations = manifest_anns
        self.images = [
            p for p in Path(self.imgs_path).glob("**/*") if p.suffix in IMAGE_EXT
        ]
        self._to_df()

    def __len__(self):
        return len(self.annotations)

    def get_class_map(self):
        return self.class_map

    def _to_df(self):
        image_paths = []
        annotations = []
        categoires = []
        for manifest_ann in self.annotations:
            img_name = manifest_ann["source-ref"].split("/")[-1]
            img_path = self.imgs_path.joinpath(img_name)
            img_bboxes = []
            anns = manifest_ann[self.task_name]["annotations"]
            for ann in anns:
                left = ann["left"]
                top = ann["top"]
                width = ann["width"]
                height = ann["height"]
                category = ann["class_id"]
                image_paths.append(img_path)
                annotations.append([left, top, width, height])
                categoires.append(category)
            if len(anns) < 1:
                annotations.append(None)
                image_paths.append(img_path)
                categoires.append(None)

        for img_path in self.images:
            if img_path not in image_paths:
                annotations.append(None)
                image_paths.append(img_path)
                categoires.append(None)

        self.df = pd.DataFrame(
            {
                "image_path": image_paths,
                "annotation": annotations,
                "category": categoires,
            }
        )


class SimpleJsonDataLoader(BaseDataLoader):
    def __init__(
        self,
        imgs_path: Path,
        annontations_path: Path,
        resize: Optional[Tuple] = (512, 512),
        threshold: float = 0.05,
    ):
        super().__init__()
        self.imgs_path = imgs_path
        self.threshold = threshold
        self.class_map = dict()
        self.resize = resize
        self.predictions = self.read_preds(annontations_path)
        self.images = [
            p for p in Path(self.imgs_path).glob("**/*") if p.suffix in IMAGE_EXT
        ]
        self.predictions_images = list(self.predictions.keys())
        self._to_df()

    def __len__(self):
        return len(self.images)

    def get_class_map(self):
        return self.class_map

    def read_preds(self, preds_path: str):
        with open(preds_path, "r") as pp:
            preds = json.load(pp)
        return preds

    def _to_df(self):
        image_paths = []
        annotations = []
        categoires = []
        scores = []
        for img_name in self.predictions_images:
            img_path = self.imgs_path.joinpath(img_name)
            for bboxes in self.predictions[img_name]:
                if bboxes["confidence"] >= self.threshold:
                    bbox = bboxes["bbox"]
                    bbox[2] = bbox[2] - bbox[0]
                    bbox[3] = bbox[3] - bbox[1]
                    image_paths.append(img_path)
                    annotations.append(bbox)
                    category = bboxes["classname"]
                    if category in self.class_map.values():
                        category_id = [
                            k for k, v in self.class_map.items() if v == category
                        ][0]
                    else:
                        category_id = len(self.class_map)
                        self.class_map[category_id] = category
                    categoires.append(category_id)
                    scores.append(bboxes["confidence"])
            if len(self.predictions[img_name]) < 1:
                annotations.append(None)
                image_paths.append(img_path)
                categoires.append(None)
                scores.append(None)

        for img_path in self.images:
            if img_path not in image_paths:
                annotations.append(None)
                image_paths.append(img_path)
                categoires.append(None)
                scores.append(None)

        self.df = pd.DataFrame(
            {
                "image_path": image_paths,
                "annotation": annotations,
                "category": categoires,
                "score": scores,
            }
        )


class Resizer(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, expected_size: Optional[Tuple] = (512, 512)):
        assert isinstance(expected_size, (int, tuple))
        self.expected_size = expected_size

    def __call__(self, sample):

        img_path, anns = sample["image_path"], sample["anns"]
        img = self._get_resized_img(img_path)
        bboxes = self._regress_boxes(anns)
        return img, bboxes

    def _set_letterbox_dims(self):

        iw, ih = self.orig_dim
        ew, eh = self.expected_size

        scale = min(eh / ih, ew / iw)
        nh = int(ih * scale)
        nw = int(iw * scale)
        self.new_dim = (nw, nh)

        offset_x, offset_y = (ew - nw) // 2, (eh - nh) // 2
        self.offset = (offset_x, offset_y)

        upsample_x, upsample_y = iw / nw, ih / nh
        self.upsample = (upsample_x, upsample_y)

    def _get_resized_img(self, img_path: str):

        img = Image.open(img_path)
        self.orig_dim = img.size
        self._set_letterbox_dims()
        img = img.resize(self.new_dim)
        new_img = Image.new("RGB", self.expected_size, color=(255, 255, 255))
        new_img.paste(img, self.offset)
        return new_img

    def _regress_boxes(self, bboxes: np.ndarray):

        if not len(bboxes):
            return []

        if not hasattr(bboxes, "ndim"):
            bboxes = np.array(bboxes)

        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]

        bboxes[:, 0] = bboxes[:, 0] / self.upsample[0]
        bboxes[:, 1] = bboxes[:, 1] / self.upsample[1]
        bboxes[:, 2] = bboxes[:, 2] / self.upsample[0]
        bboxes[:, 3] = bboxes[:, 3] / self.upsample[1]

        bboxes[:, 0] += self.offset[0]
        bboxes[:, 1] += self.offset[1]
        bboxes[:, 2] += self.offset[0]
        bboxes[:, 3] += self.offset[1]

        return bboxes