"""
__author__: satheesh.k
Created: Wednesday, 2nd December 2020 10:52:48 am
"""

import numpy as np
from pathlib import Path
import textwrap
from PIL import Image, ImageDraw, ImageOps
from typing import Optional, Dict, Tuple, List
import bounding_box.bounding_box as bb
from .dataloaders import (
    COCODataLoader,
    PascalDataLoader,
    YoloDataLoader,
    ManifestDataLoader,
    SimpleJsonDataLoader,
)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def _plot_boxes(
    img: Image,
    bboxes: np.ndarray,
    scores: Optional[List] = None,
    class_map: Optional[Dict] = dict(),
    class_color_map: Optional[Dict] = dict(),
):
    draw_img = np.array(img)
    for i, box in enumerate(bboxes):
        bbox = list(map(lambda x: max(0, int(x)), box[:-1]))
        if not isinstance(box[-1], str):
            category = class_map.get(int(box[-1]), str(int(box[-1])))
        else:
            category = box[-1]
        if scores is not None:
            category = category + ":" + str(round(scores[i], 2))
        color = class_color_map.get(int(box[-1]), "green")
        bb.add(draw_img, *bbox, category, color=color)
    return Image.fromarray(draw_img)


def get_dataloader(
    imgs_path: Path,
    annotations_path: Path,
    annotations_format: str,
    resize: Optional[Tuple] = (512, 512),
    task_idx: Optional[int] = 1,
):

    if annotations_format == "coco":
        return COCODataLoader(imgs_path, annotations_path, resize=resize)
    elif annotations_format == "pascal":
        return PascalDataLoader(imgs_path, annotations_path, resize=resize)
    elif annotations_format == "yolo":
        return YoloDataLoader(imgs_path, annotations_path, resize=resize)
    elif annotations_format == "manifest":
        return ManifestDataLoader(
            imgs_path, annotations_path, resize=resize, task_idx=task_idx
        )
    elif annotations_format == "simple_json":
        return SimpleJsonDataLoader(
            imgs_path, annotations_path, resize=resize, threshold=0.05
        )
    else:
        raise NotImplementedError


def render_grid_mpl(
    drawn_imgs: List,
    image_names: List,
    num_imgs: int,
    cols: int,
    rows: int,
    img_size: int,
    IMAGE_BORDER: int,
    save: bool = False,
    annotation_format: str = "coco",
):
    fig = plt.figure(
        figsize=(
            (rows * img_size + 3 * IMAGE_BORDER * rows) / 72,
            (cols * img_size + 3 * IMAGE_BORDER * cols) / 72,
        )
    )
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(rows, cols),
        axes_pad=0.5,  # pad between axes in inch
    )
    for ax, im, im_name in zip(grid, drawn_imgs, image_names):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_title(im_name)
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in grid[num_imgs:]:
        ax.axis("off")
    if save:
        plt.savefig(annotation_format + "_vis.jpg")
    plt.show()


def render_grid_pil(
    drawn_imgs: List,
    image_names: List,
    num_imgs: int,
    cols: int,
    rows: int,
    img_size: int,
    IMAGE_BORDER: int,
    save: bool = False,
    annotation_format: str = "coco",
):
    for i in range(len(drawn_imgs)):
        drawn_img = drawn_imgs[i]
        img_name = image_names[i]
        drawn_img = ImageOps.expand(
            drawn_img, border=IMAGE_BORDER, fill=(255, 255, 255)
        )
        lines = textwrap.wrap(img_name, width=32)
        y_text = IMAGE_BORDER // 2 if len(lines) <= 1 else 0
        dimg = ImageDraw.Draw(drawn_img)
        font = dimg.getfont()
        w = drawn_img.size[0]
        for line in lines:
            width, height = font.getsize(line)
            dimg.multiline_text(
                ((w - width) // 2, y_text), line, font=font, fill=(0, 0, 0)
            )
            y_text += height
        drawn_imgs[i] = drawn_img

    width = cols * (img_size + 2 * IMAGE_BORDER)
    height = rows * (img_size + 2 * IMAGE_BORDER)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    idx = 0
    for y in range(0, height, img_size + 2 * IMAGE_BORDER + 1):
        for x in range(0, width, img_size + 2 * IMAGE_BORDER + 1):
            if idx < num_imgs:
                canvas.paste(drawn_imgs[idx], (x, y))
                idx += 1
    if save:
        canvas.save(annotation_format + "_vis.jpg")
    return canvas