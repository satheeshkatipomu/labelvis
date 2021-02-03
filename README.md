# labelvis
This is a small python utility to quickly visualize bounding-box annotations/labels before training models on the data.

![PyPI - Python Version](https://img.shields.io/badge/python-3.7%7C3.8-blue)

## Installation

From PyPI
```
pip install labelvis
```

From source
```
git clone https://github.com/satheeshkatipomu/labelvis.git
cd labelvis
pip install -e .
```

## Usage
### COCO
```
from labelvis.labelvis import LabelVisualizer

imgs_path = "./images"
annotations_path = "./annotations/trainval.json"
annotations_format = "coco" #["coco","pascal","manifest","yolo"]
img_size = 256
labelvis = LabelVisualizer(imgs_path,annotations_path,annotations_format,img_size=img_size)

num_images = 9 #Number Images to Visualize
labelvis.show_batch(num_imgs=num_images)

```
### Output
<p align="center"><img align="centre" src="./assets/coco_vis.jpg" alt="vis output" width = "1716"></p>

### Input format
#### Images

Common for all annotations formats.
```
imgs_path = "/path/to/images"

    /path/to/images
         |_ img001.jpg
         |_ img002.jpg
         |_ img003.jpg
     ...
```
#### Annotations
##### COCO (x,y,w,h)
```
annotations_path = "/path/to/annotations/annotations.json"
```
##### Pascal
```
annotations_path = "/path/to/annotations"
    /path/to/annotations
         |_ img001.xml
         |_ img002.xml
         |_ img003.xml
         ...
```
##### Manifest (output from AWS Sagemaker groundtruth)
```
annotations_path = "/path/to/annotations/output.manifest"
```
##### Yolo
```
annotations_path = "/path/to/annotations"
    /path/to/annotations
         |_ img001.txt
         |_ img002.txt
         |_ img003.txt
         ...
```
