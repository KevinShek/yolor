import os 
from pathlib import Path
import xml.etree.ElementTree as ET

datasets = ["human"]

def xml2yolo(dir):
    from PIL import Image
    from tqdm import tqdm

    def convert_box(size, box):
        # Convert VisDrone box to YOLO xywh box
        dw = 1. / size[0]
        dh = 1. / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    def image_size(dir, name):
        try:
            img_size = Image.open((dir / 'images' / name.name).with_suffix('.JPG')).size
        except FileNotFoundError:
            img_size = Image.open((dir / 'images' / name.name).with_suffix('.jpg')).size
        
        return img_size

    location_to_save = 'labels'
    location_to_read = 'labels_orig'
    (dir / location_to_save).mkdir(parents=True, exist_ok=True)  # make labels directory
    pbar = tqdm((dir / location_to_read).glob('*.xml'), desc=f'Converting {dir}') # looking for *.xml files
    for f in tqdm((dir / location_to_read).glob('*.xml'), desc=f'Converting {dir}') :
        try:
            img_size = image_size(dir, f)
        except FileNotFoundError:
            with open('missing_images.txt', 'a') as missing:
                missing.writelines(f"Can't find Image under {dir} folder called {f.stem} \n")
        lines = []
        with open(f, 'r') as file:  # read annotation.txt
            myroot = ET.parse(file).getroot()
            for object in myroot.findall('object'):
                bbox_xywh = []
                name = object.find('name').text # finds the content of name within the object type
                cls = datasets.index(str(name)) # finds the index base on the dataset class name
                print(cls)
                # grabbing the bounding box results 
                xmin = object.find('bndbox/xmin').text
                ymin = object.find('bndbox/ymin').text
                xmax = object.find('bndbox/xmax').text
                ymax = object.find('bndbox/ymax').text
                # appending the results into a list
                width = int(xmax) - int(xmin)
                height = int(ymax) - int(ymin)
                bbox_xywh.append(xmin)
                bbox_xywh.append(ymin)
                bbox_xywh.append(width)
                bbox_xywh.append(height)
                print(bbox_xywh)

                box = convert_box(img_size, tuple(map(int, bbox_xywh[0:4])))
                print(box)
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")

                with open(str(f).replace(os.sep + location_to_read + os.sep + f.name, os.sep + location_to_save + os.sep + f.stem + ".txt"), 'w') as fl: # op.sep means / or \\ for windows so it means /labels/f.xml to /labels_new/f.txt
                    fl.writelines(lines)  # write label.txt


def txt2yolo(dir):
    from PIL import Image
    from tqdm import tqdm

    def convert_box(size, box):
        # Convert VisDrone box to YOLO xywh box
        dw = 1. / size[0]
        dh = 1. / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    n = 2 # this is focused for Visdrone video
    location_to_save = 'labels'
    location_to_read = 'labels_orig'
    (dir / location_to_save).mkdir(parents=True, exist_ok=True)  # make labels directory
    pbar = tqdm((dir / location_to_read).glob('*.txt'), desc=f'Converting {dir}')
    for f in pbar:
        img_size = Image.open((dir / 'images' / f.name).with_suffix('.jpg')).size
        lines = []
        with open(f, 'r') as file:  # read annotation.txt
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4 + n] == '0':  # VisDrone 'ignored regions' class 0
                    continue
                cls = int(row[5 + n]) - 1
                box = convert_box(img_size, tuple(map(int, row[0 + n:4 + n])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                with open(str(f).replace(os.sep + location_to_read + os.sep + f.name, os.sep + location_to_save + os.sep + f.stem + ".txt"), 'w') as fl:
                    fl.writelines(lines)  # write label.txt

# Convert
for d in 'trainImages', 'testImages':
    xml2yolo(Path(d))  # convert VisDrone annotations to YOLO labels
