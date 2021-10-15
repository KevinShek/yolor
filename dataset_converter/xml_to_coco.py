import os 
from pathlib import Path
import json
import xml.etree.ElementTree as ET

datasets = ["human"]
START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = None
# If necessary, pre-define category and its id
#  PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
#  "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
#  "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
#  "motorbike": 14, "person": 15, "pottedplant": 16,
#  "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}


def xml2coco(dir):
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
            filename = name.name.with_suffix('.JPG')
        except FileNotFoundError:
            img_size = Image.open((dir / 'images' / name.name).with_suffix('.jpg')).size
            filename = name.name.with_suffix('.jpg')
        return img_size


    def check_and_extract(root, name, length):
        variables = root.findall(name)
        if len(variables) == 0:
            raise ValueError("Can not find %s in %s" % (name, root.tag))
        if length > 0 and len(variables) != length:
            raise ValueError("The size of %s is supposed to be %d, but is %d." % (name, length, len(variables)))
        if length == 1:
            variables = variables[0]
        return variables


    def get_categories(xml_files):
        """Generate category name to id mapping from a list of xml files.
        
        Arguments:
            xml_files {list} -- A list of xml file paths.
        
        Returns:
            dict -- category name to id mapping.
        """
        classes_names = []
        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall("object"):
                classes_names.append(member[0].text)
        classes_names = list(set(classes_names))
        classes_names.sort()
        return {name: i for i, name in enumerate(classes_names)}


    def get_filename_as_int(filename):
        try:
            filename = filename.replace("\\", "/")
            filename = os.path.splitext(os.path.basename(filename))[0]
            return int(filename), False
        except:
            print("the image id needs to be integer, we will convert the data now")
            return None


    location_to_save = 'labels'
    location_to_read = 'labels_orig'
    (dir / location_to_save).mkdir(parents=True, exist_ok=True)  # make labels directory
    pbar = tqdm((dir / location_to_read).glob('*.xml'), desc=f'Converting {dir}') # looking for *.xml files
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    if PRE_DEFINE_CATEGORIES is not None:
        categories = PRE_DEFINE_CATEGORIES
    else:
        categories = get_categories(pbar)
    bnd_id = START_BOUNDING_BOX_ID

    i = 0

    for f in pbar:
        try:
            img_size, filename = image_size(dir, f)
        except FileNotFoundError:
            with open('missing_images.txt', 'a') as missing:
                missing.writelines(f"Can't find Image under {dir} folder called {f.stem} \n")
        with open(f, 'r') as file:  # read annotation.txt
            myroot = ET.parse(file).getroot()
            
            xml_image_id = check_and_extract(myroot, "filename", 1).text # the id needs to be integer so need to convert it
            image_id = get_filename_as_int(xml_image_id)

            # this would make a new rearrangement for the id assoicated with the images
            if image_id == None:
                image_id = i + 1
                i = image_id

                if image_id == 1:
                    with open("new_image_id.txt", "a") as new_image_id_txt:
                        new_image_id_txt.writelines(f"old image id = new image id")
                with open("new_image_id.txt", "a") as new_image_id_txt:
                    new_image_id_txt.writelines(f"{xml_image_id} = {image_id}")

            image = {
                "file_name": filename,
                "height": img_size[1],
                "width": img_size[0],
                "id": image_id,
            }
            json_dict["images"].append(image)

            for object in myroot.findall('object'):
                bbox_xywh = []
                category = check_and_extract(object, 'name', 1).text # finds the content of name within the object type
                if category not in categories:
                    new_id = len(categories)
                    categories[category] = new_id
                category_id = categories[category]
                # grabbing the bounding box results 
                bbox_xml = check_and_extract(object, 'bndbox', 1)
                xmin = int(check_and_extract(bbox_xml, 'xmin', 1).text) - 1
                ymin = int(check_and_extract(bbox_xml, 'ymin', 1).text) - 1
                xmax = int(check_and_extract(bbox_xml, 'xmax', 1).text)
                ymax = int(check_and_extract(bbox_xml, 'ymax', 1).text)
                # appending the results into a list
                width = abs(xmax - xmin)
                height = abs(ymax - ymin)
                bbox_xywh.append(xmin, ymin, width, height)
                print(bbox_xywh)

                ann = {
                "area": width * height,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [xmin, ymin, width, height],
                "category_id": category_id,
                "id": bnd_id,
                "ignore": 0,
                "segmentation": [],
                }
                json_dict["annotations"].append(ann)
                bnd_id = bnd_id + 1
    
    
    for cate, cid in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)

    json_file = f"{dir}/{location_to_save}/labels.json"
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json_fp = open(json_file, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()

    return json_file        

# Convert
for d in 'trainImages', 'testImages':
    json_file = xml2coco(Path(d))  # convert VisDrone annotations to YOLO labels
    print("Success: {}".format(json_file))
