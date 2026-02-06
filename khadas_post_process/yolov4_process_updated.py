import numpy as np
import argparse
import sys
from ksnn.api import KSNN
from ksnn.types import output_format
import cv2 as cv
import time
import torch
import torchvision
from math import sqrt

# GRID0 = 16 # 512
# GRID1 = 32 # 512
# GRID2 = 64 # 512

# GRID0 = 20 # 640
# GRID1 = 40 # 640
# GRID2 = 80 # 640

NUM_CLS = 1
# MAX_BOXES = 300
CLASSES = ["human"]

# YOLOv4-CSP Constants
# Standard YOLOv4 Scale factors to eliminate grid sensitivity meaning XY_Scale = [1.0, 1.0, 1.0]
XY_SCALE = [1.2, 1.1, 1.05] 
STRIDES = [32, 16, 8]

def organising_pre_data(data):
    LISTSIZE = NUM_CLS + 5 # number of classes + 5
    SPAN = 3

    # data comes in as [Smallest Grid, Medium Grid, Largest Grid] due to reorder='2 1 0'
    # data[0][0] -> Large Feature Map (e.g. 80x80) -> Matches Small Anchors
    # data[0][1] -> Medium Feature Map (e.g. 40x40) -> Matches Medium Anchors
    # data[0][2] -> Small Feature Map (e.g. 20x20) -> Matches Large Anchors
    
    # Note: We must ensure we return them in an order that matches our 'masks' definition
    # KSNN 'reorder 2 1 0' usually reverses the output. 
    # Let's organize strictly by size to be safe.
    
    outputs = [data[0][0], data[0][1], data[0][2]]
    sorted_outputs = sorted(outputs, key=lambda x: len(x), reverse=False) 
    # sorted_outputs is now: [Small Array (Low Res), Medium Array, Large Array (High Res)]
    
    input_data = []
    for out in sorted_outputs:
        grid = int(sqrt(len(out) / (LISTSIZE * SPAN)))
        reshaped = out.reshape(SPAN, LISTSIZE, grid, grid)
        # Transpose to (Grid, Grid, Anchor, Attributes) for processing
        input_data.append(np.transpose(reshaped, (2, 3, 0, 1)))
        
    # Order returned: [Small Grid (20x20), Medium Grid (40x40), Large Grid (80x80)]
    return input_data


def organising_post_data(boxes, classes, scores, NMS_THRESH, MAX_BOXES):
    # 1. Convert NumPy arrays to PyTorch Tensors
    # This fixes the AttributeError
    boxes_t = torch.from_numpy(boxes).float()
    classes_t = torch.from_numpy(classes).long()
    scores_t = torch.from_numpy(scores).float()

    if boxes_t.shape[0] == 0:
        return [torch.zeros(0, 6)]

    # 2. Vectorized Offset Trick (Class-Aware NMS)
    # This shifts boxes so that different classes never overlap, 
    # allowing us to run NMS once for all objects.
    offsets = classes_t.to(boxes_t) * 4096.0
    boxes_for_nms = boxes_t + offsets[:, None]

    # 3. High-Speed NMS
    keep_indices = torchvision.ops.nms(boxes_for_nms, scores_t, NMS_THRESH)

    # 4. Limit results
    if keep_indices.shape[0] > MAX_BOXES:
        keep_indices = keep_indices[:MAX_BOXES]

    # 5. Pack into [x1, y1, x2, y2, score, class]
    final_boxes = boxes_t[keep_indices]
    final_scores = scores_t[keep_indices].unsqueeze(1)
    final_classes = classes_t[keep_indices].unsqueeze(1).float()

    # Concatenate into one [N, 6] tensor
    result_tensor = torch.cat([final_boxes, final_scores, final_classes], dim=1)
    
    # Return as a list of one tensor to match your 'draw' function expectations
    return [result_tensor]
    

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def process(input_layer, mask, anchors, img_size, xy_scale):
    # Select relevant anchors for this layer
    layer_anchors = np.array([anchors[i] for i in mask])
    grid_h, grid_w = input_layer.shape[0:2]
    
    # Calculate Stride: How many pixels does one grid cell represent?
    # e.g., 416 / 13 = 32
    stride_w = img_size[0] / grid_w
    stride_h = img_size[1] / grid_h

    # 1. Apply Sigmoids
    box_xy = sigmoid(input_layer[..., 0:2])
    box_wh = input_layer[..., 2:4] # Raw tw, th
    box_conf = sigmoid(input_layer[..., 4:5])
    box_prob = sigmoid(input_layer[..., 5:])

    # 2. Grid Sensitivity Fix (YOLOv4-CSP)
    box_xy = box_xy * xy_scale - 0.5 * (xy_scale - 1)

    # 3. Create Grid Offsets
    col = np.tile(np.arange(0, grid_w), grid_h).reshape(grid_h, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_w)
    grid = np.stack([col, row], axis=-1)
    grid = np.repeat(grid[:, :, np.newaxis, :], 3, axis=2) # Shape (H, W, 3, 2)
#    grid = (grid + 0.5).astype(np.float32)

    # 4. Final Pixel Coordinates
    box_xy = (box_xy + grid) * [stride_w, stride_h]
    box_wh = np.exp(box_wh) * layer_anchors

<<<<<<< HEAD
=======
    # Add these two lines to fix the "Giant Box" issue
>>>>>>> 13864ca8a79535ab21616a3cd7e422f1214bec06
    box_xy /= img_size  # Normalizes to 0.0 - 1.0
    box_wh /= img_size  # Normalizes to 0.0 - 1.0

    # Convert to Corner Coordinates
    x1y1 = box_xy - (box_wh / 2.0)
    x2y2 = box_xy + (box_wh / 2.0)
    
    boxes = np.concatenate([x1y1, x2y2], axis=-1)
    
    return boxes, box_conf, box_prob
    

def filter_boxes(boxes, box_confidences, box_class_probs, OBJ_THRESH):
<<<<<<< HEAD
    raw_scores = box_confidences * box_class_probs
    max_scores = np.max(raw_scores, axis=-1)
    print("Top 10 raw scores:", sorted(max_scores[max_scores > 0.1], reverse=True)[:10])
=======
>>>>>>> 13864ca8a79535ab21616a3cd7e422f1214bec06
    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= OBJ_THRESH)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores, NMS_THRESH):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Calculate areas using coordinates: (x2 - x1) * (y2 - y1)
    areas = (x2 - x1) * (y2 - y1)
    # order = scores.argsort()[::-1]
    order = np.lexsort((boxes[:, 1], -scores))

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Calculate coordinates of the intersection box
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # Calculate intersection width and height
        # Ensure values are non-negative
        w_inter = np.maximum(0.0, xx2 - xx1)
        h_inter = np.maximum(0.0, yy2 - yy1)
        inter_area = w_inter * h_inter

        # Calculate Intersection over Union (IoU)
        # Add a tiny epsilon (1e-6) to avoid division by zero
        union_area = areas[i] + areas[order[1:]] - inter_area
        ovr = inter_area / (union_area + 1e-6)

        # Keep indices where IoU is below the threshold
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov4_post_process(data, img_size, OBJ_THRESH=0.1, NMS_THRESH=0.6, MAX_BOXES=300):
    input_data = organising_pre_data(data)

# --- ANCHOR CONFIGURATION ---
    # Masks must match the input_data order.
    # input_data[0] is Small Grid (20x20) -> Needs LARGE anchors [6,7,8]
    # input_data[2] is Large Grid (80x80) -> Needs SMALL anchors [0,1,2]
    # YOLOv4-CSP
    masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55],
            [72, 146], [142, 110], [192, 243], [459, 401]]
    
    # yolov4-leaky
    # masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2] ]
    # anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
    #         [59, 119], [116, 90], [156, 198], [373, 326]]

    # Process all layers
    results = [process(input_data[i], masks[i], anchors, img_size, XY_SCALE[i]) for i in range(3)]
    
    all_boxes = np.concatenate([r[0].reshape(-1, 4) for r in results])
    all_confs = np.concatenate([r[1].reshape(-1, 1) for r in results])
    all_probs = np.concatenate([r[2].reshape(-1, NUM_CLS) for r in results])

    boxes, classes, scores = filter_boxes(all_boxes, all_confs, all_probs, OBJ_THRESH)

    output = organising_post_data(boxes, classes, scores, NMS_THRESH, MAX_BOXES)

    return output

def draw(image, detections):
    list_of_images = []
    image_copy = image.copy()

    # Extract the tensor from the list (assuming index 0 contains the detections)
    # detections is expected to be [Tensor(N, 6)]
    if not detections or len(detections) == 0 or detections[0].shape[0] == 0:
        return [image, image, image] # Return originals if no detections

    tensor_data = detections[0] # Get the (N, 6) tensor
    
    print(tensor_data.shape[0])
    possible_target = image.copy()

    for i in range(tensor_data.shape[0]):
        # Extract data from tensor [x1, y1, x2, y2, score, cls]
        x1, y1, x2, y2, score, cl = tensor_data[i].tolist()
        
        # Convert normalized coordinates to absolute pixels
        left = max(0, int(x1 * image.shape[1]))
        top = max(0, int(y1 * image.shape[0]))
        right = min(image.shape[1], int(x2 * image.shape[1]))
        bottom = min(image.shape[0], int(y2 * image.shape[0]))

        # --- YOUR DRAWING LOGIC ---
        current_frame = image

        # Crop (ROI)
        # Ensure coordinates are valid for slicing
        colour = image_copy[top:bottom, left:right]
        
        # Handle case where crop is empty (e.g. box outside image)
        if colour.size == 0:
            colour = np.zeros((10, 10, 3), dtype=np.uint8)

        label = f"{CLASSES[int(cl)]} {score:.2f}" if int(cl) < len(CLASSES) else f"{int(cl)} {score:.2f}"

        for image_type in [possible_target, current_frame]:
            cv.rectangle(image_type, (left, top), (right, bottom), (255, 0, 0), 2)
            cv.putText(image_type, label,
                        (left, top - 6),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
        
        list_of_images.extend((colour, possible_target, current_frame))

    return list_of_images

def main():
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--library", help="Path to C static library file")
    parser.add_argument("--model", help="Path to nbg file")
    parser.add_argument("--picture", help="Path to input picture")
    parser.add_argument("--level", help="Information printer level: 0/1/2")
    args = parser.parse_args()

    if args.model :
        if os.path.exists(args.model) == False:
            sys.exit('Model \'{}\' not exist'.format(args.model))
        model = args.model
    else :
        sys.exit("NBG file not found !!! Please use format: --model")
    if args.picture :
        if os.path.exists(args.picture) == False:
            sys.exit('Input picture \'{}\' not exist'.format(args.picture))
        picture = args.picture
    else :
        sys.exit("Input picture not found !!! Please use format: --picture")
    if args.library :
        if os.path.exists(args.library) == False:
            sys.exit('C static library \'{}\' not exist'.format(args.library))
        library = args.library
    else :
        sys.exit("C static library not found !!! Please use format: --library")
    if args.level == '1' or args.level == '2' :
        level = int(args.level)
    else :
        level = 0


    # (Error checking code omitted for brevity, use your existing logic)
    model = args.model
    picture = args.picture
    library = args.library
    level = int(args.level) if args.level in ['1','2'] else 0

    yolov4 = KSNN('VIM3')
    print(' |---+ KSNN Version: {} +---| '.format(yolov4.get_nn_version()))
    yolov4.nn_init(library=library, model=model, level=level)

    orig_img = cv.imread(picture, cv.IMREAD_COLOR)
    if orig_img is None:
        sys.exit(f"Failed to load image: {picture}")
        
    # Resize to Model Input Size (640x640)
    input_size = (640, 640)
    img_resized = cv.resize(orig_img, input_size)

    # Color Conversion BGR -> RGB 
    # (OpenCV reads BGR, Model usually trained on RGB. 
    # If the model was trained on BGR, swap the lines below)
    # img_in = img_in[::-1, :, :] # This flips channel 0 and 2 (BGR->RGB)
    
    # You do not need to normalise (0 - 1) the image as the NPU takes in uint8 format meaning 0 - 255 values. If you did normalise your data then you will not get the correct results
    # Pack into list for API
    cv_img = [img_resized]

    print('Start inference ...')
    start = time.time()
    
    # Run Inference
    # Note: reorder='2 1 0' typically returns [LargeTensor, MedTensor, SmallTensor]
    data = yolov4.nn_inference(cv_img, platform='DARKNET', reorder='2 1 0', output_tensor=3, output_format=output_format.OUT_FORMAT_FLOAT32)
    # Wrap in list to match your structure or pass directly if nn_inference returns list
    if not isinstance(data, list) and not isinstance(data, tuple):
        data = [data] # Fallback
    
    # We wrap it in a list of lists because your organising_pre_data expects data[0][x]
    data_wrapper = [data] 
    
    end = time.time()
    print('Done. inference time: ', end - start)

    # Post Process
    output = yolov4_post_process(data_wrapper, img_size=input_size, OBJ_THRESH=0.37, NMS_THRESH=0.3, MAX_BOXES=300)

    outputs = output[0].numpy()
    print(outputs[:5])
    print(outputs.shape)

    if output is not None:
        list_of_images = draw(orig_img, output)
        cv.imwrite("results/results_yolov4_csp.jpg", list_of_images[-1])
        print("Result saved to results_yolov4.jpg")
    else:
        print("No detections found.")

if __name__ == '__main__':
   main()