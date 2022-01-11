import numpy as np
import os
import urllib.request
from matplotlib import gridspec
from matplotlib import pyplot as plt
from PIL import Image
import argparse
import sys
import math
from ksnn.api import KSNN
from ksnn.types import output_format
import cv2 as cv
import time
import torch
from math import sqrt

# GRID0 = 16 # 512
# GRID1 = 32 # 512
# GRID2 = 64 # 512

# GRID0 = 20 # 640
# GRID1 = 40 # 640
# GRID2 = 80 # 640

NUM_CLS = 1
MAX_BOXES = 300

CLASSES = "human"

def organising_pre_data(data):

    LISTSIZE = 6 # number of classes + 5
    SPAN = 3

    GRID0 = int(sqrt(len(data[0][2]) / (LISTSIZE * SPAN)))
    GRID1 = int(sqrt(len(data[0][1]) / (LISTSIZE * SPAN)))
    GRID2 = int(sqrt(len(data[0][0]) / (LISTSIZE * SPAN)))

    input0_data = data[0][2]
    input1_data = data[0][1]
    input2_data = data[0][0]
    # print("array0= ", len(data[0][0]))
    # print("array1= ", len(data[0][1]))
    # print("array2= ", len(data[0][2]))

    input0_data = input0_data.reshape(SPAN, LISTSIZE, GRID0, GRID0)
    input1_data = input1_data.reshape(SPAN, LISTSIZE, GRID1, GRID1)
    input2_data = input2_data.reshape(SPAN, LISTSIZE, GRID2, GRID2)

    input_data = list()
    input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

    return input_data


def organising_post_data(boxes, classes, scores, NMS_THRESH):
    output = [torch.zeros(0, 6)] * len(boxes)
    for x in range(len(boxes)):
        c = classes[x] * 4096  # classes
        boxes, scores = boxes[x] + c, scores[x]  # boxes (offset by class), scores
        i = torch.ops.torchvision.nms(boxes, scores, NMS_THRESH)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[x] = i

    return output


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def process(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2])
    box_wh = np.exp(input[..., 2:4])
    box_wh = box_wh * anchors

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)

    box_xy += grid
    box_xy /= (grid_w, grid_h)
    box_wh /= (416, 416)
    box_xy -= (box_wh / 2.)
    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs

def filter_boxes(boxes, box_confidences, box_class_probs, OBJ_THRESH):

    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= OBJ_THRESH)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores, NMS_THRESH):

    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov4_post_process(data, OBJ_THRESH=0.1, NMS_THRESH=0.6):

    input_data = organising_pre_data(data)

    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55],
            [72, 146], [142, 110], [192, 243], [459, 401]]

    boxes, classes, scores = [], [], []
    for input,mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s, OBJ_THRESH)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s, NMS_THRESH)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    output = organising_post_data(boxes, classes, scores, NMS_THRESH)

    return output

def draw(image, boxes, scores, classes):

    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(x, y, x+w, y+h))
        x *= image.shape[1]
        y *= image.shape[0]
        w *= image.shape[1]
        h *= image.shape[0]
        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)


if __name__ == '__main__':

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

    yolov4 = KSNN('VIM3')
    print(' |---+ KSNN Version: {} +---| '.format(yolov4.get_nn_version()))

    print('Start init neural network ...')
    yolov4.nn_init(library=library, model=model, level=level)
    print('Done.')

    print('Get input data ...')
    cv_img =  list()
    img = cv.imread(picture, cv.IMREAD_COLOR)
    img_resized = cv.resize(img, (640, 640))
    cv_img.append(img)
    print('Done.')

    print('Start inference ...')
    start = time.time()

    '''
        default input_tensor is 1
    '''
    data = np.array([yolov4.nn_inference(cv_img, platform='DARKNET', reorder='2 1 0', output_tensor=3, output_format=output_format.OUT_FORMAT_FLOAT32)])
    end = time.time()
    print('Done. inference time: ', end - start)

    output = yolov4_post_process(data)

    if boxes is not None:
        draw(img, output[0], output[1], output[2])

    # for ind in range(len(boxes)):
    #     print(f"boxes={boxes[ind]}, classes={classes[ind]}, scores={scores[ind]}\n")

    cv.imwrite("results/results.jpg", img)
    cv.waitKey(0)
