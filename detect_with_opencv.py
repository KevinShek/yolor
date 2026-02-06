import os
import sys
import argparse
import math
from typing import Tuple
import time
import torch

import cv2
import numpy as np
from utils.general import non_max_suppression
from utils.datasets import LoadStreams, LoadImages
from utils.torch_utils import select_device, load_classifier, time_sync

parser = argparse.ArgumentParser(description='A demo for running libfacedetection using OpenCV\'s DNN module.')
# OpenCV DNN
# Location
parser.add_argument('--image', help='Path to the image.')
parser.add_argument('--model', type=str, help='Path to .onnx model file.')
parser.add_argument('--classes', type=str, help='Path to .txt file for classes names.')
# Inference
# parser.add_argument('--conf_thresh', default=0.6, type=float, help='Threshold for filtering out faces with conf < conf_thresh.')
# parser.add_argument('--nms_thresh', default=0.3, type=float, help='Threshold for non-max suppression.')
# parser.add_argument('--keep_top_k', default=750, type=int, help='Keep keep_top_k for results outputing.')
# Result
parser.add_argument('--save', default='result.jpg', type=str, help='Path to save the result image.')
args = parser.parse_args()
img_size = 640

# Build the blob
assert os.path.exists(args.image), 'File {} does not exist!'.format(args.image)
dataset = LoadImages(args.image, img_size=[img_size,img_size], auto_size=64, auto=False)
t0, t1 = 0., 0.

  
# settings
conf_thres = 0.1
iou_thres = 0.45
max_det = 300
agnostic_nms = False
classes = None
t0, t1 = 0., 0.


# print(blob.shape)
# Load the net
net = cv2.dnn.readNet(args.model)

# NPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_TIMVX)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_NPU)

# CPU
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Run the net
# output_names = ["ignore", "ignore", "ignore", "output"]
output_names = net.getUnconnectedOutLayersNames()

#list_of_paths = ["/home/khadas/projects/datasets/heridal/testImages/images/test_BLI_0004.JPG"]#, "/home/khadas/projects/datasets/heridal/testImages/images/test_BLI_0002.JPG", 
#"/home/khadas/projects/datasets/heridal/testImages/images/test_BLI_0003.JPG", "/home/khadas/projects/datasets/heridal/testImages/images/test_BLI_0004.JPG",
#"/home/khadas/projects/datasets/heridal/testImages/images/test_BLI_0005.JPG", "/home/khadas/projects/datasets/heridal/testImages/images/test_BLI_0006.JPG" ]


for path, img, im0s, vid_cap in dataset:
#for path in list_of_paths:
  img = cv2.imread(path, cv2.IMREAD_COLOR)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  h, w, _ = img.shape
  print('Image size: h={}, w={}'.format(h, w))
  resize_img = cv2.resize(img, (img_size, img_size))
  # blob = cv2.dnn.blobFromImage(img, size=(640,640) , swapRB=True, crop=False) # 'size' param resize the output to the given shape
  blob = cv2.dnn.blobFromImage(resize_img, 1/255.0)
  
  #print(output_names)
  start = time.time()
  t = time_sync()
  net.setInput(blob)
  
  outputs = net.forward(output_names)
  #print(outputs[0][0][:20])
  
  t0 += time_sync() - t
  
  #print(np.array(outputs).shape)
  pred = torch.from_numpy(np.array(outputs))
  #print(pred.shape)
  
  t = time_sync()
  output = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms)
  end = time.time()
  t1 += time_sync() - t
  # print(output)
  #print(output[0].shape)
  print(f'Done. (inference={t0:.3f}s, NMS={t1:.3}s, total speed={t1 + t0:.3f}s)')
  
  # drawing the predictions
  def draw_predictions (class_id, score, left, top, right, bottom):
    cv2.rectangle(resize_img, (left, top), (right, bottom), (0, 255, 0))
    
    label = "%.2f" % score
    
    # print a label of class
    if classes:
      assert(class_id < len(classes))
      label = "%s: %s" % (classes[class_id], label)
      
    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, label_size[1])
    cv2.rectangle(resize_img, (left, top - label_size[1]), (left + label_size[0], top + base_line), (255, 255, 255), cv2.FILLED)
    cv2.putText(resize_img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    
  for i in range(len(output[0])):
    predition = output[0][i]
    box = output[0][i][:4]
    score = output[0][i][4]
    cls = output[0][i][5]
    draw_predictions(cls, score, int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    line = f"box = {box}, score = {score}, class = {cls}"
  
  #with open('result.txt', 'a') as f:
    #f.write(('%s ') % line + '\n')
    
#resize_img = cv2.cvtColor(resize_img, cv2.COLOR_RGB2BGR)
  
#cv2.imwrite("result.png", resize_img)


# opencv method
'''
outputs = np.array(outputs)
print(outputs.shape) # 1xNx6 N being the number of detection and 6 being x,y,w,h,cls,confidence
print(outputs[0][0])

boxes, class_ids, scores = [], [], []

conf_threshold = 0.5
nms_threshold = 0.1

# parsing the output
for output in outputs[0][0]:
  box = output[:4]
  cls = output[4]
  score = output[5]
  
  if score > conf_threshold:
    # box = [center_x, center_y, width, height]
    x1 = int(box[0] - box[2] / 2)
    y1 = int(box[1] - box[3] / 2)
    #left = int(box[0])
    #top = int(box[1])
    #right = int(box[2])
    #bottom = int(box[3])
    #width = left - right + 1
    #height = top - bottom + 1
    x2 = int(box[0] + box[2] / 2)
    y2 = int(box[3] + box[3] / 2)

    boxes.append([x1, y1, x2, y2])
    if int(cls) == 0:
      cls = 1
    class_ids.append(int(cls) - 1)
    scores.append(float(score))

print(len(scores))
print(scores[:5])

# performing NMS
indices = []
class_ids = np.array(class_ids)
boxes = np.array(boxes)
scores = np.array(scores)
unique_classes = set(class_ids)
# print(unique_classes)

for cl in unique_classes:
  class_indices = np.where(class_ids == cl)[0]
  # print(class_indices)
  score = scores[class_indices]
  box = boxes[class_indices].tolist()
  nms_indices = cv2.dnn.NMSBoxes(box, score, conf_threshold, nms_threshold)
  nms_indices = nms_indices[:] if len(nms_indices) else []
  indices.extend(class_indices[nms_indices])
  
print(len(indices))

if args.classes:
    with open(args.classes, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

for i in indices[:10]:
  box = boxes[i]
  left = box[0]
  top = box[1]
  width = box[2]
  height = box[3]
  draw_predictions(class_ids[i], scores[i], left, top, width, height)
  line = f"box = {box}, score = {scores[i]}, class = {classes[class_ids[i]]}"
  
  with open('result.txt', 'a') as f:
    f.write(('%s ') % line + '\n')
  
cv2.imwrite("result.png", resize_img)

'''

t = tuple(x / len(dataset) * 1E3 for x in (t0, t1, t0 + t1)) + (img_size, img_size, 1)  # tuple
text = f'Speed: {t[0]:.3f}/{t[1]:.3f}/{t[2]:.3f} ms inference/NMS/total per {t[3]}x{t[4]} image at batch-size {t[5]}'
print(text)

with open(f'information_{img_size}x{img_size}_CPU_files.txt', 'a') as f:
    f.write(f"{text}" + '\n\n')
