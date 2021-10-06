import onnx

w = "weights/yolor-p6.onnx"

# Load the ONNX model
model = onnx.load(w)

# Check that the IR is well formed
onnx.checker.check_model(model)

# # Print a Human readable representation of the graph
# print(onnx.helper.printable_graph(model.graph))

output =[node.name for node in model.graph.output]

input_all = [node.name for node in model.graph.input]
input_initializer =  [node.name for node in model.graph.initializer]
net_feed_input = list(set(input_all)  - set(input_initializer))

print('Inputs: ', net_feed_input)
print('Outputs: ', output)

import onnxruntime as ort
import cv2
from utils.datasets import letterbox
import numpy as np

img0 = cv2.imread("inference/images/horses.jpg").astype('float32')

# Padded resize
img = letterbox(img0, new_shape=1280, auto_size=64)[0]

# Convert
# img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
img = np.ascontiguousarray(img)

print(img.shape)
img = img / 255.0  # 0 - 255 to 0.0 - 1.0
if len(img.shape) == 3:
    img = img[None]  # expand for batch dim
ort_session = ort.InferenceSession(w,None)

outputs = ort_session.run(
    [ort_session.get_outputs()[0].name],
    {ort_session.get_inputs()[0].name: img}
)
print(outputs)