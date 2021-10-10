import torch
from pathlib import Path
from utils.general import check_suffix
import cv2
from utils.datasets import letterbox
import numpy as np
from utils.torch_utils import select_device

# checking the weights
w = "weights/"
classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
check_suffix(w, suffixes)  # check weights have acceptable suffix
pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans

device = select_device("0")

# load image
img0 = cv2.imread("inference/images/horses.jpg")
imgsz = 1280

# Padded resize
img = letterbox(img0, new_shape=imgsz, auto_size=64)[0]

# Convert
# img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
img = np.ascontiguousarray(img)

print(img.shape)
if onnx:
    img.astype('float32')
else:
    img = torch.from_numpy(img).to(device)
img = img / 255.0  # 0 - 255 to 0.0 - 1.0
if len(img.shape) == 3:
    img = img[None]  # expand for batch dim

if onnx:
    import onnx
    import onnxruntime as ort
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

    ort_session = ort.InferenceSession(w,None)

    pred = torch.tensor(ort_session.run([ort_session.get_outputs()[0].name],{ort_session.get_inputs()[0].name: img}))
    print("loaded onnx model")
else:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
    import tensorflow as tf
    imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
    if pb:
        def wrap_frozen_graph(gd, inputs, outputs):
            x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
            return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                            tf.nest.map_structure(x.graph.as_graph_element, outputs))

        graph_def = tf.Graph().as_graph_def()
        graph_def.ParseFromString(open(w, 'rb').read())
        frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")

        pred = frozen_func(x=tf.constant(imn)).numpy()

    if saved_model:
        #model = tf.saved_model.load(w)
        model = tf.keras.models.load_model(w)
        model.trainable = False

        pred = model(imn, training=False).numpy()
        print("loaded saved model")

pred[..., 0] *= imgsz[1]  # x
pred[..., 1] *= imgsz[0]  # y
pred[..., 2] *= imgsz[1]  # w
pred[..., 3] *= imgsz[0]  # h
pred = torch.tensor(pred)

print(pred)