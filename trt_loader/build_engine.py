import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import cv2
from utils.datasets import letterbox
from utils.general import non_max_suppression
from utils.plots import plot_one_box
import torch
import time
import os 
import common

def detect(self, bgr_img):   
    # Prediction
    ## Padded resize
    inp = letterbox(bgr_img, new_shape=self.imgsz, auto_size=64)[0]
    inp = inp[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    inp = inp.astype('float32') / 255.0  # 0 - 255 to 0.0 - 1.0
    inp = np.expand_dims(inp, 0)
    print(inp.shape)        

    ## Inference
    t1 = time.time()
    pred = self.model.run(inp)[0]
    t2 = time.time()
    ## Apply NMS
    with torch.no_grad():
        pred = non_max_suppression(torch.tensor(pred), conf_thres=0.5, iou_thres=0.6)
    t3 = time.time()
    print('Inference: {}'.format(t2-t1))
    print('NMS: {}'.format(t3-t2))
    print('FPS: ', 1/(t3-t1))

    # Process detections
    visualize_img = bgr_img.copy()
    det = pred[0]  # detections per image
    if det is not None and len(det):
        # Rescale boxes from img_size to im0 size
        _, _, height, width = inp.shape
        h, w, _ = bgr_img.shape
        det[:, 0] *= w/width
        det[:, 1] *= h/height
        det[:, 2] *= w/width
        det[:, 3] *= h/height
        for x1, y1, x2, y2, conf, cls in det:       # x1, y1, x2, y2 in pixel format
            label = '%s %.2f' % (self.names[int(cls)], conf)
            plot_one_box((x1, y1, x2, y2), visualize_img, label=label, color=self.colors[int(cls)], line_thickness=3)

    cv2.imwrite('result.jpg', visualize_img)
    return visualize_img

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()

# https://github.com/NVIDIA/TensorRT/blob/96e23978cd6e4a8fe869696d3d8ec2b47120629b/samples/python/yolov3_onnx/onnx_to_tensorrt.py#L65
# https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/
# https://github.com/NVIDIA-AI-IOT/torch2trt

# def build_engine(onnx_file_path):
#     # initialize TensorRT engine and parse ONNX model
#     builder = trt.Builder(TRT_LOGGER)
#     network = builder.create_network()
#     config = builder.create_builder_config()
#     parser = trt.OnnxParser(network, TRT_LOGGER)
#     runtime = trt.Runtime(TRT_LOGGER)
    
#     # parse ONNX
#     with open(onnx_file_path, 'rb') as model:
#         print('Beginning ONNX file parsing')
#         parser.parse(model.read())
#     print('Completed parsing of ONNX file')

#     # allow TensorRT to use up to 1GB of GPU memory for tactic selection
#     config.max_workspace_size = 1 << 30
#     # we have only one image in batch
#     builder.max_batch_size = 1
#     # use FP16 mode if possible
#     fp16_mode = True
#     if fp16_mode:
#         config.set_flag(trt.BuilderFlag.FP16)
    
#     # generate TensorRT engine optimized for the target platform
#     print('Building an engine...')
#     plan = builder.build_serialized_network(network, config)
#     engine = runtime.deserialize_cuda_engine(plan)
#     context = engine.create_execution_context()
#     print("Completed creating Engine")

#     return engine, context

def build_engine(onnx_file_path, engine_file_path):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
        config.max_workspace_size = 1 << 28 # 256MiB
        builder.max_batch_size = 1
        # Parse model file
        if not os.path.exists(onnx_file_path):
            print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
            exit(0)
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
        network.add_input(name="input", dtype= trt.float32, shape=(1, 3, 640, 640))
        # network.mark_output(network.get_layer(network.num_layers - 1).get_output(0))
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        plan = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(plan)
        print("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            f.write(plan)
        return engine

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def main():
    # initialize TensorRT engine and parse ONNX model
    engine = build_engine("../weights/yolov4-csp-640-hedrial-640-640.onnx", "test_enigine.trt")
    context = engine.create_execution_context()

    # get sizes of input and output and allocate memory required for input data and for output data
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
            device_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)
    
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()

    img = cv2.imread('../datasets/GettyImages_1214076743.0.jpg')

    host_input = np.array(img.numpy(), dtype=np.float32, order='C')
    cuda.memcpy_htod_async(device_input, host_input, stream)

    # run inference
    context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()

    output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[0])

    print(output_data)

if __name__ == '__main__':
    main()