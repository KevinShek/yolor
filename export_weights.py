import argparse

import torch

from torch.utils.mobile_optimizer import optimize_for_mobile
from utils.google_utils import attempt_download
from utils.torch_utils import select_device
from models.models import *
import models
from mish_cuda import MishCuda
from utils.activations import Hardswish, Mish


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    # color arguments, string
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def file_size(file):
    # Return file size in MB
    return Path(file).stat().st_size / 1e6


def export_torchscript(model, img, file, optimize=True):
    # TorchScript model export
    prefix = colorstr('TorchScript:')
    try:
        print(f'\n{prefix} starting export with torch {torch.__version__}...')
        f = file.with_suffix('.torchscript.pt')
        ts = torch.jit.trace(model, img, strict=False)
        (optimize_for_mobile(ts) if optimize else ts).save(f)
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return ts
    except Exception as e:
        print(f'{prefix} export failure: {e}')


def export_onnx(model, img, file, opset=12, train=False, dynamic=True, simplify=True):
    # ONNX model export
    prefix = colorstr('ONNX:')
    try:
        import onnx
        from onnx import shape_inference
        import onnxruntime as ort
        import onnx_graphsurgeon as gs

        print(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        f = file.with_suffix('.onnx')
        # torch.onnx.export(model, img, f, verbose=False, opset_version=opset, input_names=['images'],
        #                   output_names=['classes', 'boxes'] if y is None else ['output'])
        #                   # output_names=['output'])
        # torch.onnx.export(model, img, f, verbose=False, opset_version=opset,
        #                   training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
        #                   do_constant_folding=not train,
        #                   input_names=['images'],
        #                   output_names=['classes', 'boxes'] if y is None else ['output'],
        #                   dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
        #                                 'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
        #                                 } if dynamic else None)

        torch.onnx.export(model,               # model being run
        img,                         # model input (or a tuple for multiple inputs)
        f,   # where to save the model (can be a file or file-like object)
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=opset,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
        dynamic_axes={'input' : {0 : 'batch_size', 2: 'height', 3:'width'},    # variable length axes
                    'output' : {0 : 'batch_size', 1: 'n_boxes'}})

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # print(onnx.helper.printable_graph(model_onnx.graph))  # print

        # print('Remove unused outputs')
        # onnx_module = shape_inference.infer_shapes(onnx.load(f))
        # while len(onnx_module.graph.output) != 1:
        #     for output in onnx_module.graph.output:
        #         if output.name != 'output':
        #             print('--> remove', output.name)
        #             onnx_module.graph.output.remove(output)
        # graph = gs.import_onnx(onnx_module)
        # graph.cleanup()
        # graph.toposort()
        # graph.fold_constants().cleanup()
        # onnx.save_model(gs.export_onnx(graph), f)
        # print('Convert successfull !')

        # # Simplify
        # if simplify:
        #     try:
        #         import onnxsim

        #         print(
        #             f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
        #         model_onnx, check = onnxsim.simplify(
        #             model_onnx,
        #             dynamic_input_shape=dynamic,
        #             input_shapes={'images': list(img.shape)} if dynamic else None)
        #         assert check, 'assert check failed'
        #         onnx.save(model_onnx, f)
        #     except Exception as e:
        #         print(f'{prefix} simplifier failure: {e}')
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        print(
            f"{prefix} run --dynamic ONNX model inference with: 'python detect.py --weights {f}'")
        return f
    except Exception as e:
        print(f'{prefix} export failure: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov4.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--cfg', type=str, default='cfg/yolov4.cfg', help='*.cfg path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--include', nargs='+', default=['torchscript', 'onnx'],
                        help='available formats are (torchscript, onnx, tflite)')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    file = Path(opt.weights)

    #load device
    device = select_device(opt.device, batch_size=opt.batch_size)
    set_logging()
    t = time.time()
    include = [x.lower() for x in opt.include]

    # Input
    img = torch.zeros((opt.batch_size, 3, *opt.img_size))  # image size(1,3,320,192) iDetection
    if torch.cuda.is_available():
        img = img.to(device)


    ## Load PyTorch model
    try:
        model = attempt_load(opt.weights, map_location=torch.device(device))  # load FP32 model
        labels = model.names
        model.eval()
        model = model.to(device)
        print("pytorch model loaded")
    except:    
        #attempt_download(opt.weights)

        # # Load model
        model = Darknet(opt.cfg).to(device)
        # load_darknet_weights(model, opt.weights)
        # # model.model[-1].export = True  # set Detect() layer export=True
        # # y = model(img)  # dry run

        # load model for weights format
        try:
            ckpt = torch.load(opt.weights, map_location=device)  # load checkpoint
            ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt['model'], strict=False)
            print("pytorch checkpoint model loaded")
        except:
            load_darknet_weights(model, opt.weights)
            print("weights model loaded")
        model.eval()
    
    # model.model[-1].export = True  # set Detect() layer export=True
    y = model(img)  # dry run

    f = None

    # TorchScript export
    if 'torchscript' in include:
        export_torchscript(model, img, file)
    #     print('\nStarting TorchScript export with torch %s...' % torch.__version__)
    #     f = opt.weights.replace('.pt', '.torchscript.pt')  # filename
    #     ts = torch.jit.trace(model, img)
    #     ts.save(f)
    #     print('TorchScript export success, saved as %s' % f)
    # except Exception as e:
    #     print('TorchScript export failure: %s' % e)

    # ONNX export
    if 'onnx' in include:
        export_onnx(model, img, file)
    #     import onnx

    #     print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    #     f = opt.weights.replace('.pt', '.onnx')  # filename
    #     model.fuse()  # only for ONNX
    #     torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
    #                       output_names=['classes', 'boxes'] if y is None else ['output'])

    #     # Checks
    #     onnx_model = onnx.load(f)  # load onnx model
    #     onnx.checker.check_model(onnx_model)  # check onnx model
    #     print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
    #     print('ONNX export success, saved as %s' % f)

    # except Exception as e:
    #     print('ONNX export failure: %s' % e)

    if 'tflite' in include:
        if f is None:
            f = str(export_onnx(model, img, file))

        import onnx
        from onnx_tf.backend import prepare


        onnx_path = "/projects" + f.strip("..")
        tensorflow_location = "/projects/weights/tensorflow/"+ file.stem
        tflite_model_path = "/projects/weights/tensorflow/" + file.stem +".tflite"

        # # print(onnx_path)

        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(tensorflow_location) # exporting tensorflow model

        import tensorflow as tf

        # Convert the model
        # https://github.com/hunglc007/tensorflow-yolov4-tflite/blob/9f16748aa3f45ff240608da4bd9b1216a29127f5/convert_tflite.py#L17
        converter = tf.lite.TFLiteConverter.from_saved_model(tensorflow_location)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.allow_custom_ops = True
        # converter.target_spec.supported_ops = [
        # tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        # tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        # ]

        tflite_model = converter.convert()

        # Save the model
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)


        # Load the TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()

        # Get input and output tensors
        input_details = interpreter.get_input_details()

        print(input_details)
        interpreter.resize_tensor_input(input_details[0]['index'], (1, 3, opt.img_size[0], opt.img_size[1]))
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # input and output shape 
        print(input_details)
        print("input=",input_details[0]['shape'])
        print("output=",output_details[0]['shape'])

        import numpy as np


        # Test model on random input data
        interpreter.allocate_tensors()
        input_shape = input_details[0]['shape']
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data)

    # # CoreML export
    # try:
    #     import coremltools as ct

    #     print('\nStarting CoreML export with coremltools %s...' % ct.__version__)
    #     # convert model from torchscript and apply pixel scaling as per detect.py
    #     model = ct.convert(ts, inputs=[ct.ImageType(name='images', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
    #     f = opt.weights.replace('.pt', '.mlmodel')  # filename
    #     model.save(f)
    #     print('CoreML export success, saved as %s' % f)
    # except Exception as e:
    #     print('CoreML export failure: %s' % e)

    # Finish
    print('\nExport complete. Visualize with https://github.com/lutzroeder/netron.')
