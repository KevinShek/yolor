from torch.utils.mobile_optimizer import optimize_for_mobile
from pathlib import Path
from utils.torch_utils import select_device
from utils.general import set_logging, check_img_size
from utils.activations import Hardswish, Mish
from models.experimental import attempt_load
import models
import torch.nn as nn
import torch
import argparse
import sys
import time
from onnx import shape_inference
from models.models import Darknet, load_darknet_weights

# from mish_cuda import MishCuda

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

# convert SyncBatchNorm to BatchNorm2d
def convert_sync_batchnorm_to_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features,
                                             module.eps, module.momentum,
                                             module.affine,
                                             module.track_running_stats)

        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, convert_sync_batchnorm_to_batchnorm(child))
    del module
    return module_output


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


def export_torchscript(model, im, file, optimize, prefix=colorstr('TorchScript:')):
    # YOLOv5 TorchScript model export
    import json  # required to save graph props
    try:
        print(f'\n{prefix} starting export with torch {torch.__version__}...')
        f = file.with_suffix('.torchscript.pt')
        
        h, w = im.shape[-2:]
        batch_size = im.shape[0]
        stride = int(max(model.stride))
        dev_type = next(model.parameters()).device.type
        dct_params = {"HW": [h, w], "BATCH": batch_size, "STRIDE": stride, "DEVICE": dev_type}
        str_config = json.dumps(dct_params)
        extra_files = {'config.txt': str_config}  # torch._C.ExtraFilesMap()

        ts = torch.jit.trace(model, im, strict=False)
        (optimize_for_mobile(ts) if optimize else ts).save(f, _extra_files=extra_files)

        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    except Exception as e:
        print(f'{prefix} export failure: {e}')


def export_onnx(model, img, weights, opset=13, train=False, dynamic=False, simplify=True, remove_unwanted_output=False):
    # ONNX model export
    prefix = colorstr('ONNX:')
    try:
        import onnx
        import onnxruntime as ort
        import onnx_graphsurgeon as gs

        print(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        if dynamic:
            dyn = "dynamic"
        else:
            dyn = "static"
        if simplify:
            sim = "sim"
        else:
            sim = "_"
        f = opt.weights.replace('.pt', f'-{opt.img_size[0]}-{opt.img_size[1]}_opset_{opset}_{dyn}_{sim}.onnx')  # filename

        # torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
        #                   output_names=['classes', 'boxes'] if y is None else ['output'])
                          # output_names=['output'])
        # torch.onnx.export(model, img, f, verbose=False, opset_version=opset,
        #                   training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
        #                   do_constant_folding=not train,
        #                   input_names=['images'],
        #                   output_names=['output'],
        #                   dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
        #                                 'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
        #                                 } if dynamic else None)

        #     # Export the model
        # torch.onnx.export(model,               # model being run
        #               img,                         # model input (or a tuple for multiple inputs)
        #               f,   # where to save the model (can be a file or file-like object)
        #               export_params=True,        # store the trained parameter weights inside the model file
        #               opset_version=opset,          # the ONNX version to export the model to
        #               do_constant_folding=True,  # whether to execute constant folding for optimization
        #               input_names = ['input'],   # the model's input names
        #               output_names = ['output'], # the model's output names
        #               dynamic_axes={'input' : {0 : 'batch_size', 2: 'height', 3:'width'},    # variable length axes
        #                             'output' : {0 : 'batch_size', 1: 'n_boxes'}})

            # Export the model
        torch.onnx.export(model,               # model being run
                      img,                         # model input (or a tuple for multiple inputs)
                      f,   # where to save the model (can be a file or file-like object)
                      verbose = False,
                      opset_version=opset,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size', 2: 'height', 3:'width'},    # variable length axes
                                    'output' : {0 : 'batch_size', 1: 'n_boxes'}} if dynamic else None)


        # # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # print(onnx.helper.printable_graph(model_onnx.graph))  # print

        unwant_output = []

        # print('Remove unused outputs')
        onnx_module = shape_inference.infer_shapes(onnx.load(f))
        # while len(onnx_module.graph.output) != 1:
        for output in onnx_module.graph.output:
            if output.name != 'output':
                    unwant_output.append(str(output.name))
        #             print('--> remove', output.name)
        #             onnx_module.graph.output.remove(output)
        # graph = gs.import_onnx(onnx_module)
        # graph.cleanup()
        # graph.toposort()
        # graph.fold_constants().cleanup()
        # onnx.save_model(gs.export_onnx(graph), f)
        # print('Convert successfull !')

        # Simplify
        if simplify:
            try:
                from onnxsim import simplify

                # onnx_model, check = simplify(model_onnx, check_n=3, input_shape= {'input': [1, 3, 640, 640]})
                # static
                # onnx_model, check = simplify(model_onnx, overwrite_input_shapes= {'input': [1, 3, opt.img_size[0], opt.img_size[1]]}, unused_output=[str(665), str(752), str(839)])
                if remove_unwanted_output:
                    onnx_model, check = simplify(model_onnx, overwrite_input_shapes= {'input': [1, 3, opt.img_size[0], opt.img_size[1]]}, unused_output=unwant_output)
                else:
                    onnx_model, check = simplify(model_onnx, overwrite_input_shapes= {'input': [1, 3, opt.img_size[0], opt.img_size[1]]})

                assert check, 'assert simplify check failed'
                onnx.save(onnx_model, f)
            except Exception as e:
                print(f'{prefix} simplifier failure: {e}')

            session = ort.InferenceSession(f, providers=["CPUExecutionProvider"])

            for ii in session.get_inputs():
                print("input: ", ii)

            for oo in session.get_outputs():
                print("output: ", oo)

        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        print(f"{prefix} run --dynamic ONNX model inference with: 'python detect.py --weights {f}'")
        return f
    except Exception as e:
        print(f'{prefix} export failure: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='./yolor-p6.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int,
                        default=[1280, 1280], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--include', nargs='+', default=['torchscript', 'onnx'],
                        help='available formats are (torchscript, onnx, tflite)')
    parser.add_argument('--cfg', default='', help='')
    parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')                    
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    file = Path(opt.weights)

    # selecting devices
    device = select_device(opt.device, batch_size=opt.batch_size)
    print(opt)
    set_logging()
    t = time.time()
    include = [x.lower() for x in opt.include]

    if file.suffix == ".pt" or file.suffix == ".weights":
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
        # convert sync batchnorm
        # model = convert_sync_batchnorm_to_batchnorm(model)

        # Checks
        # gs = int(max(model.stride))  # grid size (max stride)
        # verify img_size are gs-multiples
        # opt.img_size = [check_img_size(x, gs) for x in opt.img_size]

        # Input
        # image size(1,3,320,192) iDetection
        img = torch.zeros((opt.batch_size, 3, *opt.img_size), device=device)
        # if torch.cuda.is_available() and not device == "cpu":
        #     img = img.to(device)

        # Update model
        for k, m in model.named_modules():
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
            # if isinstance(m, models.common.Conv) and isinstance(m.act, MishCuda):
            #     m.act = Mish()  # assign activation
            if isinstance(m, models.common.Conv) and isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()  # assign activation
            if isinstance(m, models.common.BottleneckCSP) or isinstance(m, models.common.BottleneckCSP2) \
                    or isinstance(m, models.common.SPPCSP):
                if isinstance(m.bn, nn.SyncBatchNorm):
                    bn = nn.BatchNorm2d(m.bn.num_features, eps=m.bn.eps, momentum=m.bn.momentum)
                    bn.training = False
                    bn._buffers = m.bn._buffers
                    bn._non_persistent_buffers_set = set()
                    m.bn = bn
                # if isinstance(m.act, MishCuda):
                #     m.act = Mish()  # assign activation
        #     # if isinstance(m, models.yolo.Detect):
        #     #     m.forward = m.forward_export  # assign forward (optional)

        # comment out the model.model[-1].export = True to export the yolov4 csp to being a onnx file

        # model.model[-1].export = True  # set Detect() layer export=True
        y = model(img)  # dry run

        f = None
    elif file.suffix == ".onnx":
        f = str(file)

    # Exporting
    if 'torchscript' in include:
        export_torchscript(model, img, file, opt.optimize)
    if 'onnx' in include:
        f = export_onnx(model, img, opt.weights)
    # TensorFlow Exports
    if 'tflite' in include:
        if f is None:
            f = export_onnx(model, img, file)
        
        import onnx
        from onnx_tf.backend import prepare


        onnx_path = "/projects" + f.strip("..")
        tensorflow_location = "/projects/weights/tensorflow/"+ file.stem +"/"
        tflite_model_path = "/projects/weights/tensorflow/" + file.stem +".tflite"

        # # print(onnx_path)

        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(tensorflow_location) # exporting tensorflow model

        import tensorflow as tf

        # Convert the model
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=tensorflow_location)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.allow_custom_ops = True
        # # converter.experimental_new_converter = True
        # converter.target_spec.supported_ops = [
        # tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        # tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        # ]

        tflite_model = converter.convert()

        # Save the model
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)

        import tflite_runtime.interpreter as tflite

        # Load the TFLite model and allocate tensors
        interpreter = tflite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()

        # Get input and output tensors
        input_details = interpreter.get_input_details()
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





    # # TorchScript export
    # try:
    #     export_torchscript(model, img, file)

    #     # print('\nStarting TorchScript export with torch %s...' % torch.__version__)
    #     # f = opt.weights.replace('.pt', '.torchscript.pt')  # filename
    #     # ts = torch.jit.trace(model, img)
    #     # ts.save(f)
    #     # print('TorchScript export success, saved as %s' % f)
    # except Exception as e:
    #     print('TorchScript export failure: %s' % e)

    # ONNX export
    # try:
    #     export_onnx(model, img, file)
    #     import onnx

    #     print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    #     f = opt.weights.replace('.pt', '.onnx')  # filename
    #     torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
    #                       output_names=['classes', 'boxes'] if y is None else ['output'])

    #     # Checks
    #     onnx_model = onnx.load(f)  # load onnx model
    #     onnx.checker.check_model(onnx_model)  # check onnx model
    #     # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
    #     print('ONNX export success, saved as %s' % f)
    # except Exception as e:
    #     print('ONNX export failure: %s' % e)

    # # CoreML export
    # try:
    #     import coremltools as ct

    #     print('\nStarting CoreML export with coremltools %s...' % ct.__version__)
    #     # convert model from torchscript and apply pixel scaling as per detect.py
    #     model = ct.convert(ts, inputs=[ct.ImageType(
    #         name='image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
    #     f = opt.weights.replace('.pt', '.mlmodel')  # filename
    #     model.save(f)
    #     print('CoreML export success, saved as %s' % f)
    # except Exception as e:
        # print('CoreML export failure: %s' % e)

    # Finish
    print(f'\nExport complete ({time.time() - t:.2f}s)'
          f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
          f'\nVisualize with https://netron.app')
