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


def export_torchscript(model, img, file, optimize=False):
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


def export_onnx(model, img, file, opset=13, train=False, dynamic=True, simplify=True):
    # ONNX model export
    prefix = colorstr('ONNX:')
    try:
        import onnx

        print(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        f = file.with_suffix('.onnx')
        # torch.onnx.export(model, img, f, verbose=False, opset_version=opset, input_names=['images'],
        #                   output_names=['classes', 'boxes'] if y is None else ['output'])
        #                   # output_names=['output'])
        torch.onnx.export(model, img, f, verbose=False, opset_version=opset,
                          training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=not train,
                          input_names=['images'],
                          output_names=['classes', 'boxes'] if y is None else ['output'],
                          dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                        'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
                                        } if dynamic else None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # print(onnx.helper.printable_graph(model_onnx.graph))  # print

        # Simplify
        if simplify:
            try:
                import onnxsim

                print(
                    f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic,
                    input_shapes={'images': list(img.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                print(f'{prefix} simplifier failure: {e}')
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        print(
            f"{prefix} run --dynamic ONNX model inference with: 'python detect.py --weights {f}'")
    except Exception as e:
        print(f'{prefix} export failure: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov4.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--cfg', type=str, default='cfg/yolov4.cfg', help='*.cfg path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    file = Path(opt.weights)

    #load device
    device = select_device(opt.device, batch_size=opt.batch_size)

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

    # TorchScript export
    try:
        export_torchscript(model, img, file)
    #     print('\nStarting TorchScript export with torch %s...' % torch.__version__)
    #     f = opt.weights.replace('.pt', '.torchscript.pt')  # filename
    #     ts = torch.jit.trace(model, img)
    #     ts.save(f)
    #     print('TorchScript export success, saved as %s' % f)
    except Exception as e:
        print('TorchScript export failure: %s' % e)

    # ONNX export
    try:
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
    except Exception as e:
        print('ONNX export failure: %s' % e)

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
