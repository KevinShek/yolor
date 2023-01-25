# YOLOv5 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
from pathlib import Path

# This call to matplotlib.use() has no effect because the backend has already
# been chosen; matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
# or matplotlib.backends is imported for the first time.
import matplotlib
matplotlib.use('Agg')  # for writing to files only


import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, check_requirements, check_suffix, colorstr, is_ascii, \
    non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, \
    save_one_box
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, load_classifier, time_sync


@torch.no_grad()
def run(weights='yolov4.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=300,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        library=None, # for khadas
        auto=True, # auto is for dynamic models but for static models turn this "False"
        opencv_onnx=False 
        ):
    if not auto:
        auto = False
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    # device = select_device(device)

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '.trt', '.nb', '', '.weights']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, trt, khadas, saved_model, darknet  = (suffix == x for x in suffixes)  # backend booleans
    if opencv_onnx:
        onnx = False
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    pt_jit = pt and 'torchscript' in w
    list_of_images = []
    if khadas:
        opt.device = "cpu"
    if not opencv_onnx:
      device = select_device(opt.device, batch_size=1)
    if not khadas and not opencv_onnx:
        half &= device.type != 'cpu'  # half precision only supported on CUDA
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        # session = onnxruntime.InferenceSession(w, None)
        session = onnxruntime.InferenceSession(w)
        names = names
    elif opencv_onnx:
        # Load the model
        model = cv2.dnn.readNet(w)
        # Setting what processor to use the model
        
        if opt.device == "cpu":
            # NPU
            model.setPreferableBackend(cv2.dnn.DNN_BACKEND_TIMVX)
            model.setPreferableTarget(cv2.dnn.DNN_TARGET_NPU)
        else:
            # CPU
            model.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


        # grabbing the output names
        output_names = model.getUnconnectedOutLayersNames()
    elif trt:
        from trt_loader.trt_loader import TrtModel
        model = TrtModel(w, imgsz, total_classes=len(load_classes(opt.names)))
    elif khadas:
        from ksnn.api import KSNN
        level = 0
        yolo = KSNN('VIM3')
        print(' |---+ KSNN Version: {} +---| '.format(yolo.get_nn_version()))
        print('Start init neural network ...')
        yolo.nn_init(library=library, model=w, level=level)
        print('Done.')
    else:  # TensorFlow models
        try:
            #check_requirements(('tensorflow>=2.4.1',))
            import tensorflow as tf
            if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                def wrap_frozen_graph(gd, inputs, outputs):
                    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                    return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                                tf.nest.map_structure(x.graph.as_graph_element, outputs))

                graph_def = tf.Graph().as_graph_def()
                graph_def.ParseFromString(open(w, 'rb').read())
                frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
            elif saved_model:
                model = tf.keras.models.load_model(w)
            elif tflite:
                interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
                interpreter.allocate_tensors()  # allocate
                input_details = interpreter.get_input_details()  # inputs
                output_details = interpreter.get_output_details()  # outputs
                int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
        except ImportError:
            from pycoral.adapters import common
            from pycoral.adapters import detect
            from pycoral.utils.dataset import read_label_file
            from pycoral.utils.edgetpu import make_interpreter

            interpreter = make_Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model

    imgsz = check_img_size(imgsz, s=stride)  # check image size
    ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)

    # Dataloader
    if webcam:
        view_img = check_imshow()
        if not khadas:
            cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, auto_size=stride, auto=auto)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, auto_size=stride, auto=auto)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        if onnx:
            # img = img.numpy()
            img = img.astype('float32')
            # img = torch.from_numpy(img).to(device)
        elif opencv_onnx:
            img = img.astype('uint8')
            # print(img.shape, type(img), img.dtype)
            # img = np.reshape(img, (640,640,3))
            img = img.transpose((1, 2, 0)) # 640x640x3
            # print(img.shape, type(img), img.dtype)
        elif trt:
            img = img.numpy()
            img = img.astype('float16')
        elif khadas:
            #print(img)
            #print(type(img))
            #if type(img) is np.array:
            #    print("hi")
            #    img = img.numpy()
            img = img.astype('float32')  
        elif saved_model:
            img = img.numpy()
            img = img.astype('float32') # it is expecting a float 32 argument
        else:
            # img = img.to(device, non_blocking=True)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        if opencv_onnx:
            img = cv2.dnn.blobFromImage(img, 1/255.0) # takes input such as 640x640x3
        else:
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time_sync()
        if pt or darknet:
            if pt_jit:
                pred = model(img)[0]
            else:
                # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(img, augment=augment)[0]
        elif onnx:
            pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
            # img size is normally dynamic allow for inputs such as 1,3,512,640 however if it is static input then turn off auto in the dataset so it will always give 1,3,640,640
            # pred = torch.tensor(session.run(None, {session.get_inputs()[0].name: img}))
            if opt.device == "0": 
                pred = pred.to(device)
        elif opencv_onnx:
            model.setInput(img)
            pred = torch.tensor(model.forward(output_names))
        elif trt:
            pred = torch.tensor(model.run(img))
            if opt.device == "0": 
                pred = pred.to(device)
        elif khadas:
            from ksnn.types import output_format
            cv_img = list()
            resize_img = cv2.resize(im0s, (imgsz[0], imgsz[0]))
            cv_img.append(resize_img)
            # cv_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # converts img from numpy to opencv array format
            pred = np.array([yolo.nn_inference(cv_img, platform='DARKNET', reorder='2 1 0', output_tensor=3, output_format=output_format.OUT_FORMAT_FLOAT32)])
        else:  # tensorflow model (tflite, pb, saved_model)
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()
            elif saved_model:
                pred = model(imn, training=False).numpy()
            elif tflite:
                if int8:
                    scale, zero_point = input_details[0]['quantization']
                    imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                interpreter.set_tensor(input_details[0]['index'], imn)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                if int8:
                    scale, zero_point = output_details[0]['quantization']
                    pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            pred = torch.tensor(pred)

        # NMS
        if khadas: 
            from khadas_post_process.yolov4_process import yolov4_post_process, draw
            img_size_orginal = im0s.shape[0:2]
            output = yolov4_post_process(pred, img_size=img_size_orginal, OBJ_THRESH=conf_thres, NMS_THRESH=iou_thres, MAX_BOXES=max_det)
        else:
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        # pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
            if khadas:
                if output is not None and len(output[0]) != 0:
                    outputs = output[0].tolist()
                    outputs = np.array(outputs)
                    boxes = outputs[:, :4]
                    scores = outputs[:,4]
                    classes = outputs[:,5]
                    classes_int = classes.astype(int)
                    
                    list_of_images = draw(resize_img, boxes, scores, classes_int)

                    if save_txt:
                        for box, score, class_int in zip(boxes, scores, classes_int):
                            line = (class_int, box, score) if save_conf else (class_int, box)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write((f'{line}') + '\n')
                    if save_img:
                        for image_number in range(int(len(list_of_images)/3)):
                            name_of_results = ["target", "target_location", "all_targets"]
                            for value, data in enumerate(name_of_results):
                                if not save_crop:
                                    if value == 0 or value == 1:
                                        continue
                                    # else:
                                        # image_number = 0
                                image_name = f"{save_path}_{data}_{image_number}.jpg"
                                image = list_of_images[value]
                                if image is not None:
                                    cv2.imwrite(image_name, image)

            elif len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    if onnx or tflite:
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if khadas:
                if len(list_of_images) > 0:
                    number_of_set_of_images = len(list_of_images)
                    im0 = list_of_images[number_of_set_of_images - 1]
            else:
                im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img and not khadas:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    t = tuple(x / len(dataset) * 1E3 for x in (t1, t2, t1 + t2)) + (imgsz, imgsz, bs)  # tuple
    print(f'Done. ({time.time() - t0:.3f}s)')
    with open(save_dir / 'information.txt', 'a') as f:
        f.write(('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g') % t + '\n\n')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov4.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--library', type=str, default='', help='the library made with khadas converter')
    parser.add_argument('--auto', action='store_true', help='Turn this on if you using dynamic inputs but if you are using static inputs then turn this False')
    parser.add_argument('--opencv_onnx', default=False, action='store_true', help='are ypu using opencv to load onnx models?')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
