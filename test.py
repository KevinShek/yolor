import argparse
import glob
import json
import os
from pathlib import Path
import cv2

# This call to matplotlib.use() has no effect because the backend has already
# been chosen; matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
# or matplotlib.backends is imported for the first time.
import matplotlib
matplotlib.use('Agg')  # for writing to files only

nano = False

from matplotlib.pyplot import pause

import numpy as np
import torch
from torch.utils.data import dataset
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader, LoadImages
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, box_iou, check_suffix, \
    non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, clip_coords, set_logging, increment_path
from utils.loss import compute_loss
from utils.metrics import ap_per_class
from utils.plots import plot_images, output_to_target
from utils.torch_utils import select_device, time_synchronized, load_classifier

from models.models import *

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def test(data,
         weights=None,
         batch_size=16,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_image=False, # for saving labelled images
         save_conf=False,
         plots=True,
         log_imgs=0, # number of logged images
         library=None):  

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
        pt, onnx, tflite, pb, saved_model, pt_jit, trt, khadas = True, False, False, False, False, False, False, False 

    else:  # called directly
        set_logging()
        save_txt = opt.save_txt  # save *.txt labels

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        if save_image:
            save_dir_image = Path(increment_path(Path(opt.project) / opt.name / "images", exist_ok=opt.exist_ok))  # increment run
            (save_dir_image).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        # model = attempt_load(weights, map_location=device)  # load FP32 model
        w = weights[0] if isinstance(weights, list) else weights
        classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '.trt', '.nb', '', '.weights']
        check_suffix(w, suffixes)  # check weights have acceptable suffix
        pt, onnx, tflite, pb, trt, khadas, saved_model, darknet  = (suffix == x for x in suffixes)  # backend booleans
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        pt_jit = pt and 'torchscript' in w
        auto = False
        if khadas:
            opt.device = "cpu"
        device = select_device(opt.device, batch_size=batch_size)
        if pt or darknet:
            if pt_jit:
                import json
                extra_files = {'config.txt': ''}
                model = torch.jit.load(w, _extra_files=extra_files)
                try:
                    d = json.loads(extra_files['config.txt'])  # extra_files dict
                    ts_h, ts_w = d['HW']
                    ts_bs = int(d['BATCH'])
                    stride = int(d['STRIDE'])
                    ts_params = True  # torchscript predefined params
                    print("Using saved graph params HW: {}, stride: {}".format((ts_h, ts_w), stride))
                except:
                    ts_params = False
                    print("Failed to load default jit graph params from {}".format(w))
            else:
                try:
                    model = attempt_load(weights, map_location=device)  # load FP32 model
                    stride = int(model.stride.max())  # model stride
                    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
                    if classify:  # second-stage classifier
                        modelc = load_classifier(name='resnet50', n=2)  # initialize
                        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
                except:
                    # Load model
                    model = Darknet(opt.cfg).to(device)

                    # load weights
                    try:
                        ckpt = torch.load(w, map_location=device)  # load checkpoint
                        ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
                        model.load_state_dict(ckpt['model'], strict=False)
                    except:
                        load_darknet_weights(model, w)

        elif onnx:
            # check_requirements(('onnx', 'onnxruntime'))
            import onnxruntime

            if opt.device == "0":
                providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider',
                ]
                session = onnxruntime.InferenceSession(w, providers=providers)
                print("device 0 is selected")
            else:
                session = onnxruntime.InferenceSession(w, None)
            names = names
        
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
        
        else: # Tensorflow
            if opt.device == "0":
                import tensorflow as tf
                physical_devices = tf.config.list_physical_devices('GPU')
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                # tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [
                #     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]) # this will limit your GPU's memory allowance
            if saved_model:  # SavedModel
                auto = False
                print(f'Loading {w} for TensorFlow SavedModel inference...')
                import tensorflow as tf
                model = tf.keras.models.load_model(w)
                # model = tf.saved_model.load(w)
                # inference = model.signatures["serving_default"]
                # print(inference)
            elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                print(f'Loading {w} for TensorFlow GraphDef inference...')
                import tensorflow as tf

                def wrap_frozen_graph(gd, inputs, outputs):
                    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                    return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                                   tf.nest.map_structure(x.graph.as_graph_element, outputs))

                graph_def = tf.Graph().as_graph_def()
                graph_def.ParseFromString(open(w, 'rb').read())
                frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
            elif tflite:
                try:
                    import tflite_runtime.interpreter as tfl  # prefer tflite_runtime if installed
                except ImportError:
                    import tensorflow.lite as tfl

                w = "/projects" + w.strip("..")
                print(f'Loading {w} for TensorFlow Lite inference...')
                interpreter = tfl.Interpreter(model_path=w)  # load TFLite model
                interpreter.allocate_tensors()  # allocate
                input_details = interpreter.get_input_details()  # inputs
                output_details = interpreter.get_output_details()  # outputs

        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
    if pt or darknet:
        # Half
        half = device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            model.half()
    else:
        half = False

    # Configure
    # model.eval()
    is_coco = data.endswith('coco.yaml')  # is COCO dataset
    is_coco_format = data.endswith('coco_format.yaml')  # is COCO dataset
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    # check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs, wandb = min(log_imgs, 100), None  # ceil
    try:
        import wandb  # Weights & Biases
    except ImportError:
        log_imgs = 0

    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        # _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
        # dataloader = create_dataloader(path, imgsz, batch_size, model.stride.max(), opt, pad=0.5, rect=True)[0]
        dataloader, dataset = create_dataloader(path, imgsz, batch_size, stride, opt, pad=0.5, rect=True, auto=auto)

    # Dataset
    # dataset = LoadImages(path, img_size=imgsz, auto_size=64)

    seen = 0
    try:
        names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    except:
        names = load_classes(opt.names)
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
    # img = img.to(device, non_blocking=True)
    # img = img.half() if half else img.float()  # uint8 to fp16/32
        if onnx:
            img = img.numpy()
            img = img.astype('float32')
            # img = torch.from_numpy(img).to(device)
        elif trt:
            img = img.numpy()
            img = img.astype('float16')
        elif khadas:
            img = img.numpy()
            img = img.astype('float32')  
        elif saved_model:
            img = img.numpy()
            img = img.astype('float32') # it is expecting a float 32 argument
        else:
            img = img.to(device, non_blocking=True)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        # height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = time_synchronized()
            if pt or darknet:
                if pt_jit:
                    inf_out, train_out = model(img)[0:2]
                else:
                    inf_out, train_out = model(img, augment=augment)[0:2]
            elif onnx:
                inf_out = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
                if opt.device == "0": 
                    inf_out = inf_out.to(device)
            elif trt:
                inf_out = torch.tensor(model.run(img))
                if opt.device == "0": 
                    inf_out = inf_out.to(device)
            elif khadas:
                from ksnn.types import output_format
                cv_img_path = "/projects" + self.img_files[index].strip("..")
                cv_img = cv2.imread(cv_img_path, cv2.IMREAD_COLOR)
                cv_img = cv.resize(cv_img, (imgsz, imgsz))
                # cv_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # converts img from numpy to opencv array format
                inf_out = np.array([yolo.nn_inference(cv_img, platform='DARKNET', reorder='2 1 0', output_tensor=3, output_format=output_format.OUT_FORMAT_FLOAT32)])
            elif pb or saved_model:

                inf_out = model(**{'input': img})
                inf_out = torch.tensor(inf_out['output'].numpy())
                if opt.device == "0": 
                    inf_out = inf_out.to(device)
            elif tflite:
                interpreter.resize_tensor_input(input_details[0]['index'], (1, 3, imgsz, imgsz))
                interpreter.allocate_tensors()
                interpreter.set_tensor(input_details[0]['index'], img)
                interpreter.invoke()
                inf_out = torch.tensor(interpreter.get_tensor(output_details[0]['index']))
                if opt.device == "0": 
                    inf_out = inf_out.to(device)

            # print(inf_out.shape)
            # inf_out, train_out = model(img, augment=augment)  # inference and training outputs
            
            t0 += time_synchronized() - t

            # Compute loss
            if training:  # if model has loss hyperparameters
                loss += compute_loss([x.float() for x in train_out], targets, model)[1][:3]  # box, obj, cls

            # Run NMS
            t = time_synchronized()
            if khadas: 
                from khadas_post_process.yolov4_process import yolov4_post_process
                output = yolov4_post_process(inf_out, OBJ_THRESH=conf_thres, NMS_THRESH=iou_thres)
            else:
                output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)
            t1 += time_synchronized() - t  

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            path = Path(paths[si])
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                x = pred.clone()
                # print("xyxy =", x[:, :4])
                x[:, :4] = scale_coords(img[si].shape[1:], x[:, :4], shapes[si][0], shapes[si][1])  # to original
                for *xyxy, conf, cls in x:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # print("xywh =", xywh)
                    # print("conf =", conf)
                    # print("cls =", cls)
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging
            if plots and len(wandb_images) < log_imgs:
                box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                            "class_id": int(cls),
                            "box_caption": "%s %.3f" % (names[cls], conf),
                            "scores": {"class_score": conf},
                            "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                boxes = {"predictions": {"box_data": box_data, "class_labels": names}}
                wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))

            # Clip boxes to image bounds
            clip_coords(pred, (height, width)) # pred is expected to be a list with a format of (N,6) where N is the number of prediction

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = pred[:, :4].clone()  # xyxy
                scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                'bbox': [round(x, 3) for x in b],
                                'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]
                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices
                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # filename
            plot_images(img, targets, paths, f, names, max_size=imgsz)  # labels
            f = save_dir / f'test_batch{batch_i}_pred.jpg'
            plot_images(img, output_to_target(output, width, height), paths, f, names, max_size=imgsz)  # predictions
        
        if save_image:
            # o_img = cv2.imread(paths[0])  # BGR
            # # rgb_img = o_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            # img0 = o_img.transpose(2, 0, 1)
            # img0 = img0[None] # makes a batch dim of 1 example (channel, height, width) to (batch dim, channel, height, width)
            f = save_dir_image / f'test_batch{batch_i}_pred.jpg'
            plot_images(img, output_to_target(output, width, height), paths, f, names)  # predictions

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        if nano:
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, fname=save_dir / 'precision-recall_curve.png')
        else:            
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, fname=save_dir / 'precision-recall_curve.png')
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # W&B logging
    if plots and wandb:
        wandb.init(resume="allow",
                project='YOLOR-test',
                name=save_dir.stem,
                id= None)
        wandb.log({"Images": wandb_images})
        wandb.log({"Validation": [wandb.Image(str(x), caption=x.name) for x in sorted(save_dir.glob('test*.jpg'))]})

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if verbose and len(stats):
        with open(save_dir / 'information.txt', 'a') as f:
            f.write(('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95') + '\n')
            f.write((pf) % ('all', seen, nt.sum(), mp, mr, map50, map) + '\n')
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
            with open(save_dir / 'information.txt', 'a') as f:
                f.write((pf) % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]) + '\n')

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)
        with open(save_dir / 'information.txt', 'a') as f:
            f.write(('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g') % t + '\n\n')

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        if is_coco:
            anno_json = glob.glob('../coco/annotations/instances_val*.json')[0]  # annotations json
        else:
            anno_json = glob.glob('../heridal/testImages/labels/labels.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        print([Path(x).stem for x in dataloader.dataset.img_files])

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            else:
                eval.params.imgIds = [Path(x).stem for x in dataloader.dataset.img_files]  # image IDs to evaluate (has to be in int)
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)

            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3} ] = {:0.3}'
            titleStr = ["Average Precision", "Average Precision", "Average Precision", "Average Precision", "Average Precision", "Average Precision", 
                        "Average Recall", "Average Recall", "Average Recall", "Average Recall", "Average Recall", "Average Recall"]
            iouStr = ["0.50:0.95", "0.50", "0.75", "0.50:0.95", "0.50:0.95", "0.50:0.95", "0.50:0.95", "0.50:0.95", "0.50:0.95", "0.50:0.95", 
                        "0.50:0.95", "0.50:0.95"]
            typeStr = ["(AP)", "(AP)", "(AP)", "(AP)", "(AP)", "(AP)", "(AR)", "(AR)", "(AR)", "(AR)", "(AR)", "(AR)"]
            areaRng = ["all", "all", "all", "small", "medium", "large", "all", "all", "all", "small", "medium", "large"]
            maxDets = ["100", "100", "100", "100", "100", "100", "1", "100", "100", "100", "100", "100"]


            with open(save_dir / 'information.txt', 'a') as f:
                for i, c in enumerate(eval.stats):
                    f.write(iStr.format(titleStr[i], typeStr[i], iouStr[i], areaRng[i], maxDets[i], c) + '\n')

        except Exception as e:
            print('ERROR: pycocotools unable to run: %s' % e)

    # Return results
    if not training:
        print('Results saved to %s' % save_dir)
    if pt:
        model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolor-p6.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save_img', action='store_true', help='save image to as jpeg')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*dataset class names path')
    parser.add_argument('--cfg', type=str, default='cfg/yolor_p6.cfg', help='*.cfg path')
    parser.add_argument('--library', type=str, default='', help='the library made with khadas converter')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)

    if opt.task in ['val', 'test']:  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt,
             save_image=opt.save_img,
             save_conf=opt.save_conf,
             library=opt.library,
             )

    elif opt.task == 'study':  # run over a range of settings and save/plot
        for weights in ['yolor-p6.pt', 'yolor-w6.pt', 'yolor-e6.pt', 'yolor-d6.pt']:
            f = 'study_%s_%s.txt' % (Path(opt.data).stem, Path(weights).stem)  # filename to save to
            x = list(range(320, 800, 64))  # x axis
            y = []  # y axis
            for i in x:  # img-size
                print('\nRunning %s point %s...' % (f, i))
                r, _, t = test(opt.data, weights, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        # utils.general.plot_study_txt(f, x)  # plot
