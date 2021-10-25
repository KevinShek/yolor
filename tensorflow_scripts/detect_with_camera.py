import numpy as np
import argparse
import tensorflow as tf
import cv2
import time
from datetime import datetime
from pathlib import Path
import glob
import re
import os
import itertools
from alive_progress import alive_bar
from interfacing_with_mmdetection.saving import Saving
from interfacing_with_mmdetection.fps_calculator import Fps

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

# this will initialise the GPUs you have.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# the following option is for limiting the memory usage by tensorflow
# if gpus:
#     try:
#         tf.config.experimental.set_virtual_device_configuration(gpu[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#     except RuntimeError as e:
#         print(e)
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [
#     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])


def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model


def load_tflite_model(model_path):
    model = tf.lite.Interpreter(model_path)
    return model


def run_inference_for_single_image_tflite(model, image):
    model.allocate_tensors()
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    output_dict = {}
    
    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32 

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2] 
    image = cv2.resize(image, (width, height))

    # add N dim
    input_data = np.expand_dims(image, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    model.set_tensor(input_details[0]['index'], input_data)

    model.invoke()


    detection_scores = np.squeeze(model.get_tensor(output_details[0]['index']))
    detection_boxes = np.squeeze(model.get_tensor(output_details[1]['index']))
    number_of_bbox = int(np.squeeze(model.get_tensor(output_details[2]['index'])))
    detection_classes = np.squeeze(model.get_tensor(output_details[3]['index']))

    output_dict["detection_scores"] = detection_scores
    output_dict["detection_boxes"] = detection_boxes
    output_dict["detection_classes"] = detection_classes

    # converting float into a int
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # print(output_dict)
    # print(list(output_dict))
    # print(f"score: {detection_scores}")
    # print(f"bbox: {detection_boxes}")
    # print(f"number_of_bbox: {number_of_bbox}")
    # print(f"classes: {detection_classes}")
    return output_dict



def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


# def increment_path(path, exist_ok=True, sep=''):
#     # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
#     path = Path(path)  # os-agnostic
#     if (path.exists() and exist_ok) or (not path.exists()):
#         return str(path)
#     else:
#         dirs = glob.glob(f"{path}{sep}*")  # similar paths
#         matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
#         i = [int(m.groups()[0]) for m in matches if m]  # indices
#         n = max(i) + 1 if i else 2  # increment number
#         return f"{path}{sep}{n}"  # update path


def run_inference(model, category_index, cap, args):
    # initial value to calculate frame rates
    fps = Fps()
    testing = str(args.testing)

    # # Directories
    # save_dir = Path(increment_path(Path("result_val") / testing, exist_ok=False))  # increment run
    # (save_dir / 'labels' if False else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    save = Saving(testing)

    if not testing == "No":
        print("test is starting")     
    else:
        print("starting detection")

    if args.video_footage:

        if args.saving:
            save.name_of_output_for_video(testing, cap, args.is_video)

        while True:
            fps.start_time()
            ret, image_np = cap.read()

            # testing frame rate with video
            if not ret:
                fps.average(save.save_dir)
                if args.saving:
                    save.video.release()
                cap.release()
                print("Finish Detection")
                break

            # Actual detection.
            if args.is_tflite:
                output_dict = run_inference_for_single_image_tflite(model, image_np)
            else:
                output_dict = run_inference_for_single_image(model, image_np)

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                line_thickness=4)

            # Timing after the detection and applying to frame and calculation
            fps.calculate()

            if not testing == "No":
                fps_number = str(int(fps.current_fps))
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image_np, fps_number, (7, 70), font, 3, (100, 255, 0), 1, cv2.LINE_AA)
                # cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    print("stopping detection")
                    cap.release()
                    cv2.destroyAllWindows()
                    break

            if args.saving:
                save.video.write(image_np)

        cap.release()
    
    else:
        # a progress bar was added to see and provide an eta on how long the progress would be
        with alive_bar(len(cap), bar = 'smooth', spinner = 'fish') as bar:
            for name in cap:
                fps.start_time()
                # local path
                # script_dir = "/projects/yolor/coco/images/val2017"
                script_dir = os.path.split(os.path.realpath(__file__))[0] # <-- absolute dir the script is in
                abs_file_path = os.path.join(script_dir, name)  # attaching the location
                image_np = cv2.imread(abs_file_path) # the location of the image

                # Actual detection.
                if args.is_tflite:
                    output_dict = run_inference_for_single_image_tflite(model, image_np)
                else:
                    output_dict = run_inference_for_single_image(model, image_np)
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    instance_masks=output_dict.get('detection_masks_reframed', None),
                    use_normalized_coordinates=True,
                    line_thickness=4)

                # save the image
                if args.saving:
                    save.save_the_image(name, image_np)

                # Timing after the detection and applying to frame and calculation
                fps.calculate()
                bar()

            fps.average(save.save_dir)
            print("Finish Detection")


# def stream(size):
#     # sending data to a tcp protocol
#     pipeline = "appsrc ! video/x-raw, format=BGR ! queue ! videoconvert ! x264enc noise-reduction=10 speed-preset=ultrafast tune=zerolatency ! h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host=192.168.0.46 port=8080 sync=true"
#     framerate = 25
#     video = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, framerate, size, True)
#     return video


# def set_saved_video(input_video, output_video, size):
#     # for mp4 use MP4V for avi use XVID
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     fps = int(input_video.get(cv2.CAP_PROP_FPS))
#     video = cv2.VideoWriter(output_video, fourcc, fps, size)
#     return video


def main():
    parser = argparse.ArgumentParser(description='Detect objects inside webcam videostream')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model Path')
    parser.add_argument('-l', '--labelmap', type=str, required=True, help='Path to Labelmap')
    parser.add_argument('-v', '--testing', type=str, required=False, default="No",
                        help='do you want to do testing? if Yes provide "NAME_OF_MODEL" else "No"')
    parser.add_argument('-b', '--media', type=str, required=False, default="video/carpark.mp4",
                        help='location of video')
    parser.add_argument('-f', '--video_footage', action='store_true',
                        help='are you using video footage (this includes live or prerecorded)?')
    parser.add_argument('-s', '--saving', action='store_true', help='enable to save media')
    parser.add_argument('-vid', '--is_video', action='store_true', help='is it a video?')
    parser.add_argument('-tf', '--is_tflite', action='store_true', help='is it a tflite?')
    parser.add_argument('-i', '--images', type=str, required=False,
                        help='path to the dir of where the images you want to test')

    args = parser.parse_args()
    print(args)

    if args.is_tflite:
        detection_model = load_tflite_model(args.model)
    else:
        detection_model = load_model(args.model)

    category_index = label_map_util.create_category_index_from_labelmap(args.labelmap, use_display_name=True)
    if args.video_footage:
            cap = cv2.VideoCapture(f"{args.media}")

    else:
        # retrieves the data's path to where the images are and locating specifically .jpg images
        # image_dir = "../../../../yolor/coco/images/val2017"
        cap = []
        data_dir = Path(args.media)

        # the following code interite over the extension that exist within a folder and place them into a single list
        image_count = list(itertools.chain.from_iterable(data_dir.glob(pattern) for pattern in ('*.jpg', '*.png')))
        # image_count = len(list(data_dir.glob('*.jpg')))
        for name in image_count:
                # head, tail = ntpath.split(name)
                filename = Path(name)  # .stem removes the extension and .name grabs the filename with extension
                cap.append(filename)
        print(f"there is a total image count of {len(image_count)} and frames appended {len(cap)}")

        if len(image_count) == 0:
            print("there is no image")
            return 


    run_inference(detection_model, category_index, cap, args)


if __name__ == '__main__':
    main()