import argparse
import pathlib
import ntpath
import cv2
import os
import tensorflow as tf
import numpy as np
import time
import json
import pandas as pd

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])


def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model


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


def run_inference(model, category_index, test_images_np, testing):
    # initial value to calculate frame rates
    previous_frame_time = time.time()
    new_frame_time = time.time()
    font = cv2.FONT_HERSHEY_SIMPLEX
    average_fps_list = []
    average_speed_list = []

    if not testing == "N":
        print("test is starting")
    else:
        print("starting detection")

    for name in test_images_np:
        # local path
        script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
        abs_file_path = os.path.join(script_dir, name)  # attaching the location
        image_np = cv2.imread(abs_file_path)

        # Actual detection.
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

        if not testing == "N":
            file_name = pathlib.Path(abs_file_path).name
            name_no_ext = pathlib.Path(abs_file_path).stem
            # write_to_json(name_no_ext, output_dict)
            write_to_file(testing, file_name, image_np)
            write_to_csv(name_no_ext, testing, output_dict, image_np)

        if args.visual == "Y":
            fps = str(int(fps))
            cv2.putText(image_np, fps, (7, 70), font, 3, (100, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                print("stopping detection")
                cv2.destroyAllWindows()
                break

        # Timing after the detection and applying to frame and calculation
        new_frame_time = time.time()
        average_speed_list.append(new_frame_time - previous_frame_time)
        fps = 1 / (new_frame_time - previous_frame_time)
        average_fps_list.append(fps)
        previous_frame_time = new_frame_time

        # testing frame rate with video
    average_fps = sum(average_fps_list) / len(average_fps_list)
    average_speed = sum(average_speed_list) / len(average_speed_list)
    current_time = time.time()
    with open (f"result_val/{testing}_{current_time}.txt", "w+") as f:
        f.write(f"Title: {testing}\r\n")
        f.write(f"Average_fps = {average_fps}\r\n")
        f.write(f"Average_speed = {average_speed}\r\n")
    print("test is finish")

    return


def write_to_csv(file_name, directory, output, image_np):
    xmin = []
    ymin = []
    xmax = []
    ymax = []

    # object detection of detection boxes are the percentages of the bounding box instead of pixel value
    # https://stackoverflow.com/questions/48915003/get-the-bounding-box-coordinates-in-the-tensorflow-object-detection-api-tutorial
    width, height = image_np.shape[:2]

    for count, _ in enumerate(output['detection_boxes']):
        xmin.append(int(output['detection_boxes'][count][0] * width))
        ymin.append(int(output['detection_boxes'][count][1] * height))
        xmax.append(int(output['detection_boxes'][count][2] * width))
        ymax.append(int(output['detection_boxes'][count][3] * height))

    prediction = pd.DataFrame(output, columns=['detection_classes', 'detection_scores'])
    list_of_filename = []
    for count, _ in enumerate(output['detection_scores']):
        list_of_filename.append(file_name)

    prediction.insert(0, "image_id", list_of_filename, True)
    prediction.insert(1, "xmin", xmin, True)
    prediction.insert(2, "ymin", ymin, True)
    prediction.insert(3, "xmax", xmax, True)
    prediction.insert(4, "ymax", ymax, True)
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    pred_name = os.path.join(directory, "prediction.csv")
    path_result = os.path.join(script_dir, pred_name)
    prediction.to_csv(path_result, mode='a', header=False)

# def write_to_json(file_name, output):
#     print(file_name)
#     print(output['detection_boxes'][0])
#     print(output['detection_classes'][0])
#     print(output['detection_scores'][0])
#
#     annotation = []
#     categories = []
#
#     annotations = {}
#
#     annotations["image_id"] = file_name
#     for number in enumerate(output['detection_classes']):
#         annotations["bbox"] = output['detection_boxes'][number]
#         annotations["iscrowd"] = 0
#         annotations
#
#     category = {}
#
#     category["supercategory"] = 'none'
#     category["id"] = 0
#     category["name"] = 'None'
#     categories.append(category)


def write_to_file(directory, name, image):
    """
  For writing an image file to the specified directory with standardised file name format
  - directory - name folder to save file
  - name of file
  - image - saving the image of interest
  """
    if directory is not None:
        # Make sure the directory exists
        script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
        path_result = os.path.join(script_dir, directory)
        if not os.path.exists(path_result):
            os.makedirs(path_result)

        # Form file name
        file_name = f"{name}"

        # Form full path
        filepath = os.path.join(path_result, file_name)

        # Write file
        cv2.imwrite(filepath, image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects inside webcam videostream')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model Path')
    parser.add_argument('-v', '--testing', type=str, required=False, default="No",
                        help='do you want to do testing? if Yes provide "NAME_OF_MODEL" else "N"')
    parser.add_argument('-l', '--labelmap', type=str, required=True, help='Path to Labelmap')
    parser.add_argument('-i', '--images', type=str, required=False,
                        help='path to the dir of where the images you want to test')
    # parser.add_argument('-s', '--saving', type=str, required=True, help='do you want to save? ("Y" or "N")')
    parser.add_argument('-vi', '--visual', type=str, required=False, default="N", help='do you want to save? ("Y" or "N")')

    args = parser.parse_args()

    detection_model = load_model(args.model)
    category_index = label_map_util.create_category_index_from_labelmap(args.labelmap, use_display_name=True)

    testing = args.testing
    image_dir = args.images
    test_images_np = []
    data_dir = pathlib.Path(image_dir)
    image_count = len(list(data_dir.glob('*.jpg')))
    for name in data_dir.glob('*.jpg'):
        # head, tail = ntpath.split(name)
        filename = pathlib.Path(name)  # .stem removes the extension and .name grabs the filename with extension
        test_images_np.append(filename)

    print("there is a total image count of ", f"{image_count}")

    run_inference(detection_model, category_index, test_images_np, testing)




