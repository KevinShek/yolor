# Author's Note
The "paper" branch of the repository was used to train the YOLOR-P6 and YOLOv4-CSP models for Chapter 5 of my [thesis](https://pure.qub.ac.uk/en/studentTheses/artificial-intelligence-based-image-recognition-using-limited-res).

The main modification of the "paper" branch of the repository is the "detection_script.py" script that is might for Chapter 5 that is used for inferencing the YOLOv4 variant models that were trained on different Machine Learning framework. While also made to work with the Vim3Pro Single Board Computer which performs quickly as it uses an NPU if you are aimming for real-time operation as seen in Chapter 6. However, I would recommand using a YOLO model that the team of Khadas has made a process for handling the output data as seen in the following [repository](https://github.com/khadas/ksnn/tree/master/examples) if you are looking for an out of the box solution otherwise you need to make your own process for handling the output data from the chosen trained model. An attempt was made of this process for YOLOv4-CSP model, however, the detection was seen successful if the confidnece threshold was above 0.37 which was the reason of the recommandation. However, for the task it was used for it was successful.

UPDATE (06/02/2026): There was an attempt to refactor the script of "detection_script.py" leading to "detection_script_v2.py" that has fixed the issue with YOLOv4-CSP model implementation as seen in "yolov4_process_updated.py" and is also working with the Annotator class.

## Generic Inference

```
python detection_script.py --weights ../weights/pytorch/yolov4-csp-640.pt --source 0 --img-size 640 --device 0 --name yolov4_csp_640
```

## Inference that was used for the paper

```
python detection_script.py --weights ../weights/yolov4-csp-all-environment-640/yolov4-csp-all-environment-640.nb --source 0 --imgsz 640 --conf-thres 0.37 --iou-thres 0.3 --max-det 300 --library ../weights/yolov4-csp-all-environment-640/libnn_yolov4-csp-all-environment-640.so --save-txt
```

## Citation
If you wish to cite this, then please cite the [thesis](https://pure.qub.ac.uk/en/studentTheses/artificial-intelligence-based-image-recognition-using-limited-res). Thank you.

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/khadas/ksnn/tree/master](https://github.com/khadas/ksnn/tree/master)

</details>
