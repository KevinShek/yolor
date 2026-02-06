import os
import cv2
import numpy as np
from fractions import Fraction
from configparser import ConfigParser
import os
import pickle
        
# for camera calbration if you have the matrix needed to calbrate the camera using opencv
def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)
        
class Settings:
    def __init__(self):
        # camera setting
        self.width = 1920 # 3840
        self.height = 1080 # 2160
        self.cali = True # pass by the calibrate
        self.ready_check = True # requires the user to press a button "Enter" to start the rest of the code
        self.framerate = 10 # 5
        self.angle = "upsidedown" # clockwise, counterclockwise, upsidedown
        self.flip_image = True # khadas camera needs to be fliped
        self.calbrate_distort_camera_path = "OS08A10_distorted_images"
        self.distorted_camera = True
        if self.distorted_camera:
            self.mtx = load_object("utils/khadas_camera_parameters/mtx.pickle")
            self.dist = load_object("utils/khadas_camera_parameters/dist.pickle")
            self.rvecs = load_object("utils/khadas_camera_parameters/rvecs.pickle")
            self.tvecs = load_object("utils/khadas_camera_parameters/tvecs.pickle")

