import numpy as np
import cv2
from pathlib import Path
import itertools
from tqdm import tqdm
import pickle
        

## calbrating the distorted camera based on saved images for calbration 
def calbrate_distorted_camera_based_on_images(calbrate_distort_camera_path):
    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    obj_points = np.zeros((6*9,3), np.float32)
    obj_points[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    real_points = []
    img_points = []
    chess_images_path = Path(calbrate_distort_camera_path)
    # print(chess_images_path)
    chess_images_count = list(itertools.chain.from_iterable(chess_images_path.glob(pattern) for pattern in ('*.jpg', '*.png')))
    # print(chess_images_count)
    for name in tqdm(chess_images_count):
        chess_img = cv2.imread(str(name))
        chess_gray = cv2.cvtColor(chess_img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(chess_gray, (9,6), None)
        if ret == True:
            real_points.append(obj_points)
            corners2 = cv2.cornerSubPix(chess_gray,corners, (11,11), (-1,-1), term_criteria)
            img_points.append(corners)
            #cv2.drawChessboardCorners(chess_img, (7,6), corners2, ret)
            #cv2.imshow('img', chess_img)
            #cv2.waitKey(0)
        else:
            print(str(name))
    
    print("camera is now calbrated")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(real_points, img_points, chess_gray.shape[::-1], None, None)

    return mtx, dist, rvecs, tvecs

## undistorting the camera
def undistort_camera(mtx, dist, rvecs, tvecs, frame):
    # img = cv2.imread(str(Path('OS08A10_distorted_images/21.png')))
    
    h,  w = frame.shape[:2]
    #print(h, w)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    #print(roi)
    h,  w = frame.shape[:2]
    
    dst = dst[0:h, 0:w]
    
    
    return dst
        
     
def save_object(name, obj):
    try:
        with open(f"{name}.pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)
 
 
def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)
 

if __name__ == "__main__":
    calbrate_distort_camera_path = Path("OS08A10_distorted_images")
    img = cv2.imread(str(Path('OS08A10_distorted_images/21.png')))
    mtx, dist, rvecs, tvecs = calbrate_distorted_camera_based_on_images(calbrate_distort_camera_path)
    
    for obj, name in zip([mtx, dist, rvecs, tvecs], ["mtx", "dist", "rvecs", "tvecs"]):
        save_object(name, obj)
    #mtx_text = f"mtx = {mtx}"
    #dist_text = f"dist = {dist}"
    #rvecs_text = f"rvecs = {rvecs}"
    #tvecs_text = f"tvecs = {tvecs}"

    #with open(f'information.txt', 'a') as f:
        #for text in [mtx_text, dist_text, rvecs_text, tvecs_text]:
            #f.write(f"{text}\n")
    dst = undistort_camera(mtx, dist, rvecs, tvecs, img)
    cv2.imshow('Undistorted Image', dst)
    cv2.imshow('distorted image', img)
    cv2.waitKey(0)
    input("ready")
    cv2.destroyAllWindows()
